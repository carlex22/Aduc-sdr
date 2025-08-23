#--- START OF MODIFIED FILE app_fluxContext_Ltx/flux_kontext_helpers.py ---
# flux_kontext_helpers.py
# Módulo de serviço para o FluxKontext, com gestão de memória e revezamento de GPU.
# Este arquivo é parte do projeto Euia-AducSdr e está sob a licença AGPL v3.
# Copyright (C) 4 de Agosto de 2025  Carlos Rodrigues dos Santos

import torch
from PIL import Image
import gc
from diffusers import FluxKontextPipeline
import huggingface_hub
import os
import threading

class FluxWorker:
    """
    Representa uma única instância do pipeline FluxKontext, associada a uma GPU específica.
    O pipeline é carregado na CPU por padrão e movido para a GPU sob demanda.
    """
    def __init__(self, device_id='cuda:0'):
        self.cpu_device = torch.device('cpu')
        self.device = torch.device(device_id if torch.cuda.is_available() else 'cpu')
        print(f"FLUX Worker: Inicializando para o dispositivo {self.device} (carregando na CPU)...")
        self.pipe = None
        self._load_pipe_to_cpu()

    def _load_pipe_to_cpu(self):
        if self.pipe is None:
            print("FLUX Worker: Carregando modelo FluxKontext para a CPU...")
            self.pipe = FluxKontextPipeline.from_pretrained(
                "black-forest-labs/FLUX.1-Kontext-dev", torch_dtype=torch.bfloat16
            ).to(self.cpu_device)
            print("FLUX Worker: Modelo FluxKontext pronto (na CPU).")

    def to_gpu(self):
        """Move o pipeline para a GPU designada."""
        if self.device.type == 'cpu': return
        print(f"FLUX Worker: Movendo modelo para {self.device}...")
        self.pipe.to(self.device)
        print(f"FLUX Worker: Modelo na GPU {self.device}.")

    def to_cpu(self):
        """Move o pipeline de volta para a CPU e limpa a memória da GPU."""
        if self.device.type == 'cpu': return
        print(f"FLUX Worker: Descarregando modelo da GPU {self.device}...")
        self.pipe.to(self.cpu_device)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"FLUX Worker: GPU {self.device} limpa.")

    def _concatenate_images(self, images, direction="horizontal"):
        if not images: return None
        valid_images = [img.convert("RGB") for img in images if img is not None]
        if not valid_images: return None
        if len(valid_images) == 1: return valid_images[0]
        
        if direction == "horizontal":
            total_width = sum(img.width for img in valid_images)
            max_height = max(img.height for img in valid_images)
            concatenated = Image.new('RGB', (total_width, max_height))
            x_offset = 0
            for img in valid_images:
                y_offset = (max_height - img.height) // 2
                concatenated.paste(img, (x_offset, y_offset))
                x_offset += img.width
        else:
            max_width = max(img.width for img in valid_images)
            total_height = sum(img.height for img in valid_images)
            concatenated = Image.new('RGB', (max_width, total_height))
            y_offset = 0
            for img in valid_images:
                x_offset = (max_width - img.width) // 2
                concatenated.paste(img, (x_offset, y_offset))
                y_offset += img.height
        return concatenated

    @torch.inference_mode()
    def generate_image_internal(self, reference_images, prompt, width, height, seed=42):
        """A lógica real da geração de imagem, que espera estar na GPU."""
        concatenated_image = self._concatenate_images(reference_images, "horizontal")
        if concatenated_image is None:
            raise ValueError("Nenhuma imagem de referência válida foi fornecida.")
        
        image = self.pipe(
            image=concatenated_image, 
            prompt=prompt,
            guidance_scale=2.5,
            width=width,
            height=height,
            generator=torch.Generator(device="cpu").manual_seed(seed)
        ).images[0]
        
        return image

class FluxPoolManager:
    """
    Gerencia um pool de FluxWorkers, orquestrando um revezamento entre GPUs
    para permitir que a limpeza de uma GPU ocorra em paralelo com a computação em outra.
    """
    def __init__(self, device_ids=['cuda:0', 'cuda:1']):
        print(f"FLUX POOL MANAGER: Criando workers para os dispositivos: {device_ids}")
        self.workers = [FluxWorker(device_id) for device_id in device_ids]
        self.current_worker_index = 0
        self.lock = threading.Lock()
        self.last_cleanup_thread = None

    def _cleanup_worker(self, worker):
        """Função alvo para a thread de limpeza."""
        print(f"FLUX CLEANUP THREAD: Iniciando limpeza da GPU {worker.device} em background...")
        worker.to_cpu()
        print(f"FLUX CLEANUP THREAD: Limpeza da GPU {worker.device} concluída.")

    def generate_image(self, reference_images, prompt, width, height, seed=42):
        worker_to_use = None
        try:
            with self.lock:
                if self.last_cleanup_thread and self.last_cleanup_thread.is_alive():
                    print("FLUX POOL MANAGER: Aguardando limpeza da GPU anterior...")
                    self.last_cleanup_thread.join()
                    print("FLUX POOL MANAGER: Limpeza anterior concluída.")

                worker_to_use = self.workers[self.current_worker_index]
                previous_worker_index = (self.current_worker_index - 1 + len(self.workers)) % len(self.workers)
                worker_to_cleanup = self.workers[previous_worker_index]

                cleanup_thread = threading.Thread(target=self._cleanup_worker, args=(worker_to_cleanup,))
                cleanup_thread.start()
                self.last_cleanup_thread = cleanup_thread
                
                worker_to_use.to_gpu()
                
                self.current_worker_index = (self.current_worker_index + 1) % len(self.workers)
            
            print(f"FLUX POOL MANAGER: Gerando imagem em {worker_to_use.device}...")
            return worker_to_use.generate_image_internal(
                reference_images=reference_images,
                prompt=prompt,
                width=width,
                height=height,
                seed=seed
            )
        finally:
            # A limpeza do worker_to_use será feita na PRÓXIMA chamada a esta função,
            # permitindo que a computação do LTX ocorra em paralelo.
            pass

# --- Instância Singleton ---
print("Inicializando o Compositor de Cenas (FluxKontext Pool Manager)...")
hf_token = os.getenv('HF_TOKEN')
if hf_token: huggingface_hub.login(token=hf_token)
# Pool do Flux usa cuda:0 e cuda:1
flux_kontext_singleton = FluxPoolManager(device_ids=['cuda:0', 'cuda:1'])
print("Compositor de Cenas pronto.")
#-- END OF MODIFIED FILE app_fluxContext_Ltx/flux_kontext_helpers.py ---