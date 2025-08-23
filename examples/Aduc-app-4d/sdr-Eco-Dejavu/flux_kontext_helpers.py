# flux_kontext_helpers.py (Versão Final Dinâmica)
# Copyright (C) 4 de Agosto de 2025  Carlos Rodrigues dos Santos

import torch
from PIL import Image
import gc
from diffusers import FluxKontextPipeline
import huggingface_hub
import os
import threading
import yaml
import logging

from hardware_manager import hardware_manager

# Configuração do Logging
logger = logging.getLogger(__name__)

class FluxWorker:
    """Representa uma única instância do pipeline FluxKontext em um dispositivo (GPU ou CPU)."""
    def __init__(self, device_id='cuda:0'):
        self.cpu_device = torch.device('cpu')
        self.device = torch.device(device_id if torch.cuda.is_available() else 'cpu')
        self.pipe = None
        self._load_pipe_to_cpu()

    def _load_pipe_to_cpu(self):
        if self.pipe is None:
            logger.info(f"FLUX Worker ({self.device}): Carregando modelo para a CPU...")
            self.pipe = FluxKontextPipeline.from_pretrained(
                "black-forest-labs/FLUX.1-Kontext-dev", torch_dtype=torch.bfloat16
            ).to(self.cpu_device)
            logger.info(f"FLUX Worker ({self.device}): Modelo pronto na CPU.")

    def to_gpu(self):
        if self.device.type == 'cpu': return
        logger.info(f"FLUX Worker: Movendo modelo para a GPU {self.device}...")
        self.pipe.to(self.device)

    def to_cpu(self):
        if self.device.type == 'cpu': return
        logger.info(f"FLUX Worker: Descarregando modelo da GPU {self.device}...")
        self.pipe.to(self.cpu_device)
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    def _concatenate_images(self, images, direction="horizontal"):
        if not images: return None
        valid_images = [img.convert("RGB") for img in images if img is not None]
        if not valid_images: return None
        if len(valid_images) == 1: return valid_images[0]
        if direction == "horizontal":
            total_width = sum(img.width for img in valid_images); max_height = max(img.height for img in valid_images)
            concatenated = Image.new('RGB', (total_width, max_height))
            x_offset = 0
            for img in valid_images:
                y_offset = (max_height - img.height) // 2; concatenated.paste(img, (x_offset, y_offset)); x_offset += img.width
        else:
            max_width = max(img.width for img in valid_images); total_height = sum(img.height for img in valid_images)
            concatenated = Image.new('RGB', (max_width, total_height))
            y_offset = 0
            for img in valid_images:
                x_offset = (max_width - img.width) // 2; concatenated.paste(img, (x_offset, y_offset)); y_offset += img.height
        return concatenated

    @torch.inference_mode()
    def generate_image_internal(self, reference_images, prompt, width, height, seed=42):
        concatenated_image = self._concatenate_images(reference_images, "horizontal")
        if concatenated_image is None: raise ValueError("Nenhuma imagem de referência válida.")
        
        # A mensagem de log sobre o ajuste de resolução é gerada internamente pela pipeline do diffusers.
        # Nosso log customizado mostrará a resolução que estamos *pedindo*.
        logger.info(f"\n===== [FLUX PIPELINE CALL on {self.device}] =====\n"
                    f"  - Prompt: '{prompt}'\n  - Resolução Solicitada: {width}x{height}, Seed: {seed}\n"
                    f"==========================================")
        
        return self.pipe(
            image=concatenated_image, 
            prompt=prompt, 
            guidance_scale=2.5, 
            width=width, 
            height=height, 
            generator=torch.Generator(device="cpu").manual_seed(seed)
        ).images[0]

class FluxPoolManager:
    """Gerencia um pool de FluxWorkers, orquestrando o revezamento entre dispositivos."""
    def __init__(self, device_ids):
        logger.info(f"FLUX POOL MANAGER: Criando workers para os dispositivos: {device_ids}")
        self.workers = [FluxWorker(device_id) for device_id in device_ids]
        self.current_worker_index = 0
        self.lock = threading.Lock()
        self.last_cleanup_thread = None

    def _cleanup_worker_thread(self, worker):
        logger.info(f"FLUX CLEANUP THREAD: Iniciando limpeza de {worker.device} em background...")
        worker.to_cpu()

    def generate_image(self, reference_images, prompt, width, height, seed=42):
        worker_to_use = None
        try:
            with self.lock:
                if self.last_cleanup_thread and self.last_cleanup_thread.is_alive():
                    self.last_cleanup_thread.join()
                worker_to_use = self.workers[self.current_worker_index]
                previous_worker_index = (self.current_worker_index - 1 + len(self.workers)) % len(self.workers)
                worker_to_cleanup = self.workers[previous_worker_index]
                cleanup_thread = threading.Thread(target=self._cleanup_worker_thread, args=(worker_to_cleanup,))
                cleanup_thread.start()
                self.last_cleanup_thread = cleanup_thread
                worker_to_use.to_gpu()
                self.current_worker_index = (self.current_worker_index + 1) % len(self.workers)
            
            logger.info(f"FLUX POOL MANAGER: Gerando imagem em {worker_to_use.device}...")
            return worker_to_use.generate_image_internal(reference_images, prompt, width, height, seed)
        except Exception as e:
            logger.error(f"FLUX POOL MANAGER: Erro durante a geração: {e}", exc_info=True)
            raise e
        finally:
            if worker_to_use:
                # A limpeza final acontece no início da próxima chamada para garantir
                # que o resultado seja retornado o mais rápido possível.
                pass

# --- Instanciação Singleton Dinâmica ---
logger.info("Lendo config.yaml para inicializar o FluxKontext Pool Manager...")

with open("config.yaml", 'r') as f:
    config = yaml.safe_load(f)

hf_token = os.getenv('HF_TOKEN')
if hf_token: huggingface_hub.login(token=hf_token)

flux_gpus_required = config['specialists']['flux']['gpus_required']
flux_device_ids = hardware_manager.allocate_gpus('Flux', flux_gpus_required)

flux_kontext_singleton = FluxPoolManager(device_ids=flux_device_ids)
logger.info("Especialista de Imagem (Flux) pronto.")