# flux_kontext_helpers.py (ADUC: O Especialista Pintor - com suporte a callback)
# AducSdr: Uma implementação aberta e funcional da arquitetura ADUC-SDR
# Copyright (C) 4 de Agosto de 2025  Carlos Rodrigues dos Santos
#
# Contato:
# Carlos Rodrigues dos Santos
# carlex22@gmail.com
# Rua Eduardo Carlos Pereira, 4125, B1 Ap32, Curitiba, PR, Brazil, CEP 8102025
#
# Repositórios e Projetos Relacionados:
# GitHub: https://github.com/carlex22/Aduc-sdr
#
#
# PENDING PATENT NOTICE: Please see NOTICE.md.
#
# Version 1.0.1

import torch
from PIL import Image, ImageOps
import gc
from diffusers import FluxKontextPipeline
import huggingface_hub
import os
import threading
import yaml
import logging

from tools.hardware_manager import hardware_manager

logger = logging.getLogger(__name__)

class FluxWorker:
    """Representa uma única instância do pipeline FluxKontext em um dispositivo."""
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

    def _create_composite_reference(self, images: list[Image.Image], target_width: int, target_height: int) -> Image.Image:
        if not images: return None
        valid_images = [img.convert("RGB") for img in images if img is not None]
        if not valid_images: return None
        if len(valid_images) == 1:
            if valid_images[0].size != (target_width, target_height):
                return ImageOps.fit(valid_images[0], (target_width, target_height), Image.Resampling.LANCZOS)
            return valid_images[0]

        base_height = valid_images[0].height
        resized_for_concat = []
        for img in valid_images:
            if img.height != base_height:
                aspect_ratio = img.width / img.height
                new_width = int(base_height * aspect_ratio)
                resized_for_concat.append(img.resize((new_width, base_height), Image.Resampling.LANCZOS))
            else:
                resized_for_concat.append(img)
        
        total_width = sum(img.width for img in resized_for_concat)
        concatenated = Image.new('RGB', (total_width, base_height))
        x_offset = 0
        for img in resized_for_concat:
            concatenated.paste(img, (x_offset, 0))
            x_offset += img.width
            
        #final_reference = ImageOps.fit(concatenated, (target_width, target_height), Image.Resampling.LANCZOS)
        return concatenated

    @torch.inference_mode()
    def generate_image_internal(self, reference_images: list[Image.Image], prompt: str, target_width: int, target_height: int, seed: int, callback: callable = None):
        composite_reference = self._create_composite_reference(reference_images, target_width, target_height)
        
        num_steps = 12 # Valor fixo otimizado

        logger.info(f"\n===== [CHAMADA AO PIPELINE FLUX em {self.device}] =====\n"
                    f"  - Prompt: '{prompt}'\n"
                    f"  - Resolução: {target_width}x{target_height}, Seed: {seed}, Passos: {num_steps}\n"
                    f"  - Nº de Imagens na Composição: {len(reference_images)}\n"
                    f"==========================================")
        
        generated_image = self.pipe(
            image=composite_reference, 
            prompt=prompt, 
            guidance_scale=2.5, 
            width=target_width,
            height=target_height,
            num_inference_steps=num_steps,
            generator=torch.Generator(device="cpu").manual_seed(seed),
            callback_on_step_end=callback,
            callback_on_step_end_tensor_inputs=["latents"] if callback else None
        ).images[0]
        
        return generated_image

class FluxPoolManager:
    def __init__(self, device_ids):
        logger.info(f"FLUX POOL MANAGER: Criando workers para os dispositivos: {device_ids}")
        self.workers = [FluxWorker(device_id) for device_id in device_ids]
        self.current_worker_index = 0
        self.lock = threading.Lock()
        self.last_cleanup_thread = None

    def _cleanup_worker_thread(self, worker):
        logger.info(f"FLUX CLEANUP THREAD: Iniciando limpeza de {worker.device} em background...")
        worker.to_cpu()

    def generate_image(self, reference_images, prompt, width, height, seed=42, callback=None):
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
            return worker_to_use.generate_image_internal(
                reference_images=reference_images,
                prompt=prompt,
                target_width=width,
                target_height=height,
                seed=seed,
                callback=callback
            )
        except Exception as e:
            logger.error(f"FLUX POOL MANAGER: Erro durante a geração: {e}", exc_info=True)
            raise e
        finally:
            pass

# --- Instanciação Singleton Dinâmica ---
logger.info("Lendo config.yaml para inicializar o FluxKontext Pool Manager...")
with open("config.yaml", 'r') as f: config = yaml.safe_load(f)
hf_token = os.getenv('HF_TOKEN'); 
if hf_token: huggingface_hub.login(token=hf_token)
flux_gpus_required = config['specialists']['flux']['gpus_required']
flux_device_ids = hardware_manager.allocate_gpus('Flux', flux_gpus_required)
flux_kontext_singleton = FluxPoolManager(device_ids=flux_device_ids)
logger.info("Especialista de Imagem (Flux) pronto.")