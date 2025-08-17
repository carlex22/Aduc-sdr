# flux_kontext_helpers.py
# Módulo de serviço para o FluxKontext, com gestão de memória atômica.
# Este arquivo é parte do projeto Euia-AducSdr e está sob a licença AGPL v3.
# Copyright (C) 4 de Agosto de 2025  Carlos Rodrigues dos Santos

import torch
from PIL import Image
import gc
from diffusers import FluxKontextPipeline
import huggingface_hub
import os

class Generator:
    def __init__(self, device_id='cuda:0'):
        self.cpu_device = torch.device('cpu')
        self.gpu_device = torch.device(device_id if torch.cuda.is_available() else 'cpu')
        print(f"WORKER COMPOSITOR: Usando dispositivo: {self.gpu_device}")
        self.pipe = None
        self._load_pipe_to_cpu()

    def _load_pipe_to_cpu(self):
        if self.pipe is None:
            print("WORKER COMPOSITOR: Carregando modelo FluxKontext para a CPU...")
            self.pipe = FluxKontextPipeline.from_pretrained(
                "black-forest-labs/FLUX.1-Kontext-dev", torch_dtype=torch.bfloat16
            ).to(self.cpu_device)
            print("WORKER COMPOSITOR: Modelo FluxKontext pronto (na CPU).")

    def to_gpu(self):
        if self.gpu_device.type == 'cpu': return
        print(f"WORKER COMPOSITOR: Movendo modelo para {self.gpu_device}...")
        self.pipe.to(self.gpu_device)
        print(f"WORKER COMPOSITOR: Modelo na GPU {self.gpu_device}.")

    def to_cpu(self):
        if self.gpu_device.type == 'cpu': return
        print(f"WORKER COMPOSITOR: Descarregando modelo da GPU {self.gpu_device}...")
        self.pipe.to(self.cpu_device)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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
    def generate_image(self, reference_images, prompt, width, height, seed=42):
        try:
            self.to_gpu()
            
            concatenated_image = self._concatenate_images(reference_images, "horizontal")
            if concatenated_image is None:
                raise ValueError("Nenhuma imagem de referência válida foi fornecida.")
            
            # ### CORREÇÃO ###
            # Ignora o tamanho da imagem concatenada e usa os parâmetros `width` e `height` fornecidos.
            image = self.pipe(
                image=concatenated_image, 
                prompt=prompt,
                guidance_scale=2.5,
                width=width,
                height=height,
                generator=torch.Generator(device="cpu").manual_seed(seed)
            ).images[0]
            
            return image
        finally:
            self.to_cpu()

# --- Instância Singleton ---
print("Inicializando o Compositor de Cenas (FluxKontext)...")
hf_token = os.getenv('HF_TOKEN')
if hf_token: huggingface_hub.login(token=hf_token)
flux_kontext_singleton = Generator(device_id='cuda:0')
print("Compositor de Cenas pronto.")