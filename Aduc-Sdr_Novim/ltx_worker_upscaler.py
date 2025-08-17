#--- START OF MODIFIED FILE app_fluxContext_Ltx/ltx_worker_upscaler.py ---
# ltx_worker_upscaler.py
# Worker para fazer upscale de latentes de vídeo para alta resolução.
# Este arquivo é parte do projeto Euia-AducSdr e está sob a licença AGPL v3.
# Copyright (C) 4 de Agosto de 2025  Carlos Rodrigues dos Santos

import torch
import gc
import os
import yaml
import numpy as np 
import imageio
from pathlib import Path
import huggingface_hub
from PIL import Image # <--- IMPORTAÇÃO ADICIONADA AQUI

from inference import create_ltx_video_pipeline
from ltx_video.models.autoencoders.latent_upsampler import LatentUpsampler
from ltx_video.models.autoencoders.vae_encode import vae_encode, vae_decode

class LtxUpscaler:
    def __init__(self, device_id='cuda:0'):
        print(f"WORKER CÂMERA-UPSCALER: Inicializando para {device_id}...")
        self.device = torch.device(device_id if torch.cuda.is_available() else 'cpu')
        self.model_dtype = torch.bfloat16
        
        config_file_path = "configs/ltxv-13b-0.9.8-dev.yaml"
        with open(config_file_path, "r") as file:
            self.config = yaml.safe_load(file)

        LTX_REPO = "Lightricks/LTX-Video"
        models_dir = "downloaded_models_gradio"
        Path(models_dir).mkdir(parents=True, exist_ok=True)

        print(f"WORKER CÂMERA-UPSCALER ({self.device}): Carregando VAE na CPU...")
        model_actual_path = huggingface_hub.hf_hub_download(
            repo_id=LTX_REPO, filename=self.config["checkpoint_path"],
            local_dir=models_dir, local_dir_use_symlinks=False
        )
        temp_pipeline = create_ltx_video_pipeline(
            ckpt_path=model_actual_path, precision=self.config["precision"],
            text_encoder_model_name_or_path=self.config["text_encoder_model_name_or_path"],
            sampler=self.config["sampler"], device='cpu'
        )
        self.vae = temp_pipeline.vae.to(self.model_dtype)
        del temp_pipeline
        gc.collect()

        print(f"WORKER CÂMERA-UPSCALER ({self.device}): Carregando Latent Upsampler na CPU...")
        upscaler_path = huggingface_hub.hf_hub_download(
            repo_id=LTX_REPO, filename=self.config["spatial_upscaler_model_path"],
            local_dir=models_dir, local_dir_use_symlinks=False
        )
        self.latent_upsampler = LatentUpsampler.from_pretrained(upscaler_path).to(self.model_dtype)
        self.latent_upsampler.to('cpu')
        self.vae.to('cpu')
        
        print(f"WORKER CÂMERA-UPSCALER ({self.device}): Pronto (na CPU).")

    def to_gpu(self):
        if self.latent_upsampler and self.vae and torch.cuda.is_available():
            print(f"WORKER CÂMERA-UPSCALER: Movendo modelos para {self.device}...")
            self.latent_upsampler.to(self.device)
            self.vae.to(self.device)

    def to_cpu(self):
        if self.latent_upsampler and self.vae:
            print(f"WORKER CÂMERA-UPSCALER: Descarregando modelos da GPU {self.device}...")
            self.latent_upsampler.to('cpu')
            self.vae.to('cpu')
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    @torch.no_grad()
    def upscale_latents_to_video(self, latent_path: str, output_path: str, video_fps: int):
        print(f"UPSCALER ({self.device}): Processando latentes de {os.path.basename(latent_path)}")
        
        latents = torch.load(latent_path).to(self.device, dtype=self.model_dtype)

        upsampled_latents = self.latent_upsampler(latents)
        
        decode_timestep = torch.tensor([0.0] * upsampled_latents.shape[0], device=self.device)
        upsampled_video_tensor = vae_decode(
            upsampled_latents, self.vae, is_video=True, timestep=decode_timestep
        )
        
        upsampled_video_tensor = (upsampled_video_tensor.clamp(-1, 1) + 1) / 2.0
        video_np_high_res = (upsampled_video_tensor[0].permute(1, 2, 3, 0).cpu().float().numpy() * 255).astype(np.uint8)
        
        with imageio.get_writer(output_path, fps=video_fps, codec='libx264', quality=8) as writer:
            for frame in video_np_high_res:
                writer.append_data(frame)

        print(f"UPSCALER ({self.device}): Arquivo de vídeo salvo em {os.path.basename(output_path)}")
        return output_path
        
    @torch.no_grad()
    def decode_single_latent_frame(self, latent_frame_tensor: torch.Tensor) -> Image.Image:
        """Decodifica um único frame latente para uma imagem PIL para o Gemini."""
        latent_frame_tensor = latent_frame_tensor.to(self.device, dtype=self.model_dtype)
        
        decode_timestep = torch.tensor([0.0] * latent_frame_tensor.shape[0], device=self.device)
        decoded_tensor = vae_decode(
            latent_frame_tensor, self.vae, is_video=True, timestep=decode_timestep
        )
        
        decoded_tensor = (decoded_tensor.clamp(-1, 1) + 1) / 2.0
        numpy_image = (decoded_tensor[0].permute(2, 3, 1, 0).squeeze().cpu().float().numpy() * 255).astype(np.uint8)
        return Image.fromarray(numpy_image)
#--- END OF MODIFIED FILE app_fluxContext_Ltx/ltx_worker_upscaler.py ---