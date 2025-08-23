# ltx_worker_upscaler.py (Corrigido com dtype=bfloat16)
# Worker para fazer upscale dos fragmentos de vídeo para alta resolução.
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
from einops import rearrange

from inference import create_ltx_video_pipeline
from ltx_video.models.autoencoders.latent_upsampler import LatentUpsampler
from ltx_video.models.autoencoders.vae_encode import vae_encode, vae_decode

class LtxUpscaler:
    def __init__(self, device_id='cuda:2'):
        print(f"WORKER CÂMERA-UPSCALER: Inicializando para {device_id}...")
        self.device = torch.device(device_id if torch.cuda.is_available() else 'cpu')
        self.model_dtype = torch.bfloat16 # <<<--- DEFINIR O DTYPE DO MODELO
        
        config_file_path = "configs/ltxv-13b-0.9.8-distilled.yaml"
        with open(config_file_path, "r") as file:
            self.config = yaml.safe_load(file)

        LTX_REPO = "Lightricks/LTX-Video"
        models_dir = "downloaded_models_gradio"
        Path(models_dir).mkdir(parents=True, exist_ok=True)

        print(f"WORKER CÂMERA-UPSCALER ({self.device}): Carregando VAE na CPU...")
        distilled_model_actual_path = huggingface_hub.hf_hub_download(
            repo_id=LTX_REPO, filename=self.config["checkpoint_path"],
            local_dir=models_dir, local_dir_use_symlinks=False
        )
        temp_pipeline = create_ltx_video_pipeline(
            ckpt_path=distilled_model_actual_path, precision=self.config["precision"],
            text_encoder_model_name_or_path=self.config["text_encoder_model_name_or_path"],
            sampler=self.config["sampler"], device='cpu'
        )
        self.vae = temp_pipeline.vae.to(self.model_dtype) # <<<--- CARREGA NO DTYPE CORRETO
        del temp_pipeline
        gc.collect()

        print(f"WORKER CÂMERA-UPSCALER ({self.device}): Carregando Latent Upsampler na CPU...")
        upscaler_path = huggingface_hub.hf_hub_download(
            repo_id=LTX_REPO, filename=self.config["spatial_upscaler_model_path"],
            local_dir=models_dir, local_dir_use_symlinks=False
        )
        self.latent_upsampler = LatentUpsampler.from_pretrained(upscaler_path).to(self.model_dtype) # <<<--- CARREGA NO DTYPE CORRETO
        self.latent_upsampler.to('cpu')
        
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
    def upscale_video_fragment(self, video_path_low_res: str, output_path: str, video_fps: int):
        print(f"UPSCALER ({self.device}): Processando {os.path.basename(video_path_low_res)}")
        
        with imageio.get_reader(video_path_low_res) as reader:
            video_frames = [frame for frame in reader]
        video_np = np.stack(video_frames)
        
        # <<<--- CORREÇÃO CRÍTICA AQUI ---_>>>
        video_tensor = torch.from_numpy(video_np).permute(0, 3, 1, 2).float() / 255.0
        video_tensor = (video_tensor * 2.0) - 1.0
        video_tensor = video_tensor.unsqueeze(0).permute(0, 2, 1, 3, 4)
        video_tensor = video_tensor.to(self.device, dtype=self.model_dtype) # Envia para GPU JÁ NO DTYPE CORRETO

        latents = vae_encode(video_tensor, self.vae)
        upsampled_latents = self.latent_upsampler(latents)
        upsampled_video_tensor = vae_decode(upsampled_latents, self.vae, is_video=True)
        
        upsampled_video_tensor = (upsampled_video_tensor.clamp(-1, 1) + 1) / 2.0
        video_np_high_res = (upsampled_video_tensor[0].permute(1, 2, 3, 0).cpu().float().numpy() * 255).astype(np.uint8) # Converte de volta para float para salvar
        
        with imageio.get_writer(output_path, fps=video_fps, codec='libx264', quality=8) as writer:
            for frame in video_np_high_res:
                writer.append_data(frame)

        print(f"UPSCALER ({self.device}): Arquivo salvo em {os.path.basename(output_path)}")
        return output_path