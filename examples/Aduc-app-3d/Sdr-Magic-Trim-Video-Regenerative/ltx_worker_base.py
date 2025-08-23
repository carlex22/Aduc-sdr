# ltx_worker_base.py (GPU-C: cuda:2)
# Worker para gerar os fragmentos de vídeo em resolução base.
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

from inference import (
    create_ltx_video_pipeline,
    ConditioningItem,
    calculate_padding,
    prepare_conditioning
)

class LtxGenerator:
    def __init__(self, device_id='cuda:2'):
        print(f"WORKER CÂMERA-BASE: Inicializando...")
        self.device = torch.device(device_id if torch.cuda.is_available() else 'cpu')
        print(f"WORKER CÂMERA-BASE: Usando dispositivo: {self.device}")
        
        config_file_path = "configs/ltxv-13b-0.9.8-distilled.yaml"
        with open(config_file_path, "r") as file:
            self.config = yaml.safe_load(file)

        LTX_REPO = "Lightricks/LTX-Video"
        models_dir = "downloaded_models_gradio"
        Path(models_dir).mkdir(parents=True, exist_ok=True)

        print("WORKER CÂMERA-BASE: Carregando pipeline LTX na CPU (estado de repouso)...")
        distilled_model_actual_path = huggingface_hub.hf_hub_download(
            repo_id=LTX_REPO,
            filename=self.config["checkpoint_path"],
            local_dir=models_dir,
            local_dir_use_symlinks=False
        )
        
        self.pipeline = create_ltx_video_pipeline(
            ckpt_path=distilled_model_actual_path,
            precision=self.config["precision"],
            text_encoder_model_name_or_path=self.config["text_encoder_model_name_or_path"],
            sampler=self.config["sampler"],
            device='cpu'
        )
        print("WORKER CÂMERA-BASE: Pronto (na CPU).")

    def to_gpu(self):
        if self.pipeline and torch.cuda.is_available():
            print(f"WORKER CÂMERA-BASE: Movendo LTX para {self.device}...")
            self.pipeline.to(self.device)

    def to_cpu(self):
        if self.pipeline:
            print(f"WORKER CÂMERA-BASE: Descarregando LTX da GPU {self.device}...")
            self.pipeline.to('cpu')
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def generate_video_fragment(
        self, motion_prompt: str, conditioning_items_data: list,
        width: int, height: int, seed: int, cfg: float, video_total_frames: int,
        video_fps: int, num_inference_steps: int, use_attention_slicing: bool,
        current_fragment_index: int, output_path: str, progress
    ):
        progress(0.1, desc=f"[Câmera LTX Base] Filmando Cena {current_fragment_index}...")
        
        target_device = self.pipeline.device
        
        if use_attention_slicing:
            self.pipeline.enable_attention_slicing()

        media_paths = [item[0] for item in conditioning_items_data]
        start_frames = [item[1] for item in conditioning_items_data]
        strengths = [item[2] for item in conditioning_items_data]

        padded_h, padded_w = ((height - 1) // 32 + 1) * 32, ((width - 1) // 32 + 1) * 32
        padding_vals = calculate_padding(height, width, padded_h, padded_w)

        conditioning_items = prepare_conditioning(
            conditioning_media_paths=media_paths, conditioning_strengths=strengths,
            conditioning_start_frames=start_frames, height=height, width=width,
            num_frames=video_total_frames, padding=padding_vals, pipeline=self.pipeline,
        )
        
        for item in conditioning_items:
            item.media_item = item.media_item.to(target_device)

        actual_num_frames = int(round((float(video_total_frames) - 1.0) / 8.0) * 8 + 1)
        first_pass_config = self.config.get("first_pass", {}).copy()
        first_pass_config['num_inference_steps'] = int(num_inference_steps)

        kwargs = {
            "prompt": motion_prompt, "negative_prompt": "blurry, distorted, bad quality, artifacts",
            "height": padded_h, "width": padded_w, "num_frames": actual_num_frames,
            "frame_rate": video_fps,
            "generator": torch.Generator(device=target_device).manual_seed(int(seed) + current_fragment_index),
            "output_type": "pt", "guidance_scale": float(cfg),
            "timesteps": first_pass_config.get("timesteps"),
            "conditioning_items": conditioning_items,
            "decode_timestep": self.config.get("decode_timestep"),
            "decode_noise_scale": self.config.get("decode_noise_scale"),
            "stochastic_sampling": self.config.get("stochastic_sampling"),
            "image_cond_noise_scale": 0.15, "is_video": True, "vae_per_channel_normalize": True,
            "mixed_precision": (self.config.get("precision") == "mixed_precision"),
            "enhance_prompt": False, "decode_every": 4, "num_inference_steps": int(num_inference_steps)
        }
        
        result_tensor = self.pipeline(**kwargs).images
        
        pad_l, pad_r, pad_t, pad_b = map(int, padding_vals)
        slice_h = -pad_b if pad_b > 0 else None; slice_w = -pad_r if pad_r > 0 else None
        
        cropped_tensor = result_tensor[:, :, :actual_num_frames, pad_t:slice_h, pad_l:slice_w]
        video_np = (cropped_tensor[0].permute(1, 2, 3, 0).cpu().float().numpy() * 255).astype(np.uint8)
        
        with imageio.get_writer(output_path, fps=video_fps, codec='libx264', quality=8) as writer:
            for frame in video_np:
                writer.append_data(frame)
        
        if use_attention_slicing and self.pipeline:
            self.pipeline.disable_attention_slicing()

        return output_path, actual_num_frames

# --- Instância Singleton para o Worker Base ---
ltx_base_singleton = LtxGenerator(device_id='cuda:2')