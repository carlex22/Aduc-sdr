# ltx_manager_helpers.py
# Gerente de Pool de Workers LTX para revezamento assíncrono em múltiplas GPUs.
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
import threading
from PIL import Image

# Importa as funções e classes necessárias do inference.py
from inference import (
    create_ltx_video_pipeline,
    create_latent_upsampler,
    ConditioningItem,
    calculate_padding,
    prepare_conditioning
)
from ltx_video.pipelines.pipeline_ltx_video import LTXMultiScalePipeline

class LtxWorker:
    """
    Representa uma única instância do pipeline LTX, associada a uma GPU específica.
    O pipeline é carregado na CPU por padrão e movido para a GPU sob demanda.
    """
    def __init__(self, device_id='cuda:0'):
        self.device = torch.device(device_id if torch.cuda.is_available() else 'cpu')
        print(f"LTX Worker: Inicializando para o dispositivo {self.device} (carregando na CPU)...")
        
        config_file_path = "configs/ltxv-13b-0.9.8-distilled.yaml"
        with open(config_file_path, "r") as file:
            self.config = yaml.safe_load(file)

        LTX_REPO = "Lightricks/LTX-Video"
        models_dir = "downloaded_models_gradio"
        
        model_actual_path = huggingface_hub.hf_hub_download(
            repo_id=LTX_REPO,
            filename=self.config["checkpoint_path"],
            local_dir=models_dir,
            local_dir_use_symlinks=False
        )
        
        self.pipeline = create_ltx_video_pipeline(
            ckpt_path=model_actual_path,
            precision=self.config["precision"],
            text_encoder_model_name_or_path=self.config["text_encoder_model_name_or_path"],
            sampler=self.config["sampler"],
            device='cpu'
        )

        print(f"LTX Worker para {self.device}: Compilando o transformer (isso pode levar um momento)...")
        self.pipeline.transformer.to(memory_format=torch.channels_last)
        try:
            # Usando um modo de compilação menos agressivo e mais compatível.
            self.pipeline.transformer = torch.compile(self.pipeline.transformer, mode="reduce-overhead", fullgraph=True)
            print(f"LTX Worker para {self.device}: Transformer compilado com sucesso.")
        except Exception as e:
            print(f"AVISO: A compilação do Transformer falhou em {self.device}: {e}. Continuando sem compilação.")

        self.latent_upsampler = None
        if self.config.get("pipeline_type") == "multi-scale":
            print(f"LTX Worker para {self.device}: Carregando Latent Upsampler (Multi-Scale)...")
            upscaler_path = huggingface_hub.hf_hub_download(
                repo_id=LTX_REPO,
                filename=self.config["spatial_upscaler_model_path"],
                local_dir=models_dir,
                local_dir_use_symlinks=False
            )
            self.latent_upsampler = create_latent_upsampler(upscaler_path, 'cpu')

        print(f"LTX Worker para {self.device} pronto na CPU.")

    def to_gpu(self):
        """Move o pipeline e o upsampler para a GPU designada."""
        if self.device.type == 'cpu': return
        print(f"LTX Worker: Movendo pipeline para {self.device}...")
        self.pipeline.to(self.device)
        if self.latent_upsampler:
            print(f"LTX Worker: Movendo Latent Upsampler para {self.device}...")
            self.latent_upsampler.to(self.device)
        print(f"LTX Worker: Pipeline na GPU {self.device}.")

    def to_cpu(self):
        """Move o pipeline de volta para a CPU e limpa a memória da GPU."""
        if self.device.type == 'cpu': return
        print(f"LTX Worker: Descarregando pipeline da GPU {self.device}...")
        self.pipeline.to('cpu')
        if self.latent_upsampler:
            self.latent_upsampler.to('cpu')
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"LTX Worker: GPU {self.device} limpa.")
    
    def generate_video_fragment_internal(self, **kwargs):
        """A lógica real da geração de vídeo, que espera estar na GPU."""
        return self.pipeline(**kwargs)

class LtxPoolManager:
    """
    Gerencia um pool de LtxWorkers, orquestrando um revezamento entre GPUs
    para permitir que a limpeza de uma GPU ocorra em paralelo com a computação em outra.
    """
    def __init__(self, device_ids=['cuda:2', 'cuda:3']):
        print(f"LTX POOL MANAGER: Criando workers para os dispositivos: {device_ids}")
        self.workers = [LtxWorker(device_id) for device_id in device_ids]
        self.current_worker_index = 0
        self.lock = threading.Lock()
        self.last_cleanup_thread = None

    def _cleanup_worker(self, worker):
        """Função alvo para a thread de limpeza."""
        print(f"CLEANUP THREAD: Iniciando limpeza da GPU {worker.device} em background...")
        worker.to_cpu()
        print(f"CLEANUP THREAD: Limpeza da GPU {worker.device} concluída.")

    def generate_video_fragment(
        self,
        motion_prompt: str, conditioning_items_data: list,
        width: int, height: int, seed: int, cfg: float, video_total_frames: int,
        video_fps: int, num_inference_steps: int, use_attention_slicing: bool,
        decode_timestep: float, image_cond_noise_scale: float,
        current_fragment_index: int, output_path: str, progress
    ):
        worker_to_use = None
        try:
            with self.lock:
                if self.last_cleanup_thread and self.last_cleanup_thread.is_alive():
                    print("LTX POOL MANAGER: Aguardando limpeza da GPU anterior...")
                    self.last_cleanup_thread.join()

                worker_to_use = self.workers[self.current_worker_index]
                previous_worker_index = (self.current_worker_index - 1 + len(self.workers)) % len(self.workers)
                worker_to_cleanup = self.workers[previous_worker_index]

                cleanup_thread = threading.Thread(target=self._cleanup_worker, args=(worker_to_cleanup,))
                cleanup_thread.start()
                self.last_cleanup_thread = cleanup_thread
                
                worker_to_use.to_gpu()
                
                self.current_worker_index = (self.current_worker_index + 1) % len(self.workers)
            
            target_device = worker_to_use.device
            
            if use_attention_slicing:
                worker_to_use.pipeline.enable_attention_slicing()

            media_paths = [item[0] for item in conditioning_items_data]
            start_frames = [item[1] for item in conditioning_items_data]
            strengths = [item[2] for item in conditioning_items_data]

            padded_h, padded_w = ((height - 1) // 32 + 1) * 32, ((width - 1) // 32 + 1) * 32
            padding_vals = calculate_padding(height, width, padded_h, padded_w)

            conditioning_items = prepare_conditioning(
                conditioning_media_paths=media_paths, conditioning_strengths=strengths,
                conditioning_start_frames=start_frames, height=height, width=width,
                num_frames=video_total_frames, padding=padding_vals, pipeline=worker_to_use.pipeline,
            )
            
            for item in conditioning_items:
                item.media_item = item.media_item.to(target_device)

            kwargs = {
                "prompt": motion_prompt,
                "negative_prompt": "blurry, distorted, bad quality, artifacts",
                "height": padded_h, 
                "width": padded_w, 
                "num_frames": video_total_frames,
                "frame_rate": video_fps,
                "generator": torch.Generator(device=target_device).manual_seed(int(seed) + current_fragment_index),
                "output_type": "pt", 
                "guidance_scale": float(cfg),
                "conditioning_items": conditioning_items,
                "decode_timestep": decode_timestep, 
                "decode_noise_scale": worker_to_use.config.get("decode_noise_scale"),
                "image_cond_noise_scale": image_cond_noise_scale,
                "stochastic_sampling": worker_to_use.config.get("stochastic_sampling"),
                "is_video": True, 
                "vae_per_channel_normalize": True,
                "mixed_precision": (worker_to_use.config.get("precision") == "mixed_precision"),
                "enhance_prompt": False,
                "num_inference_steps": int(num_inference_steps),
            }
            
            progress(0.1, desc=f"[Câmera LTX em {worker_to_use.device}] Filmando Cena {current_fragment_index}...")
            result_tensor = worker_to_use.generate_video_fragment_internal(**kwargs).images
            
            pad_l, pad_r, pad_t, pad_b = map(int, padding_vals)
            slice_h = -pad_b if pad_b > 0 else None
            slice_w = -pad_r if pad_r > 0 else None
            cropped_tensor = result_tensor[:, :, :video_total_frames, pad_t:slice_h, pad_l:slice_w]
            video_np = (cropped_tensor[0].permute(1, 2, 3, 0).cpu().float().numpy() * 255).astype(np.uint8)
            
            with imageio.get_writer(output_path, fps=video_fps, codec='libx264', quality=8) as writer:
                for frame in video_np:
                    writer.append_data(frame)
            
            return output_path, video_total_frames

        finally:
            if use_attention_slicing and worker_to_use and worker_to_use.pipeline:
                worker_to_use.pipeline.disable_attention_slicing()

ltx_manager_singleton = LtxPoolManager(device_ids=['cuda:2', 'cuda:3'])