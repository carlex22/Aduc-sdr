# ltx_manager_helpers.py (Versão que retorna o latente original)
# Copyright (C) 4 de Agosto de 2025  Carlos Rodrigues dos Santos

import torch
import gc
import os
import yaml
import numpy as np
import imageio
import threading
import logging
import huggingface_hub
import time

from hardware_manager import hardware_manager
from inference import create_ltx_video_pipeline, calculate_padding, prepare_conditioning
from ltx_video.pipelines.pipeline_ltx_video import ConditioningItem
from ltx_video.models.autoencoders.vae_encode import vae_decode

logger = logging.getLogger(__name__)

class LtxWorker:
    """Representa uma única instância do pipeline LTX em um dispositivo."""
    def __init__(self, device_id, ltx_config_file):
        self.cpu_device = torch.device('cpu')
        self.device = torch.device(device_id if torch.cuda.is_available() else 'cpu')
        logger.info(f"LTX Worker ({self.device}): Inicializando com config '{ltx_config_file}'...")
        
        with open(ltx_config_file, "r") as file:
            self.config = yaml.safe_load(file)
        
        self.is_distilled = "distilled" in self.config.get("checkpoint_path", "")

        LTX_REPO = "Lightricks/LTX-Video"
        models_dir = "downloaded_models_gradio"
        
        logger.info(f"LTX Worker ({self.device}): Carregando modelo para a CPU...")
        model_path = huggingface_hub.hf_hub_download(
            repo_id=LTX_REPO, filename=self.config["checkpoint_path"],
            local_dir=models_dir, local_dir_use_symlinks=False
        )
        
        self.pipeline = create_ltx_video_pipeline(
            ckpt_path=model_path, precision=self.config["precision"],
            text_encoder_model_name_or_path=self.config["text_encoder_model_name_or_path"],
            sampler=self.config["sampler"], device='cpu'
        )
        logger.info(f"LTX Worker ({self.device}): Modelo pronto na CPU. É um modelo destilado? {self.is_distilled}")

    def to_gpu(self):
        if self.device.type == 'cpu': return
        logger.info(f"LTX Worker: Movendo pipeline para a GPU {self.device}...")
        self.pipeline.to(self.device)

    def to_cpu(self):
        if self.device.type == 'cpu': return
        logger.info(f"LTX Worker: Descarregando pipeline da GPU {self.device}...")
        self.pipeline.to('cpu')
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    def generate_video_fragment_internal(self, **kwargs):
        return self.pipeline(**kwargs).images

class LtxPoolManager:
    """Gerencia um pool de LtxWorkers, orquestrando o revezamento entre dispositivos."""
    def __init__(self, device_ids, ltx_config_file):
        logger.info(f"LTX POOL MANAGER: Criando workers para os dispositivos: {device_ids}")
        self.workers = [LtxWorker(dev_id, ltx_config_file) for dev_id in device_ids]
        self.current_worker_index = 0
        self.lock = threading.Lock()
        self.last_cleanup_thread = None

    def _cleanup_worker_thread(self, worker):
        logger.info(f"LTX CLEANUP THREAD: Iniciando limpeza de {worker.device} em background...")
        worker.to_cpu()

    def _prepare_and_log_params(self, worker_to_use, **kwargs):
        """Prepara todos os parâmetros, incluindo a lista final de condicionamento."""
        target_device = worker_to_use.device
        height, width = kwargs['height'], kwargs['width']
        
        conditioning_data = kwargs.get('conditioning_items_data', [])
        final_conditioning_items = []

        latent_items = [item for item in conditioning_data if hasattr(item, 'latent_tensor')]
        pixel_items_data = [item for item in conditioning_data if not hasattr(item, 'latent_tensor')]

        if latent_items:
            for item in latent_items:
                item.latent_tensor = item.latent_tensor.to(target_device)
            final_conditioning_items.extend(latent_items)

        if pixel_items_data:
            padded_h_pix, padded_w_pix = ((height - 1) // 32 + 1) * 32, ((width - 1) // 32 + 1) * 32
            padding_vals_pix = calculate_padding(height, width, padded_h_pix, padded_w_pix)
            prepared_pixel_items = prepare_conditioning(
                conditioning_media_paths=[item[0] for item in pixel_items_data],
                conditioning_strengths=[item[2] for item in pixel_items_data],
                conditioning_start_frames=[item[1] for item in pixel_items_data],
                height=height, width=width, num_frames=kwargs['video_total_frames'],
                padding=padding_vals_pix, pipeline=worker_to_use.pipeline,
            )
            if prepared_pixel_items:
                for item in prepared_pixel_items:
                    item.media_item = item.media_item.to(target_device)
                final_conditioning_items.extend(prepared_pixel_items)
        
        first_pass_config = worker_to_use.config.get("first_pass", {})
        skip_block_list = []
        skip_str = kwargs.get('skip_block_list_str', '').strip()
        if skip_str:
            try:
                skip_block_list = [int(x.strip()) for x in skip_str.split(',') if x.strip().isdigit()]
            except (ValueError, TypeError):
                logger.warning(f"Não foi possível parsear a lista de blocos: '{skip_str}'.")
                skip_block_list = first_pass_config.get('skip_block_list', [])

        def get_scalar_from_config(key, default):
            val = first_pass_config.get(key, default)
            return val[0] if isinstance(val, list) else val

        padded_h, padded_w = ((height - 1) // 32 + 1) * 32, ((width - 1) // 32 + 1) * 32
        padding_vals = calculate_padding(height, width, padded_h, padded_w)

        pipeline_params = {
            "prompt": kwargs['motion_prompt'], "negative_prompt": "blurry, distorted",
            "height": padded_h, "width": padded_w, "num_frames": kwargs['video_total_frames'], "frame_rate": kwargs['video_fps'],
            "generator": torch.Generator(device=target_device).manual_seed(int(kwargs.get('seed', time.time())) + kwargs['current_fragment_index']),
            "conditioning_items": final_conditioning_items, "is_video": True, "vae_per_channel_normalize": True,
            "guidance_scale": float(kwargs.get('guidance_scale', get_scalar_from_config('guidance_scale', 1.0))),
            "stg_scale": float(kwargs.get('stg_scale', get_scalar_from_config('stg_scale', 0.0))),
            "rescaling_scale": float(kwargs.get('rescaling_scale', get_scalar_from_config('rescaling_scale', 1.0))),
            "decode_timestep": float(kwargs.get('decode_timestep', 0.05)),
            "decode_noise_scale": float(kwargs.get('decode_noise_scale', 0.025)),
            "skip_block_list": skip_block_list if skip_block_list else first_pass_config.get('skip_block_list', [])
        }
        if worker_to_use.is_distilled:
            pipeline_params["timesteps"] = first_pass_config.get("timesteps")
            pipeline_params["num_inference_steps"] = len(pipeline_params["timesteps"]) if "timesteps" in first_pass_config else 8
        else:
            pipeline_params["num_inference_steps"] = int(kwargs.get('num_inference_steps', 20))
        
        log_message = f"\n{'='*25} [PARÂMETROS DA CHAMADA LTX em {target_device}] {'='*25}\n"
        for key, value in pipeline_params.items():
            val_str = str(value)
            if len(val_str) > 150: val_str = val_str[:150] + "..."
            log_message += f"  - {key:<25}: {val_str}\n"
        log_message += "=" * (71 + len(str(target_device))) + "\n"
        logger.info(log_message)
        
        return pipeline_params, padding_vals

    def generate_video_fragment(self, **kwargs) -> (str, torch.Tensor):
        """Gera um fragmento, salva como vídeo e retorna o caminho E o tensor latente original."""
        progress = kwargs.get('progress')
        latent_tensor, padding_vals = self.generate_latent_fragment(**kwargs)
        
        # Clona o tensor latente original para retorno antes de qualquer modificação.
        original_latent_for_return = latent_tensor.clone()
        
        if progress: progress(0.9, desc="Decodificando latentes para vídeo...")

        vae = self.workers[0].pipeline.vae
        vae.to(latent_tensor.device)
        
        decode_timestep = float(kwargs.get('decode_timestep', 0.05))
        timestep_tensor = torch.tensor([decode_timestep] * latent_tensor.shape[0], device=latent_tensor.device, dtype=vae.dtype)
        
        with torch.no_grad():
            pixel_tensor = vae_decode(
                latent_tensor, vae, is_video=True, 
                vae_per_channel_normalize=True, timestep=timestep_tensor
            )
        
        pad_l, pad_r, pad_t, pad_b = padding_vals
        slice_h, slice_w = (-pad_b if pad_b > 0 else None), (-pad_r if pad_r > 0 else None)
        pixel_tensor = pixel_tensor[:, :, :, pad_t:slice_h, pad_l:slice_w]

        pixel_tensor = pixel_tensor.squeeze(0).permute(1, 2, 3, 0)
        pixel_tensor = (pixel_tensor.clamp(-1, 1) + 1) / 2.0
        
        video_np = (pixel_tensor.detach().cpu().float().numpy() * 255).astype(np.uint8)
        
        with imageio.get_writer(kwargs['output_path'], fps=kwargs['video_fps'], codec='libx264', quality=8) as writer:
            for frame in video_np: writer.append_data(frame)
        
        return kwargs['output_path'], original_latent_for_return

    def generate_latent_fragment(self, **kwargs) -> (torch.Tensor, tuple):
        """Gera um fragmento e retorna o tensor latente diretamente."""
        worker_to_use = None
        progress = kwargs.get('progress')
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
            
            if kwargs.get('use_attention_slicing'): worker_to_use.pipeline.enable_attention_slicing()

            pipeline_params, padding_vals = self._prepare_and_log_params(worker_to_use, **kwargs)
            pipeline_params['output_type'] = "latent"

            if progress: progress(0.1, desc=f"[Especialista LTX em {worker_to_use.device}] Gerando latentes...")
            
            with torch.no_grad():
                result_tensor = worker_to_use.generate_video_fragment_internal(**pipeline_params)
            
            return result_tensor, padding_vals
        except Exception as e:
            logger.error(f"LTX POOL MANAGER: Erro durante a geração de latentes: {e}", exc_info=True)
            raise e
        finally:
            if worker_to_use:
                logger.info(f"LTX POOL MANAGER: Executando limpeza final para {worker_to_use.device}...")
                if kwargs.get('use_attention_slicing') and worker_to_use.pipeline:
                    worker_to_use.pipeline.disable_attention_slicing()
                worker_to_use.to_cpu()

# --- Instanciação Singleton Dinâmica ---
logger.info("Lendo config.yaml para inicializar o LTX Pool Manager...")
with open("config.yaml", 'r') as f:
    config = yaml.safe_load(f)

ltx_gpus_required = config['specialists']['ltx']['gpus_required']
ltx_device_ids = hardware_manager.allocate_gpus('LTX', ltx_gpus_required)
ltx_config_path = config['specialists']['ltx']['config_file']
ltx_manager_singleton = LtxPoolManager(device_ids=ltx_device_ids, ltx_config_file=ltx_config_path)
logger.info("Especialista de Vídeo (LTX) pronto.")