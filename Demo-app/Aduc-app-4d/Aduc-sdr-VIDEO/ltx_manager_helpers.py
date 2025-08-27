# ltx_manager_helpers.py
# Copyright (C) 4 de Agosto de 2025  Carlos Rodrigues dos Santos
#
# ORIGINAL SOURCE: LTX-Video by Lightricks Ltd. & other open-source projects.
# Licensed under the Apache License, Version 2.0
# https://github.com/Lightricks/LTX-Video
#
# MODIFICATIONS FOR ADUC-SDR_Video:
# This file is part of ADUC-SDR_Video, a derivative work based on LTX-Video.
# It has been modified to manage pools of LTX workers, handle GPU memory,
# and prepare parameters for the ADUC-SDR orchestration framework.
# All modifications are also licensed under the Apache License, Version 2.0.

import torch
import gc
import os
import yaml
import logging
import huggingface_hub
import time
import threading
import json

from optimization import optimize_ltx_worker, can_optimize_fp8
from hardware_manager import hardware_manager
from inference import create_ltx_video_pipeline, calculate_padding
from ltx_video.pipelines.pipeline_ltx_video import LatentConditioningItem
from ltx_video.models.autoencoders.vae_encode import vae_decode

logger = logging.getLogger(__name__)

class LtxWorker:
    def __init__(self, device_id, ltx_config_file):
        self.cpu_device = torch.device('cpu')
        self.device = torch.device(device_id if torch.cuda.is_available() else 'cpu')
        logger.info(f"LTX Worker ({self.device}): Inicializando com config '{ltx_config_file}'...")
        
        with open(ltx_config_file, "r") as file:
            self.config = yaml.safe_load(file)
        
        self.is_distilled = "distilled" in self.config.get("checkpoint_path", "")

        models_dir = "downloaded_models_gradio"
        
        logger.info(f"LTX Worker ({self.device}): Carregando modelo para a CPU...")
        model_path = os.path.join(models_dir, self.config["checkpoint_path"])
        if not os.path.exists(model_path):
             model_path = huggingface_hub.hf_hub_download(
                repo_id="Lightricks/LTX-Video", filename=self.config["checkpoint_path"],
                local_dir=models_dir, local_dir_use_symlinks=False
            )
        
        self.pipeline = create_ltx_video_pipeline(
            ckpt_path=model_path, precision=self.config["precision"],
            text_encoder_model_name_or_path=self.config["text_encoder_model_name_or_path"],
            sampler=self.config["sampler"], device='cpu'
        )
        logger.info(f"LTX Worker ({self.device}): Modelo pronto na CPU. É um modelo destilado? {self.is_distilled}")

        if self.device.type == 'cuda' and can_optimize_fp8():
            logger.info(f"LTX Worker ({self.device}): GPU com suporte a FP8 detectada. Iniciando otimização...")
            self.pipeline.to(self.device)
            optimize_ltx_worker(self)
            self.pipeline.to(self.cpu_device)
            logger.info(f"LTX Worker ({self.device}): Otimização concluída. Modelo pronto.")
        elif self.device.type == 'cuda':
            logger.info(f"LTX Worker ({self.device}): Otimização FP8 não suportada ou desativada. Usando modelo padrão.")

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
        target_device = worker_to_use.device
        height, width = kwargs['height'], kwargs['width']
        
        conditioning_data = kwargs.get('conditioning_items_data', [])
        final_conditioning_items = []

        # --- LOG ADICIONADO: Detalhes dos tensores de condicionamento ---
        conditioning_log_details = []
        for i, item in enumerate(conditioning_data):
            if hasattr(item, 'latent_tensor'):
                item.latent_tensor = item.latent_tensor.to(target_device)
                final_conditioning_items.append(item)
                conditioning_log_details.append(
                    f"  - Item {i}: frame={item.media_frame_number}, strength={item.conditioning_strength:.2f}, shape={list(item.latent_tensor.shape)}"
                )

        first_pass_config = worker_to_use.config.get("first_pass", {})
        padded_h, padded_w = ((height - 1) // 32 + 1) * 32, ((width - 1) // 32 + 1) * 32
        padding_vals = calculate_padding(height, width, padded_h, padded_w)

        pipeline_params = {
            "height": padded_h, "width": padded_w, 
            "num_frames": kwargs['video_total_frames'], "frame_rate": kwargs['video_fps'],
            "generator": torch.Generator(device=target_device).manual_seed(int(kwargs.get('seed', time.time())) + kwargs['current_fragment_index']),
            "conditioning_items": final_conditioning_items, 
            "is_video": True, "vae_per_channel_normalize": True,
            "decode_timestep": float(kwargs.get('decode_timestep', worker_to_use.config.get("decode_timestep", 0.05))),
            "decode_noise_scale": float(kwargs.get('decode_noise_scale', worker_to_use.config.get("decode_noise_scale", 0.025))),
            "image_cond_noise_scale": float(kwargs.get('image_cond_noise_scale', 0.0)),
            "stochastic_sampling": bool(kwargs.get('stochastic_sampling', worker_to_use.config.get("stochastic_sampling", False))),
            "prompt": kwargs['motion_prompt'],
            "negative_prompt": kwargs.get('negative_prompt', "blurry, distorted, static, bad quality, artifacts"),
            "guidance_scale": float(kwargs.get('guidance_scale', 1.0)),
            "stg_scale": float(kwargs.get('stg_scale', 0.0)),
            "rescaling_scale": float(kwargs.get('rescaling_scale', 1.0)),
        }
        
        if worker_to_use.is_distilled:
            pipeline_params["timesteps"] = first_pass_config.get("timesteps")
            pipeline_params["num_inference_steps"] = len(pipeline_params["timesteps"]) if "timesteps" in first_pass_config else 8
        else:
            pipeline_params["num_inference_steps"] = int(kwargs.get('num_inference_steps', 7))
        
        # --- LOG ADICIONADO: Exibição completa dos parâmetros da pipeline ---
        log_friendly_params = pipeline_params.copy()
        log_friendly_params.pop('generator', None)
        log_friendly_params.pop('conditioning_items', None)
        
        logger.info("="*60)
        logger.info(f"CHAMADA AO PIPELINE LTX NO DISPOSITIVO: {worker_to_use.device}")
        logger.info(f"Modelo: {'Distilled' if worker_to_use.is_distilled else 'Base'}")
        logger.info("-" * 20 + " PARÂMETROS DA PIPELINE " + "-" * 20)
        logger.info(json.dumps(log_friendly_params, indent=2))
        logger.info("-" * 20 + " ITENS DE CONDICIONAMENTO " + "-" * 19)
        logger.info("\n".join(conditioning_log_details))
        logger.info("="*60)
        # --- FIM DO LOG ADICIONADO ---
        
        return pipeline_params, padding_vals
    
    def generate_latent_fragment(self, **kwargs) -> (torch.Tensor, tuple):
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
                worker_to_use.to_cpu()


logger.info("Lendo config.yaml para inicializar o LTX Pool Manager...")
with open("config.yaml", 'r') as f:
    config = yaml.safe_load(f)
ltx_gpus_required = config['specialists']['ltx']['gpus_required']
ltx_device_ids = hardware_manager.allocate_gpus('LTX', ltx_gpus_required)
ltx_config_path = config['specialists']['ltx']['config_file']
ltx_manager_singleton = LtxPoolManager(device_ids=ltx_device_ids, ltx_config_file=ltx_config_path)
logger.info("Especialista de Vídeo (LTX) pronto.")