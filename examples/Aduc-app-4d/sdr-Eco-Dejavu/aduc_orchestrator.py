# aduc_orchestrator.py (Versão Finalizador de Cenas com Déjà-Vu)
# Copyright (C) 4 de Agosto de 2025  Carlos Rodrigues dos Santos

import os
import time
import shutil
import json
import subprocess
import logging
import gradio as gr
from PIL import Image

from deformes4D_engine import Deformes4DEngine
from ltx_manager_helpers import ltx_manager_singleton

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def quantize_to_multiple(n, m):
    """Arredonda n para o múltiplo mais próximo de m."""
    if m == 0: return n
    return int(round(n / m) * m)

class AducDirector:
    def __init__(self, workspace_dir):
        self.workspace_dir = workspace_dir
        os.makedirs(self.workspace_dir, exist_ok=True)
        self.state = {}
        logger.info(f"Diretor ADUC inicializado. Workspace em '{self.workspace_dir}' está pronto.")

    def reset(self):
        os.makedirs(self.workspace_dir, exist_ok=True)
        self.state = {}
        logger.info("Estado do Diretor resetado.")
        
    def update_state(self, key, value):
        log_value = value if not hasattr(value, 'shape') else f"Tensor de shape {value.shape}"
        logger.info(f"Diretor: Atualizando estado '{key}' -> '{log_value}'")
        self.state[key] = value
        
    def get_state(self, key, default=None):
        return self.state.get(key, default)

class AducOrchestrator:
    """Orquestra a pipeline do Finalizador de Cenas."""
    def __init__(self, workspace_dir: str):
        self.director = AducDirector(workspace_dir)
        self.editor = Deformes4DEngine(ltx_manager_singleton, workspace_dir)

    def task_finalize_scene(self, video_path: str, image_path: str, prompt: str,
                            duration_seconds: float, n_corte: int, n_eco: int, p_caminho: float,
                            cfg: float, steps: int, stg_scale: float, rescaling_scale: float,
                            decode_timestep: float, decode_noise_scale: float,
                            skip_block_list_str: str, progress):
        
        logger.info("Iniciando tarefa de finalização de cena.")
        
        quantized_n_corte = quantize_to_multiple(n_corte, 8)
        quantized_n_eco = quantize_to_multiple(n_eco, 8)
        
        a_frames_raw = int(duration_seconds * 24)
        a_chunks = max(1, round((a_frames_raw - 1) / 8))
        quantized_a_frames = a_chunks * 8 + 1
        
        logger.info(f"Parâmetros quantizados: n_corte={quantized_n_corte}, n_eco={quantized_n_eco}, a_frames={quantized_a_frames}")

        ltx_params = {
            "motion_prompt": prompt,
            "seed": int(time.time()),
            "video_fps": 24,
            "use_attention_slicing": True,
            "guidance_scale": cfg,
            "num_inference_steps": steps,
            "stg_scale": stg_scale,
            "rescaling_scale": rescaling_scale,
            "decode_timestep": decode_timestep,
            "decode_noise_scale": decode_noise_scale,
            "skip_block_list_str": skip_block_list_str,
        }

        final_video_path = self.editor.finalize_scene(
            video_path=video_path,
            image_path=image_path,
            n_corte=quantized_n_corte,
            n_eco=quantized_n_eco,
            a_frames=quantized_a_frames,
            p_caminho=p_caminho,
            ltx_params=ltx_params,
            progress=progress
        )
        return final_video_path