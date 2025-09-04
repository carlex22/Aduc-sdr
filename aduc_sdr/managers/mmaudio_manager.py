# managers/mmaudio_manager.py
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
# PENDING PATENT NOTICE: Please see NOTICE.md.
#
# Version: 2.3.0
#
# This file defines the MMAudioManager for the ADUC-SDR framework. It is responsible
# for generating audio synchronized with video clips. This version has been refactored
# to be self-contained by automatically cloning the MMAudio dependency from its
# official repository, making the framework more portable and easier to set up.

import torch
import logging
import subprocess
import os
import time
import yaml
import gc
from pathlib import Path
import gradio as gr
import sys

logger = logging.getLogger(__name__)

# --- Dependency Management ---
DEPS_DIR = Path("./deps")
MMAUDIO_REPO_DIR = DEPS_DIR / "MMAudio"
MMAUDIO_REPO_URL = "https://github.com/hkchengrex/MMAudio.git"

def setup_mmaudio_dependencies():
    """
    Ensures the MMAudio repository is cloned and available in the sys.path.
    This function is run once when the module is first imported.
    """
    if not MMAUDIO_REPO_DIR.exists():
        logger.info(f"MMAudio repository not found at '{MMAUDIO_REPO_DIR}'. Cloning from GitHub...")
        try:
            DEPS_DIR.mkdir(exist_ok=True)
            subprocess.run(
                ["git", "clone", "--depth", "1", MMAUDIO_REPO_URL, str(MMAUDIO_REPO_DIR)],
                check=True, capture_output=True, text=True
            )
            logger.info("MMAudio repository cloned successfully.")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to clone MMAudio repository. Git stderr: {e.stderr}")
            raise RuntimeError("Could not clone the required MMAudio dependency from GitHub.")
    else:
        logger.info("Found local MMAudio repository.")
    
    if str(MMAUDIO_REPO_DIR.resolve()) not in sys.path:
        sys.path.insert(0, str(MMAUDIO_REPO_DIR.resolve()))
        logger.info(f"Added '{MMAUDIO_REPO_DIR.resolve()}' to sys.path.")

setup_mmaudio_dependencies()

from mmaudio.eval_utils import ModelConfig, all_model_cfg, generate as mmaudio_generate, load_video, make_video
from mmaudio.model.flow_matching import FlowMatching
from mmaudio.model.networks import MMAudio, get_my_mmaudio
from mmaudio.model.utils.features_utils import FeaturesUtils
from mmaudio.model.sequence_config import SequenceConfig


class MMAudioManager:
    """
    Manages the MMAudio model for audio generation tasks.
    """
    def __init__(self, workspace_dir):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.cpu_device = torch.device("cpu")
        self.dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        self.workspace_dir = workspace_dir
        
        self.all_model_cfg = all_model_cfg
        self.model_config: 'ModelConfig' = self.all_model_cfg['large_44k_v2']
        self.net: 'MMAudio' = None
        self.feature_utils: 'FeaturesUtils' = None
        self.seq_cfg: 'SequenceConfig' = None
        
        self._load_models_to_cpu()

    def _adjust_paths_for_repo(self):
        """Adjusts the checkpoint paths in the model config to point inside the cloned repo."""
        for cfg_key in self.all_model_cfg:
            cfg = self.all_model_cfg[cfg_key]
            # The paths in the original config are relative, so we join them with our repo path
            cfg.model_path = MMAUDIO_REPO_DIR / cfg.model_path
            cfg.vae_path = MMAUDIO_REPO_DIR / cfg.vae_path
            if cfg.bigvgan_16k_path is not None:
                cfg.bigvgan_16k_path = MMAUDIO_REPO_DIR / cfg.bigvgan_16k_path
            cfg.synchformer_ckpt = MMAUDIO_REPO_DIR / cfg.synchformer_ckpt

    def _load_models_to_cpu(self):
        """Loads the MMAudio models to CPU memory on initialization."""
        try:
            self._adjust_paths_for_repo()
            logger.info("Verifying and downloading MMAudio models, if necessary...")
            self.model_config.download_if_needed()
            
            self.seq_cfg = self.model_config.seq_cfg
            
            logger.info(f"Loading MMAudio model: {self.model_config.model_name} to CPU...")
            self.net = get_my_mmaudio(self.model_config.model_name).eval()
            self.net.load_weights(torch.load(self.model_config.model_path, map_location=self.cpu_device, weights_only=True))
            
            logger.info("Loading MMAudio feature utils to CPU...")
            self.feature_utils = FeaturesUtils(
                tod_vae_ckpt=self.model_config.vae_path,
                synchformer_ckpt=self.model_config.synchformer_ckpt,
                enable_conditions=True,
                mode=self.model_config.mode,
                bigvgan_vocoder_ckpt=self.model_config.bigvgan_16k_path,
                need_vae_encoder=False
            )
            self.feature_utils = self.feature_utils.eval()
            self.net.to(self.cpu_device)
            self.feature_utils.to(self.cpu_device)
            logger.info("MMAudioManager ready on CPU.")
        except Exception as e:
            logger.error(f"Failed to load audio models: {e}", exc_info=True)
            self.net = None

    def to_gpu(self):
        """Moves the models and utilities to the GPU before inference."""
        if self.device == 'cpu': return
        logger.info(f"Moving MMAudioManager to GPU ({self.device})...")
        self.net.to(self.device, self.dtype)
        self.feature_utils.to(self.device, self.dtype)

    def to_cpu(self):
        """Moves the models back to CPU and clears VRAM after inference."""
        if self.device == 'cpu': return
        logger.info("Unloading MMAudioManager from GPU...")
        self.net.to(self.cpu_device)
        self.feature_utils.to(self.cpu_device)
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    def generate_audio_for_video(self, video_path: str, prompt: str, duration_seconds: float, output_path_override: str = None) -> str:
        """
        Generates audio for a video file, applying a negative prompt to avoid speech.
        """
        if self.net is None:
            raise gr.Error("MMAudio model is not loaded. Cannot generate audio.")

        logger.info("--- Generating Audio for Video Fragment ---")
        logger.info(f"--- Video: {os.path.basename(video_path)}")
        logger.info(f"--- Duration: {duration_seconds:.2f}s")
        
        negative_prompt = "human voice, speech, talking, singing, narration"
        logger.info(f"--- Prompt: '{prompt}' | Negative Prompt: '{negative_prompt}'")
        
        if duration_seconds < 1:
            logger.warning("Fragment too short (<1s). Returning original video.")
            return video_path
        
        if self.device == 'cpu':
            logger.warning("Generating audio on CPU. This may be very slow.")

        try:
            self.to_gpu()
            with torch.no_grad():
                rng = torch.Generator(device=self.device).manual_seed(int(time.time()))
                fm = FlowMatching(min_sigma=0, inference_mode='euler', num_steps=25)
                
                video_info = load_video(Path(video_path), duration_seconds)
                self.seq_cfg.duration = video_info.duration_sec
                self.net.update_seq_lengths(self.seq_cfg.latent_seq_len, self.seq_cfg.clip_seq_len, self.seq_cfg.sync_seq_len)
                
                audios = mmaudio_generate(
                    clip_video=video_info.clip_frames.unsqueeze(0),
                    sync_video=video_info.sync_frames.unsqueeze(0),
                    text=[prompt],
                    negative_text=[negative_prompt],
                    feature_utils=self.feature_utils,
                    net=self.net,
                    fm=fm,
                    rng=rng,
                    cfg_strength=4.5
                )
                audio_waveform = audios.float().cpu()[0]
                
                output_video_path = output_path_override if output_path_override else os.path.join(self.workspace_dir, f"{Path(video_path).stem}_with_audio.mp4")
                
                make_video(video_info, Path(output_video_path), audio_waveform, sampling_rate=self.seq_cfg.sampling_rate)
                logger.info(f"--- Fragment with audio saved to: {os.path.basename(output_video_path)}")
                return output_video_path
        finally:
            self.to_cpu()

# --- Singleton Instantiation ---
try:
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    WORKSPACE_DIR = config['application']['workspace_dir']
    mmaudio_manager_singleton = MMAudioManager(workspace_dir=WORKSPACE_DIR)
except Exception as e:
    logger.error(f"Could not initialize MMAudioManager: {e}", exc_info=True)
    mmaudio_manager_singleton = None