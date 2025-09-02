#Uma implementação aberta e funcional da arquitetura ADUC-SDR para geração de vídeo coerente.
# Copyright (C) 4 de Agosto de 2025  Carlos Rodrigues dos Santos
#
# Contato:
# Carlos Rodrigues dos Santos
# carlex22@gmail.com
#
# Repositórios e Projetos Relacionados:
# GitHub: https://github.com/carlex22/Aduc-sdr
# YouTube (Resultados): https://m.youtube.com/channel/UC3EgoJi_Fv7yuDpvfYNtoIQ
#
# Este programa é software livre: você pode redistribuí-lo e/ou modificá-lo
# sob os termos da Licença Pública Geral Affero da GNU como publicada pela
# Free Software Foundation, seja a versão 3 da Licença, ou
# (a seu critério) qualquer versão posterior.
#
# Este programa é distribuído na esperança de que seja útil,
# mas SEM QUALQUER GARANTIA; sem mesmo a garantia implícita de
# COMERCIALIZAÇÃO ou ADEQUAÇÃO A UM DETERMINADO FIM. Consulte a
# Licença Pública Geral Affero da GNU para mais detalhes.
#
# Você deve ter recebido uma cópia da Licença Pública Geral Affero da GNU
# junto com este programa. Se não, veja <https://www.gnu.org/licenses/>.
#
# AVISO DE PATENTE PENDENTE: O método e sistema ADUC implementado neste 
# software está em processo de patenteamento. Consulte NOTICE.md.

import torch
import logging
import subprocess
import os
import time
import yaml
import gc
from pathlib import Path
import gradio as gr

# Importa as classes e funções necessárias do MMAudio
try:
    from mmaudio.eval_utils import ModelConfig, all_model_cfg, generate as mmaudio_generate, load_video, make_video
    from mmaudio.model.flow_matching import FlowMatching
    from mmaudio.model.networks import MMAudio, get_my_mmaudio
    from mmaudio.model.utils.features_utils import FeaturesUtils
    from mmaudio.model.sequence_config import SequenceConfig
except ImportError:
    raise ImportError("MMAudio não foi encontrado. Por favor, instale-o a partir do GitHub: git+https://github.com/hkchengrex/MMAudio.git")

logger = logging.getLogger(__name__)

class AudioSpecialist:
    """
    Especialista responsável por gerar áudio para fragmentos de vídeo.
    Gerencia o carregamento e descarregamento de modelos de áudio da VRAM.
    """
    def __init__(self, workspace_dir):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.cpu_device = torch.device("cpu")
        self.dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        self.workspace_dir = workspace_dir
        
        self.model_config: ModelConfig = all_model_cfg['large_44k_v2']
        self.net: MMAudio = None
        self.feature_utils: FeaturesUtils = None
        self.seq_cfg: SequenceConfig = None
        
        self._load_models_to_cpu()

    def _load_models_to_cpu(self):
        """Carrega os modelos MMAudio para a memória da CPU na inicialização."""
        try:
            logger.info("Verificando e baixando modelos MMAudio, se necessário...")
            self.model_config.download_if_needed()
            
            self.seq_cfg = self.model_config.seq_cfg
            
            logger.info(f"Carregando modelo MMAudio: {self.model_config.model_name} para a CPU...")
            self.net = get_my_mmaudio(self.model_config.model_name).eval()
            self.net.load_weights(torch.load(self.model_config.model_path, map_location=self.cpu_device, weights_only=True))
            
            logger.info("Carregando utilitários de features do MMAudio para a CPU...")
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
            logger.info("Especialista de áudio pronto na CPU.")
        except Exception as e:
            logger.error(f"Falha ao carregar modelos de áudio: {e}", exc_info=True)
            self.net = None

    def to_gpu(self):
        """Move os modelos e utilitários para a GPU antes da inferência."""
        if self.device == 'cpu': return
        logger.info(f"Movendo especialista de áudio para a GPU ({self.device})...")
        self.net.to(self.device, self.dtype)
        self.feature_utils.to(self.device, self.dtype)

    def to_cpu(self):
        """Move os modelos de volta para a CPU e limpa a VRAM após a inferência."""
        if self.device == 'cpu': return
        logger.info("Descarregando especialista de áudio da GPU...")
        self.net.to(self.cpu_device)
        self.feature_utils.to(self.cpu_device)
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    def generate_audio_for_video(self, video_path: str, prompt: str, duration_seconds: float) -> str:
        """
        Gera áudio para um arquivo de vídeo, aplicando um prompt negativo para evitar fala.

        Args:
            video_path (str): Caminho para o vídeo silencioso.
            prompt (str): Descrição da cena para guiar a geração de SFX.
            duration_seconds (float): Duração do áudio a ser gerado.

        Returns:
            str: Caminho para o novo arquivo de vídeo com áudio.
        """
        if self.net is None:
            raise gr.Error("Modelo MMAudio não está carregado. Não é possível gerar áudio.")

        logger.info("------------------------------------------------------")
        logger.info("--- Gerando Áudio para Fragmento de Vídeo ---")
        logger.info(f"--- Vídeo Fragmento: {os.path.basename(video_path)}")
        logger.info(f"--- Duração: {duration_seconds:.2f}s")
        logger.info(f"--- Prompt (Descrição da Cena): '{prompt}'")
        
        negative_prompt = "human voice"
        logger.info(f"--- Negative Prompt: '{negative_prompt}'")
        
        if duration_seconds < 1:
            logger.warning("Fragmento muito curto (<1s). Retornando vídeo silencioso.")
            logger.info("------------------------------------------------------")
            return video_path

        if self.device == 'cpu':
            logger.warning("Gerando áudio na CPU. Isso pode ser muito lento.")

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
                
                fragment_name = Path(video_path).stem
                output_video_path = os.path.join(self.workspace_dir, f"{fragment_name}_com_audio.mp4")
                
                make_video(video_info, Path(output_video_path), audio_waveform, sampling_rate=self.seq_cfg.sampling_rate)
                logger.info(f"--- Fragmento com áudio salvo em: {os.path.basename(output_video_path)}")
                logger.info("------------------------------------------------------")
                return output_video_path
        finally:
            self.to_cpu()

# Singleton instantiation
try:
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    WORKSPACE_DIR = config['application']['workspace_dir']
    audio_specialist_singleton = AudioSpecialist(workspace_dir=WORKSPACE_DIR)
except Exception as e:
    logger.error(f"Não foi possível inicializar o AudioSpecialist: {e}", exc_info=True)
    audio_specialist_singleton = None