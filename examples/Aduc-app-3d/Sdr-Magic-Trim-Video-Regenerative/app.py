# app_deformes_v1.5_final.py
# Framework Deformes: Cirurgia e Costura de Realidade Latente.
# Copyright (C) 2025 Carlos Rodrigues dos Santos

# ==============================================================================
# SEÇÃO 0: IMPORTS
# Todas as dependências são importadas no início do arquivo para evitar NameError.
# ==============================================================================
import gradio as gr
import torch
import os
import shutil
import time
import imageio
import numpy as np
import json
import subprocess
from PIL import Image

# Presume-se que a estrutura do projeto LTX (helpers, etc.) está no lugar
from ltx_manager_helpers import ltx_manager_singleton
from ltx_video.models.autoencoders.vae_encode import vae_encode, vae_decode

# ==============================================================================
# SEÇÃO 0.5: CONFIGURAÇÃO GLOBAL
# ==============================================================================
WORKSPACE_DIR = "deformes_workspace"
os.makedirs(WORKSPACE_DIR, exist_ok=True)

# ==============================================================================
# SEÇÃO 1: O FRAMEWORK DEFORMES
# ==============================================================================

class Deformes:
    """
    Framework Deformes: Um conjunto de ferramentas para manipulação e edição de vídeo
    diretamente no espaço latente, minimizando a perda de informação.
    """
    def __init__(self, ltx_manager):
        self.ltx_manager = ltx_manager
        self._vae = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    @property
    def vae(self):
        """Acessa o VAE de forma preguiçosa (lazy-loading) para economizar recursos."""
        if self._vae is None:
            print(">>> Inicializando VAE do worker LTX para operações Deformes...")
            self._vae = self.ltx_manager.workers[0].pipeline.vae.to(self.device)
            self._vae.eval()
        return self._vae

    @torch.no_grad()
    def pixels_to_latents(self, video_tensor: torch.Tensor) -> torch.Tensor:
        """Codifica um tensor de vídeo (pixels) para um tensor de latentes."""
        video_tensor = video_tensor.to(self.vae.device, dtype=torch.bfloat16)
        return vae_encode(video_tensor, self.vae)

    @torch.no_grad()
    def latents_to_pixels(self, latent_tensor: torch.Tensor, decode_timestep: float = 0.0) -> torch.Tensor:
        """Decodifica um tensor de latentes de volta para um tensor de vídeo (pixels)."""
        latent_tensor = latent_tensor.to(self.vae.device, dtype=torch.bfloat16)
        return vae_decode(latent_tensor, self.vae, is_video=True, timestep=decode_timestep)

    def save_video_from_tensor(self, video_tensor: torch.Tensor, path: str, fps: int = 24):
        """Salva um tensor de vídeo (B, C, F, H, W) em um arquivo .mp4."""
        video_tensor = video_tensor.squeeze(0).permute(1, 2, 3, 0)
        video_tensor = (video_tensor.clamp(-1, 1) + 1) / 2.0
        # CORREÇÃO TypeError: Converte para .float() antes de passar para a CPU e NumPy
        video_np = (video_tensor.cpu().float().numpy() * 255).astype(np.uint8)
        with imageio.get_writer(path, fps=fps, codec='libx264', quality=8) as writer:
            for frame in video_np:
                writer.append_data(frame)
        print(f">>> Vídeo salvo em: {path}")

    @torch.no_grad()
    def GenerateBaseAndQuantumTwin(
        self,
        prompt: str,
        num_frames_requested: int = 240,
        width: int = 512,
        height: int = 512,
        num_inference_steps: int = 4,
        cfg: float = 1.0,
        progress=gr.Progress()
    ) -> (str, str):
        """Etapa 1: Gera um vídeo base e seu "gêmeo quântico" latente."""
        # CORREÇÃO EinopsError: Garante compatibilidade de frames com o modelo LTX
        num_chunks = (num_frames_requested - 1) // 8
        num_frames_compatible = num_chunks * 8 + 1
        print(f"Solicitação de {num_frames_requested} frames ajustada para {num_frames_compatible}.")

        if progress: progress(0.1, desc="Gerando realidade base (vídeo)...")
        
        video_path = os.path.join(WORKSPACE_DIR, "base_video.mp4")
        latents_path = os.path.join(WORKSPACE_DIR, "quantum_twin.pt")
        
        self.ltx_manager.generate_video_fragment(
            motion_prompt=prompt, conditioning_items_data=[],
            width=width, height=height, seed=int(time.time()), cfg=cfg,
            video_total_frames=num_frames_compatible, video_fps=24, 
            num_inference_steps=int(num_inference_steps), use_attention_slicing=True,
            decode_timestep=0.05, image_cond_noise_scale=0.025,
            current_fragment_index=0, output_path=video_path, progress=progress
        )

        if progress: progress(0.8, desc="Extraindo a alma quântica (latentes)...")
        with imageio.get_reader(video_path) as reader:
            frames = np.stack([frame for frame in reader])
        
        video_tensor = torch.from_numpy(frames).permute(3, 0, 1, 2).unsqueeze(0).float() / 255.0
        video_tensor = (video_tensor * 2.0) - 1.0
        
        source_latents = self.pixels_to_latents(video_tensor)
        torch.save(source_latents, latents_path)
        
        print("Vídeo base e gêmeo quântico gerados com sucesso.")
        return video_path, latents_path

    @torch.no_grad()
    def RegenerateTransition(
        self,
        source_latents_path: str,
        x_corte: int,
        n_corte: int,
        n_eco: int,
        progress=gr.Progress()
    ) -> str:
        if not source_latents_path or not os.path.exists(source_latents_path):
            raise gr.Error("Gêmeo quântico não encontrado. Por favor, gere um vídeo base primeiro na Etapa 1.")
            
        if progress: progress(0, desc="Iniciando cirurgia latente...")
        source_latents = torch.load(source_latents_path).to(self.device)

        vae_temporal_scale = 8 
        x_corte_latente = x_corte // vae_temporal_scale
        n_corte_latente = abs(n_corte) // vae_temporal_scale
        n_eco_latente = n_eco // vae_temporal_scale
        
        if n_corte_latente < 1 or n_eco_latente < 1:
            raise gr.Error("Corte ou eco muito curtos. Aumente 'Frames a Remover' ou 'Tamanho do Eco'.")
        if x_corte_latente <= n_eco_latente and n_corte < 0:
            raise gr.Error("Ponto de corte muito próximo do início para o eco definido.")

        if progress: progress(0.1, desc="Dissecando a realidade...")
        latentes_parte1 = source_latents[:, :, :x_corte_latente, :, :]
        latentes_parte2 = source_latents[:, :, x_corte_latente:, :, :]
        if n_corte < 0:
            if latentes_parte1.shape[2] <= n_corte_latente: raise gr.Error("Corte negativo excede o tamanho da Parte 1.")
            latentes_parte1_aparada = latentes_parte1[:, :, :-n_corte_latente, :, :]
            latentes_parte2_aparada = latentes_parte2
            caminho_latente = latentes_parte1[:, :, -n_corte_latente // 2, :, :].unsqueeze(2)
        else:
            if latentes_parte2.shape[2] <= n_corte_latente: raise gr.Error("Corte positivo excede o tamanho da Parte 2.")
            latentes_parte1_aparada = latentes_parte1
            latentes_parte2_aparada = latentes_parte2[:, :, n_corte_latente:, :, :]
            caminho_latente = latentes_parte2[:, :, n_corte_latente // 2, :, :].unsqueeze(2)
        
        if latentes_parte1_aparada.shape[2] < n_eco_latente: raise gr.Error("Eco maior que a Parte 1 aparada.")
        eco_latente = latentes_parte1_aparada[:, :, -n_eco_latente:, :, :]
        destino_latente = latentes_parte2_aparada[:, :, :1, :, :]
        
        temp_dir = os.path.join(WORKSPACE_DIR, "temp_ingredients")
        if os.path.exists(temp_dir): shutil.rmtree(temp_dir)
        os.makedirs(temp_dir, exist_ok=True)
        if progress: progress(0.2, desc="Materializando ingredientes...")

        decode_timestep_val = 0.05
        eco_video_path = os.path.join(temp_dir, "eco.mp4")
        self.save_video_from_tensor(self.latents_to_pixels(eco_latente, decode_timestep=decode_timestep_val), eco_video_path)
        
        def save_latent_as_img(latent, path):
            pixels = self.latents_to_pixels(latent, decode_timestep=decode_timestep_val).squeeze(0).squeeze(1)
            pixels = (pixels.permute(1, 2, 0).clamp(-1, 1) + 1) / 2.0 * 255
            Image.fromarray(pixels.cpu().float().numpy().astype(np.uint8)).save(path)

        caminho_img_path, destino_img_path = os.path.join(temp_dir, "caminho.png"), os.path.join(temp_dir, "destino.png")
        save_latent_as_img(caminho_latente, caminho_img_path)
        save_latent_as_img(destino_latente, destino_img_path)
        
        # CORREÇÃO EinopsError (para a ponte): Garantir que a ponte tenha um número compatível de frames.
        transition_frames_req = 24
        ponte_chunks = max(1, (transition_frames_req -1) // 8)
        transition_frames_compatible = ponte_chunks * 8 + 1
        
        conditioning_items_data = [(eco_video_path, 0, 1.0), (caminho_img_path, transition_frames_compatible // 2, 0.7), (destino_img_path, transition_frames_compatible - 1, 1.0)]

        if progress: progress(0.4, desc="Regenerando a ponte latente...")
        ponte_video_path = os.path.join(temp_dir, "ponte_video.mp4")
        self.ltx_manager.generate_video_fragment(
            motion_prompt="a smooth and seamless transition", conditioning_items_data=conditioning_items_data,
            width=512, height=512, seed=int(time.time()), cfg=1.0,
            video_total_frames=transition_frames_compatible, video_fps=24, 
            num_inference_steps=4, use_attention_slicing=True,
            decode_timestep=decode_timestep_val, image_cond_noise_scale=0.025,
            current_fragment_index=1, output_path=ponte_video_path, progress=progress
        )
        with imageio.get_reader(ponte_video_path) as reader:
            frames = np.stack([frame for frame in reader])
        video_tensor = torch.from_numpy(frames).permute(3, 0, 1, 2).unsqueeze(0).float() / 255.0
        video_tensor = (video_tensor * 2.0) - 1.0
        ponte_latente = self.pixels_to_latents(video_tensor)

        if progress: progress(0.9, desc="Costurando a realidade...")
        latentes_parte1_final = latentes_parte1_aparada[:, :, :-n_eco_latente, :, :]
        
        tensor_regenerado = torch.cat([latentes_parte1_final, ponte_latente, latentes_parte2_aparada], dim=2)

        final_pixels = self.latents_to_pixels(tensor_regenerado, decode_timestep=decode_timestep_val)
        final_video_path = os.path.join(WORKSPACE_DIR, "final_video_deformes.mp4")
        self.save_video_from_tensor(final_pixels, final_video_path)
        
        shutil.rmtree(temp_dir)
        return final_video_path

# ==============================================================================
# SEÇÃO 2: INTERFACE GRÁFICA (GRADIO)
# ==============================================================================
deformes_engine = Deformes(ltx_manager_singleton)

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Deformes V1.5: Final com Correções")
    
    latent_twin_path_state = gr.State()

    with gr.Tabs():
        with gr.TabItem("Etapa 1: Geração Base"):
            prompt_input = gr.Textbox(label="Prompt para o vídeo base", value="a cinematic shot of a majestic lion walking on the savanna")
            gen_button = gr.Button("Gerar Vídeo Base", variant="primary")
            base_video_output = gr.Video(label="Vídeo Base Gerado (Analógico)")

        with gr.TabItem("Etapa 2: Edição Regenerativa"):
            editor_video_input = gr.Video(label="Vídeo para Editar")
            x_corte_slider = gr.Slider(label="Ponto de Corte (Frame)", interactive=False, step=1)
            n_corte_slider = gr.Slider(label="Frames a Remover (n_corte)", minimum=-96, maximum=96, value=24, step=1, info="< 0 remove do passado, > 0 remove do futuro")
            n_eco_slider = gr.Slider(label="Tamanho do Eco (Frames)", minimum=8, maximum=32, value=8, step=8)
            edit_button = gr.Button("▶️ Executar Regeneração", variant="primary")
            final_video_output = gr.Video(label="✨ Vídeo Final Regenerado ✨")

    def get_video_info_for_slider(video_path):
        if not video_path: return gr.update(maximum=1, value=0, interactive=False)
        try:
            command = f'ffprobe -v error -select_streams v:0 -count_frames -show_entries stream=nb_read_frames -of json "{video_path}"'
            result = subprocess.run(command, shell=True, check=True, text=True, capture_output=True)
            data = json.loads(result.stdout)["streams"][0]
            total_frames = int(data["nb_read_frames"])
            return gr.update(maximum=total_frames - 1, value=total_frames // 2, interactive=True)
        except Exception as e:
            print(f"FFprobe falhou ao analisar o vídeo para UI: {e}")
            return gr.update(maximum=1, value=0, interactive=False)
            
    def handle_generation_and_ui_update(prompt, progress=gr.Progress()):
        video_path, latent_path = deformes_engine.GenerateBaseAndQuantumTwin(prompt, progress=progress)
        slider_update = get_video_info_for_slider(video_path)
        return {
            base_video_output: video_path,
            latent_twin_path_state: latent_path,
            editor_video_input: video_path,
            x_corte_slider: slider_update
        }

    gen_button.click(
        fn=handle_generation_and_ui_update,
        inputs=[prompt_input],
        outputs=[base_video_output, latent_twin_path_state, editor_video_input, x_corte_slider]
    )

    edit_button.click(
        fn=deformes_engine.RegenerateTransition,
        inputs=[latent_twin_path_state, x_corte_slider, n_corte_slider, n_eco_slider],
        outputs=[final_video_output]
    )

if __name__ == "__main__":
    demo.queue().launch(share=True)