# app.py (Versão com fluxo de trabalho Déjà-Vu)
# Copyright (C) 4 de Agosto de 2025  Carlos Rodrigues dos Santos

import gradio as gr
import yaml
import logging
import os
import shutil

from aduc_orchestrator import AducOrchestrator

# --- 1. CONFIGURAÇÃO E INICIALIZAÇÃO ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
try:
    with open("config.yaml", 'r') as f: config = yaml.safe_load(f)
    WORKSPACE_DIR = config['application']['workspace_dir']
    aduc = AducOrchestrator(workspace_dir=WORKSPACE_DIR)
    logger.info("Orquestrador ADUC e Especialistas inicializados com sucesso.")
except Exception as e:
    logger.error(f"ERRO CRÍTICO ao inicializar: {e}", exc_info=True)
    exit()

# --- 2. WRAPPER DA UI ---
def finalizer_wrapper(video_in, image_in, prompt, duration_seconds, n_corte, n_eco, p_caminho,
                      cfg, steps, stg_scale, rescaling_scale, 
                      decode_timestep, decode_noise_scale, skip_block_list_str,
                      progress=gr.Progress()):
    try:
        if video_in is None or image_in is None: raise gr.Error("Por favor, forneça um vídeo e uma imagem.")
        if not prompt or not prompt.strip(): raise gr.Error("Por favor, forneça um prompt.")

        if os.path.exists(WORKSPACE_DIR): shutil.rmtree(WORKSPACE_DIR)
        aduc.director.reset()
        
        final_video_path = aduc.task_finalize_scene(
            video_path=video_in, image_path=image_in, prompt=prompt,
            duration_seconds=float(duration_seconds), n_corte=int(n_corte), n_eco=int(n_eco),
            p_caminho=float(p_caminho), # Novo parâmetro
            cfg=float(cfg), steps=int(steps), stg_scale=float(stg_scale),
            rescaling_scale=float(rescaling_scale), decode_timestep=float(decode_timestep),
            decode_noise_scale=float(decode_noise_scale), skip_block_list_str=skip_block_list_str,
            progress=progress
        )
        return final_video_path
    except Exception as e:
        logger.error(f"Erro no Finalizador de Cenas: {e}", exc_info=True)
        raise gr.Error(f"A finalização da cena falhou: {e}")

# --- 3. DEFINIÇÃO DA UI ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Deformes4D 🚀 - Finalizador de Cenas com Déjà-Vu")
    gr.Markdown("Crie uma transição de um vídeo para uma imagem, guiada por um prompt e pela memória do 'caminho não seguido'.")

    with gr.Row():
        video_input = gr.Video(label="Vídeo de Entrada (Início)")
        image_input = gr.Image(label="Imagem de Destino (Fim)", type="filepath")
    
    prompt_input = gr.Textbox(label="Prompt para a Transição", info="Descreva a ação que conecta o vídeo à imagem.")
    
    with gr.Accordion("Parâmetros da Transição", open=True):
        with gr.Row():
            duration_slider = gr.Slider(label="Duração da Nova Cena (s)", minimum=1.0, maximum=10.0, value=2.1, step=0.1)
            n_corte_slider = gr.Slider(label="Frames a Descartar (n_corte)", minimum=8, maximum=96, value=24, step=8)
            n_eco_slider = gr.Slider(label="Memória de Movimento (n_eco)", minimum=8, maximum=32, value=8, step=8)
        # Slider para o novo parâmetro p_caminho
        p_caminho_slider = gr.Slider(label="Força do 'Déjà-Vu' (p_caminho)", minimum=0.0, maximum=1.0, value=0.5, step=0.05, info="Quão forte a memória do último frame original deve ser no meio da transição.")

    with gr.Accordion("Controles Avançados (Opcional)", open=False):
        with gr.Row():
            cfg_slider = gr.Slider(minimum=1.0, maximum=8.0, value=1.0, step=0.1, label="CFG Scale")
            steps_slider = gr.Slider(minimum=4, maximum=30, value=4, step=1, label="Passos de Inferência")
        stg_scale_slider = gr.Slider(label="STG Scale", minimum=0.0, maximum=5.0, value=0.0, step=0.1)
        rescaling_scale_slider = gr.Slider(label="Rescaling Scale", minimum=0.0, maximum=1.0, value=1.0, step=0.05)
        decode_timestep_slider = gr.Slider(label="Decode Timestep", minimum=0.0, maximum=0.5, value=0.05, step=0.01)
        decode_noise_scale_slider = gr.Slider(label="Decode Noise Scale", minimum=0.0, maximum=0.5, value=0.025, step=0.01)
        skip_block_list_input = gr.Textbox(label="Lista de Blocos a Pular (ex: 42)", value="42")

    finalize_button = gr.Button("🎬 Gerar Transição", variant="primary")
    final_video_output = gr.Video(label="✨ Vídeo Finalizado ✨")

    # --- 4. CONEXÕES DA UI ---
    inputs = [
        video_input, image_input, prompt_input, duration_slider, n_corte_slider, n_eco_slider, p_caminho_slider,
        cfg_slider, steps_slider, stg_scale_slider, rescaling_scale_slider,
        decode_timestep_slider, decode_noise_scale_slider, skip_block_list_input
    ]
    finalize_button.click(fn=finalizer_wrapper, inputs=inputs, outputs=final_video_output)

if __name__ == "__main__":
    demo.launch()