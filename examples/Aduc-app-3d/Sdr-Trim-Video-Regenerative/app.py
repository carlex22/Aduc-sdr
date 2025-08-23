# app.py
# Versão 2.0 - Final, Corrigida e Documentada
# Editor de Transição com LTX, implementando a pipeline "Eco + Caminho + Destino".
# Corrige a lógica para o "corte positivo", garantindo um fluxo conceitual consistente.
# Parte do projeto Euia-AducSdr, Copyright (C) 2025 Carlos Rodrigues dos Santos.

import gradio as gr
import subprocess
import os
import shutil
import time
import json
import math

from ltx_manager_helpers import ltx_manager_singleton

WORKSPACE_DIR = "transition_workspace"

# ==============================================================================
# SEÇÃO 1: FUNÇÕES UTILITÁRIAS (FFMPEG HELPERS)
# ==============================================================================

def run_ffmpeg_command(command: str):
    try:
        result = subprocess.run(command, shell=True, check=True, text=True, capture_output=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        error_message = f"O FFmpeg falhou: {e.stderr}"
        print(error_message)
        raise gr.Error(error_message)

def get_video_info(video_path: str) -> dict:
    if not video_path or not os.path.exists(video_path):
        return {"total_frames": 0, "fps": 0}
    try:
        command = f'ffprobe -v error -select_streams v:0 -count_frames -show_entries stream=nb_read_frames,r_frame_rate,width,height -of json "{video_path}"'
        output = run_ffmpeg_command(command)
        data = json.loads(output)["streams"][0]
        total_frames = int(data["nb_read_frames"])
        fps_str = data["r_frame_rate"].split('/')
        fps = float(fps_str[0]) / float(fps_str[1]) if len(fps_str) == 2 and float(fps_str[1]) != 0 else 30.0
        return {"total_frames": total_frames, "fps": fps, "width": int(data.get("width", 512)), "height": int(data.get("height", 512))}
    except Exception as e:
        raise gr.Error(f"Falha ao ler informações do vídeo com ffprobe: {e}")

def split_video_at_frame(input_path: str, frame_num: int, output_prefix: str) -> (str, str):
    part1_path = f"{output_prefix}_part1.mp4"
    part2_path = f"{output_prefix}_part2.mp4"
    run_ffmpeg_command(f"ffmpeg -y -v error -i \"{input_path}\" -vf \"select='lt(n,{frame_num})'\" -an \"{part1_path}\"")
    run_ffmpeg_command(f"ffmpeg -y -v error -i \"{input_path}\" -vf \"select='gte(n,{frame_num})'\" -an \"{part2_path}\"")
    return part1_path, part2_path

def extract_frame(input_path: str, frame_num: int, output_path: str) -> str:
    run_ffmpeg_command(f"ffmpeg -y -v error -i \"{input_path}\" -vf \"select='eq(n,{frame_num})'\" -vframes 1 \"{output_path}\"")
    return output_path

def trim_video_end(input_path: str, frames_to_remove: int, output_path: str) -> str:
    info = get_video_info(input_path)
    frames_to_keep = max(1, info["total_frames"] - frames_to_remove)
    run_ffmpeg_command(f"ffmpeg -y -v error -i \"{input_path}\" -vf \"select='lt(n,{frames_to_keep})'\" -an \"{output_path}\"")
    return output_path
    
def trim_video_start(input_path: str, frames_to_remove: int, output_path: str) -> str:
    run_ffmpeg_command(f"ffmpeg -y -v error -i \"{input_path}\" -vf \"select='gte(n,{frames_to_remove})'\" -an \"{output_path}\"")
    return output_path

def extract_video_segment(input_path: str, start_frame: int, num_frames: int, output_path: str) -> str:
    run_ffmpeg_command(f"ffmpeg -y -v error -i \"{input_path}\" -vf \"select='between(n,{start_frame},{start_frame + num_frames - 1})'\" -vframes {num_frames} -an \"{output_path}\"")
    return output_path

def concatenate_videos(video_paths: list, output_path: str) -> str:
    list_file = os.path.join(os.path.dirname(output_path), f"concat_list_{time.time()}.txt")
    with open(list_file, "w") as f:
        for path in video_paths:
            f.write(f"file '{os.path.abspath(path)}'\n")
    run_ffmpeg_command(f"ffmpeg -y -v error -f concat -safe 0 -i \"{list_file}\" -c copy \"{output_path}\"")
    os.remove(list_file)
    return output_path

# ==============================================================================
# SEÇÃO 2: LÓGICA PRINCIPAL DA PIPELINE
# ==============================================================================

def process_and_generate_transition(
    input_video: str, x_corte: int, n_corte: int, n_eco: int, 
    cfg: float, num_inference_steps: int,
    progress=gr.Progress()
):
    if not input_video: raise gr.Error("Por favor, carregue um vídeo primeiro.")
    if n_corte == 0: raise gr.Error("O valor de 'Frames a Remover' (n_corte) não pode ser zero.")
    
    if os.path.exists(WORKSPACE_DIR): shutil.rmtree(WORKSPACE_DIR)
    os.makedirs(WORKSPACE_DIR, exist_ok=True)
    
    video_info = get_video_info(input_video)
    fps = video_info["fps"]
    base_res = min(video_info.get('width', 512), video_info.get('height', 512))
    VIDEO_RESOLUTION = min(base_res, 720) 
    TRANSITION_FRAMES = 24
    
    progress(0.1, desc="Passo 1: Dividindo o vídeo no ponto de corte...")
    part1_path, part2_path = split_video_at_frame(input_video, x_corte, os.path.join(WORKSPACE_DIR, "split"))
    
    progress(0.2, desc="Passo 2 & 3: Aparando clipes e definindo o 'Caminho'...")
    if n_corte < 0:
        abs_n_corte = abs(n_corte)
        caminho_frame_num = max(0, x_corte - (abs_n_corte // 2))
        caminho_path = extract_frame(input_video, caminho_frame_num, os.path.join(WORKSPACE_DIR, "caminho.png"))
        part1_trimmed_path = trim_video_end(part1_path, abs_n_corte, os.path.join(WORKSPACE_DIR, "part1_trimmed.mp4"))
        part2_trimmed_path = part2_path
    else:
        caminho_frame_num = min(video_info["total_frames"] - 1, x_corte + (n_corte // 2))
        caminho_path = extract_frame(input_video, caminho_frame_num, os.path.join(WORKSPACE_DIR, "caminho.png"))
        part1_trimmed_path = part1_path
        part2_trimmed_path = trim_video_start(part2_path, n_corte, os.path.join(WORKSPACE_DIR, "part2_trimmed.mp4"))

    yield {
        log_output: "Ingredientes para LTX extraídos. 'Caminho' criado.",
        caminho_img_output: gr.update(value=caminho_path, visible=True),
    }

    progress(0.4, desc="Passo 4 & 5: Extraindo 'Eco' (Passado) e 'Destino' (Futuro)...")
    
    # --- LÓGICA UNIFICADA E CORRIGIDA ---
    # O "Eco" sempre vem do passado (final da parte 1).
    part1_info = get_video_info(part1_trimmed_path)
    eco_start_frame = max(0, part1_info["total_frames"] - n_eco)
    eco_path = extract_video_segment(part1_trimmed_path, eco_start_frame, n_eco, os.path.join(WORKSPACE_DIR, "eco.mp4"))
    
    # O "Destino" sempre aponta para o futuro (início da parte 2).
    destino_path = extract_frame(part2_trimmed_path, 0, os.path.join(WORKSPACE_DIR, "destino.png"))
    
    # Preparamos os clipes para a concatenação final.
    clip_for_concat_1 = trim_video_end(part1_trimmed_path, n_eco, os.path.join(WORKSPACE_DIR, "concat_part1.mp4"))
    clip_for_concat_2 = part2_trimmed_path
        
    yield {
        log_output: "Ingredientes 'Eco' e 'Destino' prontos.",
        eco_vid_output: gr.update(value=eco_path, visible=True),
        destino_img_output: gr.update(value=destino_path, visible=True),
    }
    
    progress(0.5, desc="Passo 6: Gerando a ponte de transição com LTX...")
    conditioning_items_data = [
        (eco_path, 0, 1.0),
        (caminho_path, TRANSITION_FRAMES // 2, 0.65),
        (destino_path, TRANSITION_FRAMES - 1, 1.0)
    ]
    transition_video_path = os.path.join(WORKSPACE_DIR, "generated_transition.mp4")
    
    ltx_manager_singleton.generate_video_fragment(
        motion_prompt="a smooth and seamless transition",
        conditioning_items_data=conditioning_items_data,
        width=VIDEO_RESOLUTION, height=VIDEO_RESOLUTION,
        seed=int(time.time()), cfg=cfg, video_total_frames=TRANSITION_FRAMES,
        video_fps=fps, num_inference_steps=int(num_inference_steps), use_attention_slicing=True,
        decode_timestep=0.05, image_cond_noise_scale=0.025,
        current_fragment_index=1, output_path=transition_video_path, progress=progress
    )

    yield {
        log_output: f"Transição gerada com sucesso: {os.path.basename(transition_video_path)}",
        transition_vid_output: gr.update(value=transition_video_path, visible=True),
    }
    
    progress(0.9, desc="Passo 7: Montando o vídeo final...")
    final_video_path = os.path.join(WORKSPACE_DIR, "masterpiece.mp4")
    video_list_to_concat = [clip_for_concat_1, transition_video_path, clip_for_concat_2]
    final_result = concatenate_videos(video_list_to_concat, final_video_path)
    
    progress(1.0, desc="Concluído!")
    yield {
        log_output: f"Vídeo final montado com sucesso! Salvo em: {final_result}",
        final_video_output: gr.update(value=final_result, visible=True),
    }

# ==============================================================================
# SEÇÃO 3: INTERFACE GRÁFICA (GRADIO)
# ==============================================================================

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Editor de Transição com LTX (Demonstração de Corte + Eco)")
    gr.Markdown("Carregue um vídeo e a interface se adaptará a ele. Defina um ponto de corte e os parâmetros de remoção e eco. O sistema usará LTX para gerar uma transição suave.")

    with gr.Row():
        with gr.Column(scale=1):
            input_video_component = gr.Video(label="1. Carregue seu Vídeo")
            gr.Markdown("O slider de corte será ativado e ajustado automaticamente após o upload.")
            x_corte_slider = gr.Slider(label="2. Ponto de Corte (Frame)", interactive=False, step=1)
            n_corte_slider = gr.Slider(label="3. Frames a Remover (n_corte)", minimum=-200, maximum=200, value=50, step=1, info="< 0 remove do 1º clipe, > 0 remove do 2º")
            n_eco_slider = gr.Slider(label="4. Tamanho do Eco (n_eco)", minimum=4, maximum=48, value=8, step=1, info="Frames usados como 'memória' de movimento")
            with gr.Accordion("Controles Avançados do LTX", open=False):
                 cfg_slider = gr.Slider(label="CFG (Guidance Scale)", minimum=0.5, maximum=10.0, value=1.5, step=0.1)
                 num_inference_steps_slider = gr.Slider(label="Etapas de Inferência", minimum=10, maximum=50, value=25, step=1)
            generate_button = gr.Button("▶️ Gerar Transição", variant="primary")
        with gr.Column(scale=2):
            log_output = gr.Textbox(label="Log de Processamento", lines=5, interactive=False)
            gr.Markdown("### Ingredientes para a Transição (Gerados)")
            with gr.Row():
                eco_vid_output = gr.Video(label="Eco (Início)", interactive=False, visible=False)
                caminho_img_output = gr.Image(label="Caminho (Meio)", interactive=False, visible=False)
                destino_img_output = gr.Image(label="Destino (Fim)", interactive=False, visible=False)
            transition_vid_output = gr.Video(label="Ponte de Transição Gerada pelo LTX", interactive=False, visible=False)
            final_video_output = gr.Video(label="✨ Obra-Prima Final ✨", interactive=False, visible=False)

    def analyze_video_on_upload(video_path):
        if not video_path: return gr.update(interactive=False, value=0, maximum=1)
        info = get_video_info(video_path)
        return gr.update(maximum=info["total_frames"] - 1, value=math.floor(info["total_frames"] / 2), interactive=True)
    input_video_component.upload(fn=analyze_video_on_upload, inputs=[input_video_component], outputs=[x_corte_slider])
    
    def clear_outputs():
        return {log_output: "", eco_vid_output: gr.update(value=None, visible=False), caminho_img_output: gr.update(value=None, visible=False), destino_img_output: gr.update(value=None, visible=False), transition_vid_output: gr.update(value=None, visible=False), final_video_output: gr.update(value=None, visible=False)}
    input_video_component.clear(fn=clear_outputs, outputs=[log_output, eco_vid_output, caminho_img_output, destino_img_output, transition_vid_output, final_video_output])

    generate_button.click(
        fn=process_and_generate_transition,
        inputs=[input_video_component, x_corte_slider, n_corte_slider, n_eco_slider, cfg_slider, num_inference_steps_slider],
        outputs=[log_output, caminho_img_output, eco_vid_output, destino_img_output, transition_vid_output, final_video_output]
    )

if __name__ == "__main__":
    if os.path.exists(WORKSPACE_DIR): shutil.rmtree(WORKSPACE_DIR)
    os.makedirs(WORKSPACE_DIR, exist_ok=True)
    demo.queue().launch(share=True)