# Euia-AducSdr: Uma implementação aberta e funcional da arquitetura ADUC-SDR para geração de vídeo coerente.
# Copyright (C) 4 de Agosto de 2025  Carlos Rodrigues dos Santos
#
# Contato:
# Carlos Rodrigues dos Santos
# carlex22@gmail.com
# Rua Eduardo Carlos Pereira, 4125, B1 Ap32, Curitiba, PR, Brazil, CEP 8102025
#
# Repositórios e Projetos Relacionados:
# GitHub: https://github.com/carlex22/Aduc-sdr
# Hugging Face: https://huggingface.co/spaces/Carlexx/Ltx-SuperTime-60Secondos/
# Hugging Face: https://huggingface.co/spaces/Carlexxx/Novinho/
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

# --- app.py (NOVINHO-4.7: Arquitetura Otimizada com Handoff Cinético) ---

# --- Ato 1: A Convocação da Orquestra (Importações) ---
import gradio as gr
import torch
import os
import yaml
from PIL import Image, ImageOps, ExifTags
import shutil
import gc
import subprocess
import google.generativeai as genai
import numpy as np
import imageio
from pathlib import Path
import huggingface_hub
import json
import time

from inference import create_ltx_video_pipeline, load_image_to_tensor_with_resize_and_crop, ConditioningItem, calculate_padding
from dreamo_helpers import dreamo_generator_singleton

# --- Ato 2: A Preparação do Palco (Configurações) ---
config_file_path = "configs/ltxv-13b-0.9.8-distilled.yaml"
with open(config_file_path, "r") as file: PIPELINE_CONFIG_YAML = yaml.safe_load(file)

LTX_REPO = "Lightricks/LTX-Video"
models_dir = "downloaded_models_gradio"
Path(models_dir).mkdir(parents=True, exist_ok=True)
WORKSPACE_DIR = "aduc_workspace"
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

VIDEO_FPS = 30
VIDEO_DURATION_SECONDS = 3.5
VIDEO_TOTAL_FRAMES = VIDEO_DURATION_SECONDS * VIDEO_FPS
TARGET_RESOLUTION = 720

print("Criando pipelines LTX na CPU (estado de repouso)...")
distilled_model_actual_path = huggingface_hub.hf_hub_download(repo_id=LTX_REPO, filename=PIPELINE_CONFIG_YAML["checkpoint_path"], local_dir=models_dir, local_dir_use_symlinks=False)
pipeline_instance = create_ltx_video_pipeline(
    ckpt_path=distilled_model_actual_path,
    precision=PIPELINE_CONFIG_YAML["precision"],
    text_encoder_model_name_or_path=PIPELINE_CONFIG_YAML["text_encoder_model_name_or_path"],
    sampler=PIPELINE_CONFIG_YAML["sampler"],
    device='cpu'
)
print("Modelos LTX prontos (na CPU).")


# --- Ato 3: As Partituras dos Músicos (Funções de Geração e Análise) ---

# --- Funções da ETAPA 1 (Roteiro) - Otimizadas para uma única chamada de IA ---
def robust_json_parser(raw_text: str) -> dict:
    """Um parser robusto que encontra e decodifica o primeiro objeto JSON válido em uma string."""
    try:
        start_index = raw_text.find('{')
        end_index = raw_text.rfind('}')
        if start_index != -1 and end_index != -1 and end_index > start_index:
            json_str = raw_text[start_index : end_index + 1]
            return json.loads(json_str)
        else: raise ValueError("Nenhum objeto JSON válido encontrado na resposta da IA.")
    except json.JSONDecodeError as e:
        raise ValueError(f"Falha ao decodificar JSON: {e}")

def extract_image_exif(image_path: str) -> str:
    """Extrai metadados EXIF de uma imagem e os formata como uma string."""
    try:
        img = Image.open(image_path); exif_data = img._getexif()
        if not exif_data: return "No EXIF metadata found."
        exif = { ExifTags.TAGS[k]: v for k, v in exif_data.items() if k in ExifTags.TAGS }
        relevant_tags = ['DateTimeOriginal', 'Model', 'LensModel', 'FNumber', 'ExposureTime', 'ISOSpeedRatings', 'FocalLength']
        metadata_str = ", ".join(f"{key}: {exif[key]}" for key in relevant_tags if key in exif)
        return metadata_str if metadata_str else "No relevant EXIF metadata found."
    except Exception: return "Could not read EXIF data."

def run_storyboard_generation(num_fragments: int, prompt: str, initial_image_path: str):
    """Orquestra a geração do roteiro em UMA ÚNICA ETAPA, combinando análise de visão e criação de roteiro."""
    if not initial_image_path: raise gr.Error("Por favor, forneça uma imagem de referência inicial.")
    if not GEMINI_API_KEY: raise gr.Error("Chave da API Gemini não configurada!")

    exif_metadata = extract_image_exif(initial_image_path)
    prompt_file = "prompts/unified_storyboard_prompt.txt"
    with open(os.path.join(os.path.dirname(__file__), prompt_file), "r", encoding="utf-8") as f: template = f.read()
    director_prompt = template.format(user_prompt=prompt, num_fragments=int(num_fragments), image_metadata=exif_metadata)
    
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-2.0-flash')
    img = Image.open(initial_image_path)
    print("Gerando roteiro com análise de visão integrada...")
    response = model.generate_content([director_prompt, img])
    
    try:
        storyboard_data = robust_json_parser(response.text)
        storyboard = storyboard_data.get("scene_storyboard", [])
        if not storyboard or len(storyboard) != int(num_fragments):
            raise ValueError(f"A IA não gerou o número correto de cenas. Esperado: {num_fragments}, Recebido: {len(storyboard)}")
        return storyboard
    except Exception as e:
        raise gr.Error(f"O Roteirista (Gemini) falhou ao criar o roteiro: {e}. Resposta recebida: {response.text}")


# --- Funções da ETAPA 2 (Keyframes) ---
def get_dreamo_prompt_for_transition(previous_image_path: str, target_scene_description: str) -> str:
    genai.configure(api_key=GEMINI_API_KEY)
    prompt_file = "prompts/img2img_evolution_prompt.txt"
    with open(os.path.join(os.path.dirname(__file__), prompt_file), "r", encoding="utf-8") as f: template = f.read()
    director_prompt = template.format(target_scene_description=target_scene_description)
    model = genai.GenerativeModel('gemini-2.0-flash'); img = Image.open(previous_image_path)
    response = model.generate_content([director_prompt, "Previous Image:", img])
    return response.text.strip().replace("\"", "")

def run_keyframe_generation(storyboard, initial_ref_image_path, sequential_ref_task, progress=gr.Progress()):
    if not storyboard: raise gr.Error("Nenhum roteiro para gerar keyframes.")
    if not initial_ref_image_path or not os.path.exists(initial_ref_image_path): raise gr.Error("A imagem de referência principal é obrigatória.")
    log_history = ""
    try:
        pipeline_instance.to('cpu'); gc.collect(); torch.cuda.empty_cache()
        dreamo_generator_singleton.to_gpu()
        with Image.open(initial_ref_image_path) as img: width, height = (img.width // 32) * 32, (img.height // 32) * 32
        keyframe_paths, current_ref_image_path = [initial_ref_image_path], initial_ref_image_path
        for i, scene_description in enumerate(storyboard):
            progress(i / len(storyboard), desc=f"Pintando Keyframe {i+1}/{len(storyboard)}")
            log_history += f"\n--- PINTANDO KEYFRAME {i+1}/{len(storyboard)} ---\n"
            dreamo_prompt = get_dreamo_prompt_for_transition(current_ref_image_path, scene_description)
            log_history += f"  - Roteiro: '{scene_description}'\n  - Base: {os.path.basename(current_ref_image_path)}\n  - Prompt do D.A.: \"{dreamo_prompt}\"\n"
            yield {keyframe_log_output: gr.update(value=log_history), keyframe_gallery_output: gr.update(value=keyframe_paths)}
            reference_items = [{'image_np': np.array(Image.open(current_ref_image_path).convert("RGB")), 'task': sequential_ref_task}]
            output_path = os.path.join(WORKSPACE_DIR, f"keyframe_{i+1}.png")
            image = dreamo_generator_singleton.generate_image_with_gpu_management(reference_items=reference_items, prompt=dreamo_prompt, width=width, height=height)
            image.save(output_path)
            keyframe_paths.append(output_path)
            current_ref_image_path = output_path
            yield {keyframe_log_output: gr.update(value=log_history), keyframe_gallery_output: gr.update(value=keyframe_paths)}
    except Exception as e: raise gr.Error(f"O Pintor (DreamO) ou Diretor de Arte (Gemini) falhou: {e}")
    finally:
        dreamo_generator_singleton.to_cpu(); gc.collect(); torch.cuda.empty_cache()
    log_history += "\nPintura de todos os keyframes concluída.\n"
    yield {keyframe_log_output: gr.update(value=log_history), keyframe_gallery_output: gr.update(value=keyframe_paths), keyframe_images_state: keyframe_paths}


# --- Funções da ETAPA 3 (Produção de Vídeo) ---
def get_dynamic_motion_prompt(user_prompt: str, story_history: str, memory_image_path: str, path_image_path: str, destination_image_path: str, path_scene_desc: str, dest_scene_desc: str):
    if not GEMINI_API_KEY: raise gr.Error("Chave da API Gemini não configurada!")
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel('gemini-2.0-flash')
        prompt_file = "prompts/dynamic_motion_prompt.txt"
        with open(os.path.join(os.path.dirname(__file__), prompt_file), "r", encoding="utf-8") as f: template = f.read()
        cinematographer_prompt = template.format(user_prompt=user_prompt, story_history=story_history, midpoint_scene_description=path_scene_desc, destination_scene_description=dest_scene_desc)
        mem_img, path_img, dest_img = Image.open(memory_image_path), Image.open(path_image_path), Image.open(destination_image_path)
        model_contents = ["START Image (Memory):", mem_img, "MIDPOINT Image (Path):", path_img, "DESTINATION Image (Destination):", dest_img, cinematographer_prompt]
        response = model.generate_content(model_contents)
        return response.text.strip()
    except Exception as e:
        raise gr.Error(f"O Cineasta de IA (Gemini) falhou: {e}. Resposta: {getattr(e, 'text', 'No text available.')}")

def run_video_production(prompt_geral, keyframe_images_state, scene_storyboard, seed, cfg, cut_frames_value, progress=gr.Progress()):
    if not keyframe_images_state or len(keyframe_images_state) < 2: raise gr.Error("Pinte pelo menos um keyframe (para um total de 2+ imagens) na Etapa 2.")
    log_history = "\n--- FASE 3/4: Iniciando Produção com 'Handoff Cinético'...\n"
    yield {production_log_output: log_history, video_gallery_glitch: []}
    
    MID_COND_FRAME, MID_COND_STRENGTH = 54, 0.5
    END_COND_FRAME = VIDEO_TOTAL_FRAMES - 8

    target_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    try:
        pipeline_instance.to(target_device)
        video_fragments, story_history = [], ""
        kinetic_memory_path = keyframe_images_state[0]
        with Image.open(keyframe_images_state[0]) as img: width, height = img.size
        
        num_transitions = len(keyframe_images_state) - 1
        for i in range(num_transitions):
            fragment_num = i + 1
            progress(i / num_transitions, desc=f"Filmando Fragmento {fragment_num}/{num_transitions}")
            
            memory_path, path_path, destination_path = kinetic_memory_path, keyframe_images_state[i], keyframe_images_state[i+1]
            path_scene_desc = scene_storyboard[i]
            dest_scene_desc = scene_storyboard[i+1] if (i+1) < len(scene_storyboard) else "Final scene"
            
            log_history += f"\n--- FRAGMENTO {fragment_num} ---\n  - Memória Cinética: {os.path.basename(memory_path)}\n  - Caminho: {os.path.basename(path_path)}\n  - Destino: {os.path.basename(destination_path)}\n"
            current_motion_prompt = get_dynamic_motion_prompt(prompt_geral, story_history, memory_path, path_path, destination_path, path_scene_desc, dest_scene_desc)
            story_history += f"\n- Ato {fragment_num}: {current_motion_prompt}"
            log_history += f"  - Instrução do Cineasta: '{current_motion_prompt}'\n"; yield {production_log_output: log_history}

            if i == 0:
                conditioning_items_data = [(memory_path, 0, 1.0), (destination_path, END_COND_FRAME, 1.0)]
            else:
                conditioning_items_data = [(memory_path, 0, 1.0), (path_path, MID_COND_FRAME, MID_COND_STRENGTH), (destination_path, END_COND_FRAME, 1.0)]

            full_fragment_path, _ = run_ltx_animation(fragment_num, current_motion_prompt, conditioning_items_data, width, height, seed, cfg, progress)
            
            is_last_fragment = (i == num_transitions - 1)
            if not is_last_fragment:
                trimmed_fragment_path = os.path.join(WORKSPACE_DIR, f"fragment_{fragment_num}_trimmed.mp4")
                trim_video_to_frames(full_fragment_path, trimmed_fragment_path, int(cut_frames_value))
                eco_output_path = os.path.join(WORKSPACE_DIR, f"eco_from_frag_{fragment_num}.png")
                kinetic_memory_path = extract_last_frame_as_image(trimmed_fragment_path, eco_output_path)
                video_fragments.append(trimmed_fragment_path)
                log_history += f"  - Gerado e cortado. Novo Eco Dinâmico: {os.path.basename(kinetic_memory_path)}\n"
            else:
                video_fragments.append(full_fragment_path)
                log_history += "  - Último fragmento gerado, mantendo a duração total.\n"
            yield {production_log_output: log_history, video_gallery_glitch: video_fragments}
            
        progress(1.0, desc="Produção Concluída.")
        yield {production_log_output: log_history, video_gallery_glitch: video_fragments, fragment_list_state: video_fragments}
    finally:
        pipeline_instance.to('cpu'); gc.collect(); torch.cuda.empty_cache()

# --- Funções Utilitárias e de Pós-Produção ---
def process_image_to_square(image_path: str, size: int = TARGET_RESOLUTION) -> str:
    if not image_path or not os.path.exists(image_path): return None
    try:
        img = Image.open(image_path).convert("RGB"); img_square = ImageOps.fit(img, (size, size), Image.Resampling.LANCZOS)
        output_path = os.path.join(WORKSPACE_DIR, f"initial_ref_{size}x{size}.png"); img_square.save(output_path)
        return output_path
    except Exception as e: raise gr.Error(f"Falha ao processar a imagem de referência: {e}")

def load_conditioning_tensor(media_path: str, height: int, width: int) -> torch.Tensor:
    return load_image_to_tensor_with_resize_and_crop(media_path, height, width)

def run_ltx_animation(current_fragment_index, motion_prompt, conditioning_items_data, width, height, seed, cfg, progress=gr.Progress()):
    progress(0, desc=f"[Câmera LTX] Filmando Cena {current_fragment_index}...");
    output_path = os.path.join(WORKSPACE_DIR, f"fragment_{current_fragment_index}_full.mp4"); target_device = pipeline_instance.device
    conditioning_items = [ConditioningItem(load_conditioning_tensor(p, height, width).to(target_device), s, t) for p, s, t in conditioning_items_data]
    actual_num_frames = int(round((float(VIDEO_TOTAL_FRAMES) - 1.0) / 8.0) * 8 + 1)
    padded_h, padded_w = ((height - 1) // 32 + 1) * 32, ((width - 1) // 32 + 1) * 32
    padding_vals = calculate_padding(height, width, padded_h, padded_w)
    for item in conditioning_items: item.media_item = torch.nn.functional.pad(item.media_item, padding_vals)
    kwargs = {"prompt": motion_prompt, "negative_prompt": "blurry, distorted, bad quality, artifacts", "height": padded_h, "width": padded_w, "num_frames": actual_num_frames, "frame_rate": VIDEO_FPS, "generator": torch.Generator(device=target_device).manual_seed(int(seed) + current_fragment_index), "output_type": "pt", "guidance_scale": float(cfg), "timesteps": PIPELINE_CONFIG_YAML.get("first_pass", {}).get("timesteps"), "conditioning_items": conditioning_items, "decode_timestep": PIPELINE_CONFIG_YAML.get("decode_timestep"), "decode_noise_scale": PIPELINE_CONFIG_YAML.get("decode_noise_scale"), "stochastic_sampling": PIPELINE_CONFIG_YAML.get("stochastic_sampling"), "image_cond_noise_scale": 0.15, "is_video": True, "vae_per_channel_normalize": True, "mixed_precision": (PIPELINE_CONFIG_YAML.get("precision") == "mixed_precision"), "enhance_prompt": False, "decode_every": 4}
    result_tensor = pipeline_instance(**kwargs).images
    pad_l, pad_r, pad_t, pad_b = map(int, padding_vals); slice_h = -pad_b if pad_b > 0 else None; slice_w = -pad_r if pad_r > 0 else None
    cropped_tensor = result_tensor[:, :, :VIDEO_TOTAL_FRAMES, pad_t:slice_h, pad_l:slice_w]; video_np = (cropped_tensor[0].permute(1, 2, 3, 0).cpu().float().numpy() * 255).astype(np.uint8)
    with imageio.get_writer(output_path, fps=VIDEO_FPS, codec='libx264', quality=8) as writer:
        for i, frame in enumerate(video_np): writer.append_data(frame)
    return output_path, actual_num_frames

def trim_video_to_frames(input_path: str, output_path: str, frames_to_keep: int) -> str:
    try:
        subprocess.run(f"ffmpeg -y -v error -i \"{input_path}\" -vf \"select='lt(n,{frames_to_keep})'\" -an \"{output_path}\"", shell=True, check=True, text=True)
        return output_path
    except subprocess.CalledProcessError as e: raise gr.Error(f"FFmpeg falhou ao cortar vídeo: {e.stderr}")

def extract_last_frame_as_image(video_path: str, output_image_path: str) -> str:
    try:
        subprocess.run(f"ffmpeg -y -v error -sseof -1 -i \"{video_path}\" -update 1 -q:v 1 \"{output_image_path}\"", shell=True, check=True, text=True)
        return output_image_path
    except subprocess.CalledProcessError as e: raise gr.Error(f"FFmpeg falhou ao extrair último frame: {e.stderr}")

def concatenate_and_trim_masterpiece(fragment_paths: list, progress=gr.Progress()):
    if not fragment_paths: raise gr.Error("Nenhum fragmento de vídeo para concatenar.")
    progress(0.5, desc="Montando a obra-prima final...");
    try:
        list_file_path = os.path.join(WORKSPACE_DIR, "concat_list.txt"); final_output_path = os.path.join(WORKSPACE_DIR, "masterpiece_final.mp4")
        with open(list_file_path, "w") as f:
            for p in fragment_paths: f.write(f"file '{os.path.abspath(p)}'\n")
        subprocess.run(f"ffmpeg -y -v error -f concat -safe 0 -i \"{list_file_path}\" -c copy \"{final_output_path}\"", shell=True, check=True, text=True)
        progress(1.0, desc="Montagem concluída!")
        return final_output_path
    except subprocess.CalledProcessError as e: raise gr.Error(f"FFmpeg falhou na concatenação final: {e.stderr}")

# --- Ato 5: A Interface com o Mundo (UI) ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# NOVINHO-4.7 (Arquitetura Otimizada com Handoff Cinético)\n*By Carlex & Gemini & DreamO*")
    
    if os.path.exists(WORKSPACE_DIR): shutil.rmtree(WORKSPACE_DIR)
    os.makedirs(WORKSPACE_DIR); Path("prompts").mkdir(exist_ok=True)
    
    scene_storyboard_state, keyframe_images_state, fragment_list_state = gr.State([]), gr.State([]), gr.State([])
    prompt_geral_state, processed_ref_path_state = gr.State(""), gr.State("")

    gr.Markdown("--- \n ## ETAPA 1: O ROTEIRO (IA Roteirista)")
    with gr.Row():
        with gr.Column(scale=1):
            prompt_input = gr.Textbox(label="Ideia Geral (Prompt)")
            num_fragments_input = gr.Slider(2, 10, 4, step=1, label="Número de Atos (Keyframes)")
            image_input = gr.Image(type="filepath", label=f"Imagem de Referência Principal (será {TARGET_RESOLUTION}x{TARGET_RESOLUTION})")
            director_button = gr.Button("▶️ 1. Gerar Roteiro", variant="primary")
        with gr.Column(scale=2): storyboard_to_show = gr.JSON(label="Roteiro de Cenas Gerado (em Inglês)")

    gr.Markdown("--- \n ## ETAPA 2: OS KEYFRAMES (IA Pintor & Diretor de Arte)")
    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown("O Diretor de Arte (IA) gerará prompts dinamicamente para cada Keyframe.")
            with gr.Group():
                with gr.Row():
                    ref_image_inputs_auto = gr.Image(label="Referência Sequencial (Automática)", type="filepath", interactive=False)
                    ref_task_input = gr.Dropdown(choices=["ip", "id", "style"], value="ip", label="Tarefa da Referência")
            photographer_button = gr.Button("▶️ 2. Pintar Imagens-Chave em Cadeia", variant="primary")
        with gr.Column(scale=1):
            keyframe_log_output = gr.Textbox(label="Diário de Bordo do Pintor", lines=15, interactive=False)
            keyframe_gallery_output = gr.Gallery(label="Imagens-Chave Pintadas", object_fit="contain", height="auto", type="filepath")

    gr.Markdown("--- \n ## ETAPA 3: A PRODUÇÃO (IA Cineasta & Câmera)")
    with gr.Row():
        with gr.Column(scale=1):
            with gr.Row(): seed_number = gr.Number(42, label="Seed"); cfg_slider = gr.Slider(1.0, 10.0, 2.5, step=0.1, label="CFG")
            cut_frames_slider = gr.Slider(label="Duração do Fragmento (Frames)", minimum=36, maximum=VIDEO_TOTAL_FRAMES, value=72, step=1)
            animator_button = gr.Button("▶️ 3. Produzir Cenas (Handoff Cinético)", variant="primary")
            production_log_output = gr.Textbox(label="Diário de Bordo da Produção", lines=15, interactive=False)
        with gr.Column(scale=1): video_gallery_glitch = gr.Gallery(label="Fragmentos Gerados", object_fit="contain", height="auto", type="video")
    
    gr.Markdown(f"--- \n ## ETAPA 4: PÓS-PRODUÇÃO (IA Editor)")
    editor_button = gr.Button("▶️ 4. Montar Vídeo Final", variant="primary")
    final_video_output = gr.Video(label="A Obra-Prima Final", width=TARGET_RESOLUTION)

    gr.Markdown(
        """
        ---
        ### A Arquitetura: Handoff Cinético
        Nossa arquitetura é inspirada no conceito de "Handoff" da engenharia, como em uma corrida de revezamento. Cada fragmento de vídeo passa o "bastão" para o próximo de forma fluida, garantindo um movimento contínuo e ininterrupto.

        *   **O Bastão (O `Eco`):** Em vez de terminar em uma imagem estática, cada fragmento é cortado enquanto ainda está em movimento. O último frame deste clipe cortado, o `Eco`, carrega a "energia cinética" da cena.

        *   **O Handoff (A Geração):** O próximo fragmento é forçado a começar a partir deste `Eco` dinâmico. Isso garante a herança da "física" do movimento, iluminação e composição da cena anterior.

        *   **A Sincronização (O Cineasta de IA):** Para cada Handoff, o Cineasta de IA (`Γ`) analisa o ponto de partida (`Eco`), o caminho (`Keyframe` anterior) e o destino (`Keyframe` futuro) para criar uma instrução de movimento precisa, sincronizando a transição.

        O resultado é um vídeo que flui como um único plano-sequência, em vez de uma série de clipes colados.
        """
    )
    
    director_button.click(fn=run_storyboard_generation, inputs=[num_fragments_input, prompt_input, image_input], outputs=[scene_storyboard_state]).success(fn=lambda s, p: (s, p), inputs=[scene_storyboard_state, prompt_input], outputs=[storyboard_to_show, prompt_geral_state]).success(fn=process_image_to_square, inputs=[image_input], outputs=[processed_ref_path_state]).success(fn=lambda p: p, inputs=[processed_ref_path_state], outputs=[ref_image_inputs_auto])
    photographer_button.click(fn=run_keyframe_generation, inputs=[scene_storyboard_state, processed_ref_path_state, ref_task_input], outputs=[keyframe_log_output, keyframe_gallery_output, keyframe_images_state])
    animator_button.click(fn=run_video_production, inputs=[prompt_geral_state, keyframe_images_state, scene_storyboard_state, seed_number, cfg_slider, cut_frames_slider], outputs=[production_log_output, video_gallery_glitch, fragment_list_state])
    editor_button.click(fn=concatenate_and_trim_masterpiece, inputs=[fragment_list_state], outputs=[final_video_output])

if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", share=True)