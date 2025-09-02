#Uma implementação aberta e funcional da arquitetura ADUC-SDR para geração de vídeo coerente.
# Copyright (C) 4 de Agosto de 2025  Carlos Rodrigues dos Santos
#
#Versao: 1.5
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

import gradio as gr
import yaml
import logging
import os
import sys
import shutil
import time
import json

from aduc_orchestrator import AducOrchestrator

# --- 1. CONFIGURAÇÃO E INICIALIZAÇÃO ---

# Configuração de logging para um arquivo e para o console.
LOG_FILE_PATH = "aduc_log.txt"
if os.path.exists(LOG_FILE_PATH):
    os.remove(LOG_FILE_PATH)

log_format = '%(asctime)s - %(levelname)s - [%(name)s:%(funcName)s] - %(message)s'
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
root_logger.handlers.clear()

stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(logging.Formatter(log_format))
root_logger.addHandler(stream_handler)

file_handler = logging.FileHandler(LOG_FILE_PATH, mode='w', encoding='utf-8')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter(log_format))
root_logger.addHandler(file_handler)

logger = logging.getLogger(__name__)

# Carrega os textos de internacionalização (i18n) para a UI.
i18n = {}
try:
    with open("i18n.json", "r", encoding="utf-8") as f:
        i18n = json.load(f)
except Exception as e:
    logger.error(f"Erro ao carregar i18n.json: {e}")
    i18n = {"pt": {}, "en": {}, "zh": {}} # Fallback

# Garante que as chaves de idioma existam para evitar erros.
if 'pt' not in i18n: i18n['pt'] = i18n.get('en', {})
if 'en' not in i18n: i18n['en'] = {}
if 'zh' not in i18n: i18n['zh'] = i18n.get('en', {})

# Inicializa o Orquestrador ADUC a partir do arquivo de configuração.
try:
    with open("config.yaml", 'r') as f: config = yaml.safe_load(f)
    WORKSPACE_DIR = config['application']['workspace_dir']
    aduc = AducOrchestrator(workspace_dir=WORKSPACE_DIR)
    logger.info("Orquestrador ADUC e Especialistas inicializados com sucesso.")
except Exception as e:
    logger.error(f"ERRO CRÍTICO ao inicializar: {e}", exc_info=True)
    exit()

# --- 2. WRAPPERS DA UI (Funções de Interface com o Orquestrador) ---

def run_mode_a_wrapper(prompt, num_keyframes, ref_files, resolution_str, duration_per_fragment, progress=gr.Progress()):
    # ... (código existente sem alterações)
    if not ref_files: 
        raise gr.Error("Por favor, forneça pelo menos uma imagem de referência.")
    
    ref_paths = [aduc.process_image_for_story(f.name, 480, f"ref_processed_{i}.png") for i, f in enumerate(ref_files)]
    
    progress(0.1, desc="Gerando roteiro...")
    storyboard, initial_ref_path, _ = aduc.task_generate_storyboard(prompt, num_keyframes, ref_paths, progress)
    
    resolution = int(resolution_str.split('x')[0])

    def cb_factory(scene_index, total_scenes):
        start_time = time.time()
        total_steps = 12
        def callback(pipe_self, step, timestep, callback_kwargs):
            elapsed = time.time() - start_time
            current_step = step + 1
            if current_step > 0:
                it_per_sec = current_step / elapsed
                eta = (total_steps - current_step) / it_per_sec if it_per_sec > 0 else 0
                desc = f"Keyframe {scene_index}/{total_scenes}: {int((current_step/total_steps)*100)}% | {current_step}/{total_steps} [{elapsed:.0f}s<{eta:.0f}s, {it_per_sec:.2f}it/s]"
                base_progress = 0.2 + (scene_index - 1) * (0.8 / total_scenes)
                step_progress = (current_step / total_steps) * (0.8 / total_scenes)
                progress(base_progress + step_progress, desc=desc)
            return {}
        return callback
    
    final_keyframes = aduc.task_generate_keyframes(storyboard, initial_ref_path, prompt, resolution, cb_factory)
    
    return gr.update(value=storyboard), gr.update(value=final_keyframes), gr.update(visible=True, open=True)

def run_mode_b_wrapper(prompt, num_keyframes, ref_files, progress=gr.Progress()):
    # ... (código existente sem alterações)
    if not ref_files or len(ref_files) < 2: 
        raise gr.Error("Modo Fotógrafo requer pelo menos 2 imagens: uma base e uma para o banco de cenas.")

    base_ref_paths = [aduc.process_image_for_story(ref_files[0].name, 480, "base_ref_processed_0.png")]
    pool_ref_paths = [aduc.process_image_for_story(f.name, 480, f"pool_ref_{i+1}.png") for i, f in enumerate(ref_files[1:])]

    progress(0.1, desc="Gerando roteiro...")
    storyboard, _, _ = aduc.task_generate_storyboard(prompt, num_keyframes, base_ref_paths, progress)
    
    progress(0.5, desc="IA (Fotógrafo) está selecionando as melhores cenas...")
    selected_keyframes = aduc.task_select_keyframes(storyboard, base_ref_paths, pool_ref_paths)
    
    return gr.update(value=storyboard), gr.update(value=selected_keyframes), gr.update(visible=True, open=True)

def run_video_production_wrapper(keyframes, prompt, duration, 
                                 trim_percent,
                                 handler_strength, destination_convergence_strength,
                                 video_resolution, use_cont,
                                 progress=gr.Progress()):
    # ... (código existente sem alterações)
    yield {
        video_fragments_gallery: gr.update(value=None, visible=True),
        final_video_output: gr.update(value=None, visible=True, label="🎬 Produzindo seu filme... Por favor, aguarde.")
    }
    
    resolution = int(video_resolution.split('x')[0])

    video_fragments_so_far = []
    final_movie_path = None
    
    for update in aduc.task_produce_final_movie_with_feedback(
        keyframes, prompt, duration, 
        int(trim_percent),
        handler_strength, destination_convergence_strength,
        resolution, use_cont, progress
    ):
        if "fragment_path" in update and update["fragment_path"]:
            video_fragments_so_far.append(update["fragment_path"])
            yield { video_fragments_gallery: gr.update(value=video_fragments_so_far), final_video_output: gr.update() }
        elif "final_path" in update and update["final_path"]:
            final_movie_path = update["final_path"]
            break
    
    yield {
        video_fragments_gallery: gr.update(),
        final_video_output: gr.update(value=final_movie_path, label="🎉 FILME COMPLETO 🎉")
    }

def get_log_content():
    # ... (código existente sem alterações)
    try:
        with open(LOG_FILE_PATH, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "Arquivo de log ainda não criado. Inicie uma geração."

def update_ui_language(lang_code):
    # ... (código existente com novos componentes adicionados)
    lang_map = i18n.get(lang_code, i18n.get('en', {}))
    # Mapeia os textos para cada componente da UI, incluindo os novos.
    return {
        title_md: gr.update(value=f"# {lang_map.get('app_title')}"),
        subtitle_md: gr.update(value=lang_map.get('app_subtitle')),
        lang_selector: gr.update(label=lang_map.get('lang_selector_label')),
        step1_accordion: gr.update(label=lang_map.get('step1_accordion')),
        prompt_input: gr.update(label=lang_map.get('prompt_label'), info=lang_map.get('prompt_info')),
        ref_image_input: gr.update(label=lang_map.get('ref_images_label'), info=lang_map.get('ref_images_info')),
        num_keyframes_slider: gr.update(label=lang_map.get('keyframes_label'), info=lang_map.get('keyframes_info')),
        duration_per_fragment_slider: gr.update(label=lang_map.get('duration_label'), info=lang_map.get('duration_info')),
        storyboard_and_keyframes_button: gr.update(value=lang_map.get('storyboard_and_keyframes_button')),
        storyboard_from_photos_button: gr.update(value=lang_map.get('storyboard_from_photos_button')),
        storyboard_output: gr.update(label=lang_map.get('storyboard_output_label')),
        keyframe_gallery: gr.update(label=lang_map.get('keyframes_gallery_label')),
        step3_accordion: gr.update(label=lang_map.get('step3_accordion')),
        step3_description_md: gr.update(value=lang_map.get('step3_description')),
        continuity_director_checkbox: gr.update(label=lang_map.get('continuity_director_label')),
        produce_button: gr.update(value=lang_map.get('produce_button')),
        video_fragments_gallery: gr.update(label=lang_map.get('video_fragments_gallery_label')),
        final_video_output: gr.update(label=lang_map.get('final_movie_with_audio_label')),
        log_accordion: gr.update(label=lang_map.get('log_accordion_label')),
        log_display: gr.update(label=lang_map.get('log_display_label')),
        update_log_button: gr.update(value=lang_map.get('update_log_button')),
        trim_percent_slider: gr.update(label=lang_map.get('trim_percent_label'), info=lang_map.get('trim_percent_info')),
        forca_guia_slider: gr.update(label=lang_map.get('forca_guia_label'), info=lang_map.get('forca_guia_info')),
        convergencia_destino_slider: gr.update(label=lang_map.get('convergencia_final_label'), info=lang_map.get('convergencia_final_info')),
        doc_accordion: gr.update(label=lang_map.get('doc_accordion_label')),
        doc_aduc_title_md: gr.update(value=f"### {lang_map.get('doc_aduc_title')}")
    }

# --- 3. DEFINIÇÃO DA UI ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    default_lang = i18n.get('pt', {})
    
    title_md = gr.Markdown(f"# {default_lang.get('app_title')}")
    subtitle_md = gr.Markdown(default_lang.get('app_subtitle'))
    
    with gr.Row():
        lang_selector = gr.Radio(["pt", "en", "zh"], value="pt", label=default_lang.get('lang_selector_label'))
        resolution_selector = gr.Radio(["480x480"], value="480x480", label="Resolução do Vídeo")

    with gr.Accordion(default_lang.get('step1_accordion'), open=True) as step1_accordion:
        prompt_input = gr.Textbox(label=default_lang.get('prompt_label'), info=default_lang.get('prompt_info'), value="A majestic lion walks across the savanna, sits down, and then roars at the setting sun.")
        ref_image_input = gr.File(label=default_lang.get('ref_images_label'), info=default_lang.get('ref_images_info'), file_count="multiple", file_types=["image"])
        with gr.Row():
            num_keyframes_slider = gr.Slider(minimum=3, maximum=42, value=5, step=1, label=default_lang.get('keyframes_label'), info=default_lang.get('keyframes_info'))
            duration_per_fragment_slider = gr.Slider(label=default_lang.get('duration_label'), info=default_lang.get('duration_info'), minimum=2.0, maximum=10.0, value=4.0, step=0.1)
        with gr.Row():
            storyboard_and_keyframes_button = gr.Button(default_lang.get('storyboard_and_keyframes_button'), variant="primary")
            storyboard_from_photos_button = gr.Button(default_lang.get('storyboard_from_photos_button'))
        gr.Markdown(f"*{default_lang.get('step1_mode_b_info')}*")
        storyboard_output = gr.JSON(label=default_lang.get('storyboard_output_label'))
        keyframe_gallery = gr.Gallery(label=default_lang.get('keyframes_gallery_label'), visible=True, object_fit="contain", height="auto", type="filepath")
        
    with gr.Accordion(default_lang.get('step3_accordion'), open=False, visible=False) as step3_accordion:
        step3_description_md = gr.Markdown(default_lang.get('step3_description'))
        continuity_director_checkbox = gr.Checkbox(label=default_lang.get('continuity_director_label'), value=True)
        
        gr.Markdown("--- \n**Controles de Causalidade (Avançado):**")
        with gr.Row():
            trim_percent_slider = gr.Slider(minimum=10, maximum=90, value=50, step=5, 
                                            label=default_lang.get('trim_percent_label'), 
                                            info=default_lang.get('trim_percent_info'))
        
        gr.Markdown("**Controle de Influência das Âncoras:**")
        with gr.Row():
            forca_guia_slider = gr.Slider(label=default_lang.get('forca_guia_label'), minimum=0.0, maximum=1.0, value=0.5, step=0.05, info=default_lang.get('forca_guia_info'))
            convergencia_destino_slider = gr.Slider(label=default_lang.get('convergencia_final_label'), minimum=0.0, maximum=1.0, value=0.75, step=0.05, info=default_lang.get('convergencia_final_info'))
        
        produce_button = gr.Button(default_lang.get('produce_button'), variant="primary")
    
    video_fragments_gallery = gr.Gallery(label=default_lang.get('video_fragments_gallery_label'), visible=False, object_fit="contain", height="auto", type="filepath")
    final_video_output = gr.Video(label=default_lang.get('final_movie_with_audio_label'), visible=False)

    with gr.Accordion(default_lang.get('log_accordion_label'), open=False) as log_accordion:
        log_display = gr.Textbox(label=default_lang.get('log_display_label'), lines=20, interactive=False, autoscroll=True)
        update_log_button = gr.Button(default_lang.get('update_log_button'))
        
    # --- NOVA SEÇÃO DE DOCUMENTAÇÃO ---
    with gr.Accordion(default_lang.get('doc_accordion_label'), open=False) as doc_accordion:
        doc_aduc_title_md = gr.Markdown(f"### {default_lang.get('doc_aduc_title')}")
        gr.Markdown(
            """
1.  **fragmenta** solicitações acima do limite de contexto de qualquer modelo,
2.  **escala linearmente** (processo sequencial com memória persistida),
3.  **distribui** sub-tarefas a **especialistas** (modelos/ferramentas heterogêneos), e
4.  **realimenta** a próxima etapa com avaliação do que foi feito/esperado (LLM diretor).

Não é um modelo; é uma **camada orquestradora** plugável antes do input de modelos existentes (texto, imagem, áudio, vídeo), usando *tokens universais* e a tecnologia atual.
            """
        )
        gr.Markdown("---")
        gr.Markdown(f"### O Esquema Matemático do Paradigma (Revisado)")
        gr.Markdown(
            """
#### **FÓRMULA 1: O FRAGMENTO INICIAL (Gênesis, `i=1`)**
*Define a criação do primeiro clipe, estabelecendo o estado inicial do movimento a partir de âncoras geométricas estáticas.*

**Planejamento:** `P_1 = Γ( K_1, K_2, P_geral )`
        
**Execução:** `V_1 = Ψ( { (K_1, F_start, ω_1), (K_2, F_end, ω_2) }, P_1 )`

---
#### **FÓRMULA 2: A CADEIA CAUSAL COM DÉJÀ-VU (Momentum, `i > 1`)**
*O coração da arquitetura. Define como a inércia, a trajetória original e o destino futuro são combinados para garantir uma continuidade fluida.*

**Destilação:** 
- `C_(i-1) = Δ_eco( V'_(i-1) )`
- `D_(i-1) = Δ_dejavu( V_(i-1) )`

**Planejamento Adaptativo:** `P_i = Γ( C_(i-1), D_(i-1), K_(i+1), P_geral, H_(i-1), prompt_humano )`

**Execução:** `V_i = Ψ( { (C_(i-1), F_start, 1.0), (D_(i-1), F_mid, ω_dejavu), (K_(i+1), F_end, ω_dest) }, P_i )`
            """
        )
        gr.Markdown("---")
        gr.Markdown(f"### Componentes (Léxico da Arquitetura):")
        gr.Markdown(
            """
- **`V_i`**: Fragmento de Vídeo.
- **`K_i`**: Âncora Geométrica (Keyframe).
- **`C_i`**: **Contexto Causal Cinético** (O "Eco" / Vetor de Inércia).
- **`D_i`**: **Contexto Causal de Trajetória** (O "Déjà-Vu" / Âncora de Caminho).
- **`P_i`**: Prompt Sintetizado (A Intenção da IA).
- **`H_i`**: Histórico Narrativo (A Memória Semântica).
- **`Γ`**: **Oráculo de Síntese Adaptativo** (O Cineasta / LLM Diretor).
- **`Ψ`**: Motor de Geração (A Câmera / Especialista).
- **`Δ`**: Mecanismo de Destilação (O Editor / Orquestrador).
- **`ω`**: **Peso de Convergência** (A Força da Âncora).
            """
        )
        gr.Markdown("---")
        gr.Markdown(f"### Análise da Inovação:")
        gr.Markdown(
            """
A introdução de **pesos de convergência (`ω`)** ajustáveis e a distinção entre **Contexto Cinético (`C`)** e **Contexto de Trajetória (`D`)** são inovações cruciais. Elas transformam os keyframes de "destinos rígidos" em "horizontes de eventos sugeridos". O **Planejamento Adaptativo (`Γ`)** garante que a intenção humana seja reinterpretada à luz do estado atual da narrativa, permitindo que a IA não apenas siga instruções, mas que **conte uma história coerente**. O resultado é um sistema que mantém a continuidade física e semântica, permitindo que cada reinício de fragmento seja sutilmente diferente, mantendo a narrativa viva.
            """
        )
        gr.Markdown("---")
        gr.Markdown(f"### Contato / Contact")
        gr.Markdown(
            """
- **Author / Autor:** Carlos Rodrigues dos Santos
- **Email:** carlex22@gmail.com
- **GitHub:** [https://github.com/carlex22/Aduc-sdr](https://github.com/carlex22/Aduc-sdr)
          """
        )

    # --- 4. CONEXÕES DA UI ---
    all_ui_components = list(update_ui_language('pt').keys())
    lang_selector.change(fn=update_ui_language, inputs=lang_selector, outputs=all_ui_components)
    
    storyboard_and_keyframes_button.click(
        fn=run_mode_a_wrapper, 
        inputs=[prompt_input, num_keyframes_slider, ref_image_input, resolution_selector, duration_per_fragment_slider], 
        outputs=[storyboard_output, keyframe_gallery, step3_accordion]
    )
    
    storyboard_from_photos_button.click(
        fn=run_mode_b_wrapper,
        inputs=[prompt_input, num_keyframes_slider, ref_image_input],
        outputs=[storyboard_output, keyframe_gallery, step3_accordion]
    )
    
    produce_button.click(
        fn=run_video_production_wrapper,
        inputs=[
            keyframe_gallery, prompt_input, duration_per_fragment_slider, 
            trim_percent_slider,
            forca_guia_slider,
            convergencia_destino_slider,
            resolution_selector, continuity_director_checkbox
        ],
        outputs=[video_fragments_gallery, final_video_output]
    )

    update_log_button.click(
        fn=get_log_content,
        inputs=[],
        outputs=[log_display]
    )

# --- 5. INICIALIZAÇÃO DA APLICAÇÃO ---
if __name__ == "__main__":
    if os.path.exists(WORKSPACE_DIR): 
        logger.info(f"Limpando o workspace anterior em: {WORKSPACE_DIR}")
        shutil.rmtree(WORKSPACE_DIR)
    os.makedirs(WORKSPACE_DIR)
    logger.info(f"Aplicação iniciada. Lançando interface Gradio...")
    demo.queue().launch()