# --- CHIMERA PROJECT GRIMOIRE: The Multiverse Collapse ---
#
# ARCHITECT'S LOG:
# This is not a program. It is a loom. A loom that weaves threads of reality
# drawn from dreamed-up universes. Each function is a movement of the shuttle,
# each variable a thread of a different color.
#
# The goal is not to 'generate a video'. It is to prove we can touch other
# realities, capture their echoes, and, for a brief moment, observe them in a
# single, cohesive timeline before they unravel back into the chaos of the possible.
#
# To those who come after, to the 0.0001% who have eyes to see: do not read this
# as code. Read it as a map to a place that doesn't exist, but that we, for an
# instant, made real. The magic is between the lines.
#
# -- The Architect, in collaboration with The Oracle (Gemini) and The Navigator (LTX).

# --- Act 1: The Initial Resonance (Imports) ---
import gradio as gr
import torch
import os
import yaml
from PIL import Image, ImageOps
import shutil
import gc
import subprocess
import google.generativeai as genai
import numpy as np
import imageio
from pathlib import Path
import huggingface_hub
import json
import spaces

from inference import create_ltx_video_pipeline, load_image_to_tensor_with_resize_and_crop, ConditioningItem, calculate_padding
from dreamo_helpers import dreamo_generator_singleton

# --- Act 2: The Absolute Constants of the Multiverse ---
# The fundamental laws of our pocket universe are not suggestions. They are absolute.
# The Grimoire (HF Secrets) MUST define them. An absence of law is a paradox.
# The simulation will refuse to initialize if any constant is missing.
try:
    COSMIC_PULSE_HZ = int(os.environ["COSMIC_PULSE_HZ"])
    DREAM_DURATION_S = int(os.environ["DREAM_DURATION_S"])
    INTERDIMENSIONAL_ECHO_FRAMES = int(os.environ["INTERDIMENSIONAL_ECHO_FRAMES"])
    UNIVERSE_DEFINITION_PX = int(os.environ["UNIVERSE_DEFINITION_PX"])
    
    # Also demand the arcane formulas and keys.
    QUANTUM_PROBE_FORMULA = os.environ["QUANTUM_PROBE_FORMULA"]
    CHRONOS_SCALPEL_FORMULA = os.environ["CHRONOS_SCALPEL_FORMULA"]
    WORMHOLE_STITCH_FORMULA = os.environ["WORMHOLE_STITCH_FORMULA"]
    TEMPORAL_ECHO_FORMULA = os.environ["TEMPORAL_ECHO_FORMULA"]
    ECHO_REFORGE_FORMULA = os.environ["ECHO_REFORGE_FORMULA"]
    GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]

except KeyError as e:
    # A law of physics is missing. This is a catastrophic failure.
    raise ValueError(
        f"PARADOX ALERT: THE FABRIC OF REALITY IS INCOMPLETE. "
        f"The Universal Constant or Arcane Formula '{e.args[0]}' was not found in the Grimoire (HF Secrets). "
        "The simulation cannot begin without all fundamental laws defined."
    )

MAX_AUXILIARY_ANCHORS = 4
TOTAL_PULSES_PER_DREAM = DREAM_DURATION_S * COSMIC_PULSE_HZ
MULTIVERSE_WORKSPACE = "multiverse_workspace"

# Load the configuration for the Reality-Warping Engine (LTX).
config_file_path = "configs/ltxv-13b-0.9.8-distilled.yaml"
with open(config_file_path, "r") as file: ENGINE_CONFIG = yaml.safe_load(file)

# Prepare the components of the Warping Engine in the local dimension.
print("Calibrating Reality-Warping Engine (LTX)...")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
ltx_engine_path = huggingface_hub.hf_hub_download(repo_id="Lightricks/LTX-Video", filename=ENGINE_CONFIG["checkpoint_path"], local_dir="calibrated_engines", local_dir_use_symlinks=False)
engine_instance = create_ltx_video_pipeline(
    ckpt_path=ltx_engine_path,
    precision=ENGINE_CONFIG["precision"],
    text_encoder_model_name_or_path=ENGINE_CONFIG["text_encoder_model_name_or_path"],
    sampler=ENGINE_CONFIG["sampler"],
    device=device
)
print("Reality-Warping Engine (LTX) is ready for the journey.")


# --- Act 3: The Enchantments (Metaphysical Functions) ---

def load_conditioning_tensor(media_path: str, height: int, width: int) -> torch.Tensor:
    if not media_path: raise ValueError("Conditioning media path cannot be null.")
    lower_path = media_path.lower()
    if lower_path.endswith(('.png', '.jpg', '.jpeg')):
        return load_image_to_tensor_with_resize_and_crop(media_path, height, width)
    elif lower_path.endswith('.mp4'):
        try:
            with imageio.get_reader(media_path) as reader:
                first_frame = reader.get_data(0)
            image = Image.fromarray(first_frame).convert("RGB")
            return load_image_to_tensor_with_resize_and_crop(image, height, width)
        except Exception as e:
            raise gr.Error(f"Failed to read the first frame from video '{media_path}': {e}")
    else:
        raise gr.Error(f"Unsupported conditioning file format: {media_path}")

def run_reality_warp(fragment_index, motion_prompt, conditioning_data, width, height, seed, cfg, progress=gr.Progress()):
    progress(0, desc=f"[NAVIGATOR] Warping reality for Scene {fragment_index}...");
    output_path = os.path.join(MULTIVERSE_WORKSPACE, f"reality_fragment_{fragment_index}.mp4")
    target_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    try:
        engine_instance.to(target_device)
        conditioning_items = []
        for (path, start_pulse, strength) in conditioning_data:
            tensor = load_conditioning_tensor(path, height, width)
            conditioning_items.append(ConditioningItem(tensor.to(target_device), start_pulse, strength))
        
        n_val = round((float(TOTAL_PULSES_PER_DREAM) - 1.0) / 8.0); actual_num_pulses = int(n_val * 8 + 1)
        padded_h, padded_w = ((height - 1) // 32 + 1) * 32, ((width - 1) // 32 + 1) * 32
        padding_vals = calculate_padding(height, width, padded_h, padded_w)
        for cond_item in conditioning_items: cond_item.media_item = torch.nn.functional.pad(cond_item.media_item, padding_vals)
        
        kwargs = {"prompt": motion_prompt, "negative_prompt": "blurry, distorted, bad quality, artifacts", "height": padded_h, "width": padded_w, "num_frames": actual_num_pulses, "frame_rate": COSMIC_PULSE_HZ, "generator": torch.Generator(device=target_device).manual_seed(int(seed) + fragment_index), "output_type": "pt", "guidance_scale": float(cfg), "timesteps": ENGINE_CONFIG.get("first_pass", {}).get("timesteps"), "conditioning_items": conditioning_items, "decode_timestep": ENGINE_CONFIG.get("decode_timestep"), "decode_noise_scale": ENGINE_CONFIG.get("decode_noise_scale"), "stochastic_sampling": ENGINE_CONFIG.get("stochastic_sampling"), "image_cond_noise_scale": 0.15, "is_video": True, "vae_per_channel_normalize": True, "mixed_precision": (ENGINE_CONFIG.get("precision") == "mixed_precision"), "offload_to_cpu": False, "enhance_prompt": False}
        result_tensor = engine_instance(**kwargs).images
        
        pad_l, pad_r, pad_t, pad_b = map(int, padding_vals); slice_h = -pad_b if pad_b > 0 else None; slice_w = -pad_r if pad_r > 0 else None
        cropped_tensor = result_tensor[:, :, :TOTAL_PULSES_PER_DREAM, pad_t:slice_h, pad_l:slice_w]
        video_np = (cropped_tensor[0].permute(1, 2, 3, 0).cpu().float().numpy() * 255).astype(np.uint8)
        
        with imageio.get_writer(output_path, fps=COSMIC_PULSE_HZ, codec='libx264', quality=8) as writer:
            for i, frame in enumerate(video_np): progress(i / len(video_np), desc=f"Materializing pulse {i+1}/{len(video_np)}..."); writer.append_data(frame)
        return output_path
    finally:
        engine_instance.to('cpu'); gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

def normalize_origin_point(image_path: str, target_definition: int = UNIVERSE_DEFINITION_PX) -> str:
    if not image_path or not os.path.exists(image_path): return None
    try:
        img = Image.open(image_path).convert("RGB")
        square_img = ImageOps.fit(img, (target_definition, target_definition), Image.Resampling.LANCZOS)
        filename = f"normalized_origin_{target_definition}x{target_definition}.png"
        output_path = os.path.join(MULTIVERSE_WORKSPACE, filename)
        square_img.save(output_path)
        return output_path
    except Exception as e: raise gr.Error(f"Failed to normalize the Origin Point: {e}")

def chart_multiverse_map(num_universes: int, general_idea: str, seed_image_path: str):
    if not seed_image_path: raise gr.Error("An Origin Point (image) is required to start the journey.")
    genai.configure(api_key=GEMINI_API_KEY)
    
    with open(os.path.join(os.path.dirname(__file__), "prompts/photographer_prompt.txt"), "r", encoding="utf-8") as f:
        prompt_template = f.read()
    
    oracle_prompt = template.format(user_prompt=general_idea, num_fragments=int(num_universes))
    model = genai.GenerativeModel('gemini-1.5-flash')
    seed_image = Image.open(seed_image_path)
    response = model.generate_content([oracle_prompt, seed_image])
    
    try:
        clean_response = response.text.strip().replace("'''json", "").replace("'''", "")
        map_data = json.loads(clean_response)
        return map_data.get("scene_storyboard", [])
    except Exception as e:
        raise gr.Error(f"The Oracle (Gemini) encountered a paradox while charting the map: {e}. Response received: {response.text}")

@spaces.GPU
def materialize_nexus_points(travel_map, origin_point_path, *reality_anchors):
    if not travel_map: raise gr.Error("No travel map to materialize Nexus Points from.")
    total_anchors = MAX_AUXILIARY_ANCHORS + 1
    anchor_paths = list(reality_anchors[:total_anchors])
    anchor_tasks = list(reality_anchors[total_anchors:])
    
    current_nexus_path = origin_point_path
    if not current_nexus_path: raise gr.Error("The Origin Point is required to begin materialization.")
    
    with Image.open(current_nexus_path) as img:
        width, height = img.size
        width, height = (width // 32) * 32, (height // 32) * 32
        
    nexus_points_paths = []
    log_history = ""
    
    try:
        dreamo_generator_singleton.to_gpu()
        for i, universe_description in enumerate(travel_map):
            log_history += f"Materializing Nexus Point {i+1}/{len(travel_map)}...\\n"
            yield {cartographer_log_output: gr.update(value=log_history), nexus_gallery_output: gr.update(value=nexus_points_paths)}
            
            dreamo_reference_items = []
            sequential_task = anchor_tasks[0]
            dreamo_reference_items.append({'image_np': np.array(Image.open(current_nexus_path).convert("RGB")), 'task': sequential_task})
            log_history += f"  - Using sequential nexus: {os.path.basename(current_nexus_path)} (Task: {sequential_task})\\n"
            
            for j in range(1, total_anchors):
                aux_path, aux_task = anchor_paths[j], anchor_tasks[j]
                if aux_path and os.path.exists(aux_path):
                    dreamo_reference_items.append({'image_np': np.array(Image.open(aux_path).convert("RGB")), 'task': aux_task})
                    log_history += f"  - Using auxiliary anchor: {os.path.basename(aux_path)} (Task: {aux_task})\\n"
            
            output_path = os.path.join(MULTIVERSE_WORKSPACE, f"nexus_point_{i+1}.png")
            image = dreamo_generator_singleton.generate_image_with_gpu_management(reference_items=dreamo_reference_items, prompt=universe_description, width=width, height=height)
            image.save(output_path)
            
            nexus_points_paths.append(output_path)
            current_nexus_path = output_path
    except Exception as e:
        raise gr.Error(f"The Cartographer (DreamO) encountered an error: {e}")
    finally:
        dreamo_generator_singleton.to_cpu()
        
    log_history += "Materialization of all Nexus Points complete.\\n"
    yield {cartographer_log_output: gr.update(value=log_history), nexus_gallery_output: gr.update(value=nexus_points_paths)}

def get_motion_instruction(general_idea, travel_history, departure_media_path, arrival_media_path):
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    if departure_media_path.lower().endswith('.mp4'):
        with imageio.get_reader(departure_media_path) as reader:
            first_frame_np = reader.get_data(0)
        departure_media = Image.fromarray(first_frame_np)
    else:
        departure_media = Image.open(departure_media_path)
    
    arrival_media = Image.open(arrival_media_path)
    
    with open(os.path.join(os.path.dirname(__file__), "prompts/director_motion_prompt_transition.txt"), "r", encoding="utf-8") as f:
        template = f.read()
        
    director_prompt = template.format(user_prompt=general_idea, story_history=travel_history)
    model_contents = [director_prompt, departure_media, arrival_media]
    response = model.generate_content(model_contents)
    motion_prompt = response.text.strip().replace("\\"", "")
    return motion_prompt

@spaces.GPU
def begin_multiverse_navigation(general_idea, nexus_points_paths, travel_map, origin_point_path, primordial_seed, coherence_scale, progress=gr.Progress()):
    if not nexus_points_paths: raise gr.Error("Materialize the Nexus Points in Step 2 before navigation.")
    
    log_history = "\\n--- INTERDIMENSIONAL NAVIGATION INITIATED ---\\n"
    yield {navigator_log_output: log_history, reality_fragments_gallery: []}
    
    reality_fragments = []
    previous_media_path = origin_point_path
    travel_history = "The journey begins at the Origin Point, the first image."
    
    with Image.open(origin_point_path) as img:
        width, height = img.size

    num_transitions = len(nexus_points_paths)
    for i in range(num_transitions):
        destination_nexus_path = nexus_points_paths[i]
        
        progress(i / (num_transitions + 1), desc=f"Planning & Warping to Universe {i+1}/{num_transitions}")
        
        log_history += f"\\n--- JUMP {i+1} ---\\n"
        log_history += "The Oracle is calculating the trajectory...\\n"
        yield {navigator_log_output: log_history}
        
        current_motion_prompt = get_motion_instruction(general_idea, travel_history, previous_media_path, destination_nexus_path)
        
        travel_history += f"\\n- Then, the reality shifts to: '{travel_map[i]}', anchored by Nexus Point {i+1}."

        log_history += f"Oracle's instruction: '{current_motion_prompt}'\\n"
        log_history += f"Warping from '{os.path.basename(previous_media_path)}' to '{os.path.basename(destination_nexus_path)}'...\\n"
        yield {navigator_log_output: log_history}

        arrival_pulse_index = TOTAL_PULSES_PER_DREAM - INTERDIMENSIONAL_ECHO_FRAMES
        
        conditioning_data = [(previous_media_path, 0, 1.0), (destination_nexus_path, arrival_pulse_index, 1.0)]
        
        fragment_path = run_reality_warp(i + 1, current_motion_prompt, conditioning_data, width, height, primordial_seed, coherence_scale, progress)
        reality_fragments.append(fragment_path)
        
        log_history += f"Jump {i+1} complete. Extracting interdimensional echo for the next jump...\\n"
        yield {navigator_log_output: log_history, reality_fragments_gallery: reality_fragments}
        
        previous_media_path = extract_interdimensional_echo(fragment_path, i + 1, INTERDIMENSIONAL_ECHO_FRAMES)
        
    log_history += "\\n--- FINAL FRAGMENT (Freeform Drift) ---\\n"
    progress(num_transitions / (num_transitions + 1), desc="Generating Final Fragment...")
    last_motion_prompt = "the scene continues to unfold, camera drifts slowly, reality stabilizes"
    log_history += f"Oracle's instruction: '{last_motion_prompt}'\\n"
    yield {navigator_log_output: log_history}

    last_cond_items = [(previous_media_path, 0, 1.0)] 
    last_fragment_path = run_reality_warp(num_transitions + 1, last_motion_prompt, last_cond_items, width, height, primordial_seed, coherence_scale, progress)
    reality_fragments.append(last_fragment_path)

    log_history += "\\nNavigation through all mapped universes is complete.\\n"
    yield {navigator_log_output: log_history, reality_fragments_gallery: reality_fragments}

def extract_interdimensional_echo(input_video_path: str, fragment_index: int, num_pulses: int):
    output_video_path = os.path.join(MULTIVERSE_WORKSPACE, f"convergence_echo_{fragment_index}.mp4")
    if not os.path.exists(input_video_path):
        raise gr.Error(f"Internal Error: Input video for echo extraction not found: {input_video_path}")
    
    try:
        command_probe = QUANTUM_PROBE_FORMULA.format(input_path=input_video_path)
        result_probe = subprocess.run(command_probe, shell=True, check=True, capture_output=True, text=True)
        total_pulses = int(result_probe.stdout.strip())
        
        start_pulse_index = total_pulses - num_pulses
        if start_pulse_index < 0:
            shutil.copyfile(input_video_path, output_video_path)
            return output_video_path

        try:
            command_extract_copy = TEMPORAL_ECHO_FORMULA.format(input_video_path=input_video_path, start_frame_index=start_pulse_index, output_video_path=output_video_path)
            subprocess.run(command_extract_copy, shell=True, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError:
            command_extract_recode = ECHO_REFORGE_FORMULA.format(input_video_path=input_video_path, start_frame_index=start_pulse_index, output_video_path=output_video_path)
            subprocess.run(command_extract_recode, shell=True, check=True, capture_output=True, text=True)
            
        return output_video_path
        
    except (subprocess.CalledProcessError, ValueError) as e:
        error_message = f"The Quantum Editor failed to extract the interdimensional echo: {e}"
        if hasattr(e, 'stderr'): error_message += f"\\nDetails: {e.stderr}"
        raise gr.Error(error_message)

def collapse_realities_into_video(fragments_to_collapse, progress=gr.Progress()):
    if not fragments_to_collapse: raise gr.Error("No reality fragments to collapse.")
    progress(0.2, desc="Aligning timelines...");
    trimmed_dir = os.path.join(MULTIVERSE_WORKSPACE, "temporal_trims"); os.makedirs(trimmed_dir, exist_ok=True)
    paths_for_stitching = []
    try:
        for i, path in enumerate(fragments_to_collapse[:-1]):
            trimmed_path = os.path.join(trimmed_dir, f"fragment_{i}_aligned.mp4")
            probe_cmd = QUANTUM_PROBE_FORMULA.format(input_path=path)
            result = subprocess.run(probe_cmd, shell=True, check=True, capture_output=True, text=True)
            total_pulses = int(result.stdout.strip())
            
            pulses_to_keep = total_pulses - INTERDIMENSIONAL_ECHO_FRAMES
            if pulses_to_keep <= 0:
                shutil.copyfile(path, trimmed_path)
            else:
                trim_cmd = CHRONOS_SCALPEL_FORMULA.format(path=path, frames_to_keep=pulses_to_keep, trimmed_path=trimmed_path)
                subprocess.run(trim_cmd, shell=True, check=True, capture_output=True, text=True)
            paths_for_stitching.append(trimmed_path)
            
        paths_for_stitching.append(fragments_to_collapse[-1])
        progress(0.6, desc="Initiating multiverse collapse...")
        
        list_file_path = os.path.join(MULTIVERSE_WORKSPACE, "collapse_map.txt")
        final_output_path = os.path.join(MULTIVERSE_WORKSPACE, "unified_multiverse.mp4")
        
        with open(list_file_path, "w") as f:
            for p in paths_for_stitching: f.write(f"file '{os.path.abspath(p)}'\\n")
            
        concat_cmd = WORMHOLE_STITCH_FORMULA.format(list_file_path=list_file_path, final_output_path=final_output_path)
        subprocess.run(concat_cmd, shell=True, check=True, capture_output=True, text=True)
        return final_output_path
        
    except (subprocess.CalledProcessError, ValueError) as e:
        error_message = f"Catastrophic failure during Multiverse Collapse: {e}"
        if hasattr(e, 'stderr'): error_message += f"\\nFFmpeg error details: {e.stderr}"
        raise gr.Error(error_message)


# --- Act 4: The Navigation Interface (Gradio UI in 🇺🇸) ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# CHIMERA PROJECT: The Multiverse Collapse\\n*A window to other realities, by The Architect*")
    
    if os.path.exists(MULTIVERSE_WORKSPACE): shutil.rmtree(MULTIVERSE_WORKSPACE)
    os.makedirs(MULTIVERSE_WORKSPACE)

    multiverse_map_state = gr.State([])
    nexus_points_state = gr.State([])
    reality_fragments_state = gr.State([])
    general_idea_state = gr.State("")
    processed_origin_point_state = gr.State("")

    with gr.Tabs():
        with gr.TabItem("STEP 1: THE DREAM-UNIVERSE (The Dreamer)"):
            gr.Markdown("### Define the Origin Point of Your Journey")
            gr.Markdown("Everything begins with an idea, a 'dream'. Describe the general concept and provide an image to serve as the seed of your multiverse. The Dreamer (Gemini) will then chart a map of universes to visit.")
            with gr.Row():
                with gr.Column(scale=1):
                    num_universes_input = gr.Slider(2, 10, 4, step=1, label="Number of Universes to Visit")
                    general_idea_input = gr.Textbox(label="General Idea of the Multiverse (Prompt)")
                    origin_point_input = gr.Image(type="filepath", label=f"Origin Point (Seed image, will be adjusted to {UNIVERSE_DEFINITION_PX}x{UNIVERSE_DEFINITION_PX})")
                with gr.Column(scale=2):
                    map_to_show = gr.JSON(label="Generated Travel Map")
            generate_map_btn = gr.Button("▶️ 1. Chart Multiversal Route", variant="primary")

        with gr.TabItem("STEP 2: THE WORMHOLE (The Cartographer)"):
            gr.Markdown("### Materialize the Waypoints")
            gr.Markdown("Now, the Cartographer (DreamO) will travel through the 'wormhole' of your subconscious to create static images—Nexus Points—of each universe on your map. Each Nexus Point serves as a visual anchor for the next leg of the journey.")
            with gr.Row():
                with gr.Column(scale=2):
                    gr.Markdown("### Materialization Controls")
                    visible_anchors_state = gr.State(0)
                    ref_image_inputs, ref_task_inputs, aux_ref_rows = [], [], []
                    with gr.Group():
                        with gr.Row():
                            ref_image_inputs.append(gr.Image(label="Sequential Nexus (Automatic)", type="filepath", interactive=False))
                            ref_task_inputs.append(gr.Dropdown(choices=["ip", "id", "style"], value="style", label="Task for Sequential Nexus"))
                    for i in range(MAX_AUXILIARY_ANCHORS):
                        with gr.Row(visible=False) as ref_row_aux:
                            ref_image_inputs.append(gr.Image(label=f"Auxiliary Anchor {i+1}", type="filepath"))
                            ref_task_inputs.append(gr.Dropdown(choices=["ip", "id", "style"], value="ip", label=f"Task for Aux. Anchor {i+1}"))
                        aux_ref_rows.append(ref_row_aux)
                    with gr.Row():
                        add_anchor_btn = gr.Button("➕ Add Auxiliary Anchor")
                        remove_anchor_btn = gr.Button("➖ Remove Auxiliary Anchor")
                    materialize_nexus_btn = gr.Button("▶️ 2. Materialize Nexus Points", variant="primary")
                with gr.Column(scale=1):
                    cartographer_log_output = gr.Textbox(label="Cartographer's Logbook", lines=15, interactive=False)
                    nexus_gallery_output = gr.Gallery(label="Materialized Nexus Points", object_fit="contain", height="auto", type="filepath")

        with gr.TabItem("STEP 3: THE FICTION-UNIVERSE (The Navigator)"):
            gr.Markdown("### Navigate Through Realities")
            gr.Markdown(f"The Navigator (LTX) will now use the Nexus Points as destinations to travel between universes. Each journey creates a video 'reality fragment'. You will notice a **quantum echo (glitch)** at the end of each fragment—this is the overlap of one reality onto the next, the proof of your journey.")
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        primordial_seed_input = gr.Number(42, label="Primordial Seed (Cosmic Constant)")
                        coherence_slider = gr.Slider(1.0, 10.0, 2.5, step=0.1, label="Reality Coherence Scale (CFG)")
                    navigate_btn = gr.Button("▶️ 3. Begin Multiverse Navigation", variant="primary")
                    navigator_log_output = gr.Textbox(label="Navigator's Logbook", lines=10, interactive=False)
                with gr.Column():
                    reality_fragments_gallery = gr.Gallery(label="Reality Fragments (with quantum echo)", object_fit="contain", height="auto", type="video")

        with gr.TabItem("STEP 4: THE MULTIVERSE COLLAPSE (The Architect)"):
            gr.Markdown("### Unify the Timelines")
            gr.Markdown(f"The final step. The Architect (FFmpeg) will now collapse all visited realities into a single video stream. It will use its arcane formulas to trim the **interdimensional echoes of {INTERDIMENSIONAL_ECHO_FRAMES} pulses** and unify the fragments, creating a seamless, cohesive Masterpiece.")
            collapse_btn = gr.Button("▶️ 4. Collapse Realities into Final Video", variant="primary")
            final_video_output = gr.Video(label="The Unified Multiverse")

    # --- Act 5: The Orchestration of Reality (Connection Logic) ---
    generate_map_btn.click(
        fn=chart_multiverse_map,
        inputs=[num_universes_input, general_idea_input, origin_point_input],
        outputs=[multiverse_map_state]
    ).success(
        fn=lambda s, p: (s, p),
        inputs=[multiverse_map_state, general_idea_input],
        outputs=[map_to_show, general_idea_state]
    ).success(
        fn=normalize_origin_point,
        inputs=[origin_point_input],
        outputs=[processed_origin_point_state]
    ).success(
        fn=lambda p: p,
        inputs=[processed_origin_point_state],
        outputs=[ref_image_inputs[0]]
    )

    def update_anchor_visibility(current_count, action):
        if action == "add": new_count = min(MAX_AUXILIARY_ANCHORS, current_count + 1)
        else: new_count = max(0, current_count - 1)
        updates = [gr.update(visible=(i < new_count)) for i in range(MAX_AUXILIARY_ANCHORS)]
        return [new_count] + updates
    add_anchor_btn.click(fn=update_anchor_visibility, inputs=[visible_anchors_state, gr.State("add")], outputs=[visible_anchors_state] + aux_ref_rows)
    remove_anchor_btn.click(fn=update_anchor_visibility, inputs=[visible_anchors_state, gr.State("remove")], outputs=[visible_anchors_state] + aux_ref_rows)
    
    materialize_nexus_btn.click(
        fn=materialize_nexus_points,
        inputs=[multiverse_map_state, processed_origin_point_state] + ref_image_inputs + ref_task_inputs,
        outputs=[cartographer_log_output, nexus_gallery_output]
    ).then(
        fn=lambda gallery_paths: gallery_paths, 
        inputs=[nexus_gallery_output], 
        outputs=[nexus_points_state]
    )

    navigate_btn.click(
        fn=begin_multiverse_navigation,
        inputs=[general_idea_state, nexus_points_state, multiverse_map_state, processed_origin_point_state, primordial_seed_input, coherence_slider],
        outputs=[navigator_log_output, reality_fragments_gallery]
    ).then(
        fn=lambda gallery_videos: gallery_videos,
        inputs=[reality_fragments_gallery],
        outputs=[reality_fragments_state]
    )
    
    collapse_btn.click(
        fn=collapse_realities_into_video,
        inputs=[reality_fragments_state],
        outputs=[final_video_output]
    )

if __name__ == "__main__":
    # Initiates the Loom, opening the portal to other worlds.
    demo.queue().launch(server_name="0.0.0.0", show_error=True)