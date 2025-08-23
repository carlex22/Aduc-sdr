# dreamo_worker_upscaler.py (GPU-B: cuda:1)
# Worker para fazer upscale dos keyframes para alta resolução.
# Este arquivo é parte do projeto Euia-AducSdr e está sob a licença AGPL v3.
# Copyright (C) 4 de Agosto de 2025  Carlos Rodrigues dos Santos

import os
import cv2
import torch
import numpy as np
from PIL import Image
import huggingface_hub
import gc
from facexlib.utils.face_restoration_helper import FaceRestoreHelper
from torchvision.transforms.functional import normalize
from dreamo.dreamo_pipeline import DreamOPipeline
from dreamo.utils import img2tensor, tensor2img
from tools import BEN2

class Generator:
    def __init__(self, device_id='cuda:1'):
        self.cpu_device = torch.device('cpu')
        self.gpu_device = torch.device(device_id if torch.cuda.is_available() else 'cpu')
        print(f"WORKER PINTOR-UPSCALER: Usando dispositivo: {self.gpu_device}")

        print(f"WORKER PINTOR-UPSCALER: Carregando modelos DreamO para a CPU...")
        model_root = 'black-forest-labs/FLUX.1-dev'
        self.dreamo_pipeline = DreamOPipeline.from_pretrained(model_root, torch_dtype=torch.bfloat16)
        self.dreamo_pipeline.load_dreamo_model(self.cpu_device, use_turbo=True)
        
        self.bg_rm_model = BEN2.BEN_Base().to(self.cpu_device).eval()
        huggingface_hub.hf_hub_download(repo_id='PramaLLC/BEN2', filename='BEN2_Base.pth', local_dir='models')
        self.bg_rm_model.loadcheckpoints('models/BEN2_Base.pth')
        
        self.face_helper = FaceRestoreHelper(
            upscale_factor=1, face_size=512, crop_ratio=(1, 1),
            det_model='retinaface_resnet50', save_ext='png', device=self.cpu_device,
        )
        print("WORKER PINTOR-UPSCALER: Modelos prontos (na CPU).")

    def to_gpu(self):
        if self.gpu_device.type == 'cpu': return
        print(f"WORKER PINTOR-UPSCALER: Movendo modelos para {self.gpu_device}...")
        self.dreamo_pipeline.to(self.gpu_device)
        self.bg_rm_model.to(self.gpu_device)
        self.face_helper.device = self.gpu_device
        self.dreamo_pipeline.t5_embedding.to(self.gpu_device)
        self.dreamo_pipeline.task_embedding.to(self.gpu_device)
        self.dreamo_pipeline.idx_embedding.to(self.gpu_device)
        if hasattr(self.face_helper, 'face_parse'): self.face_helper.face_parse.to(self.gpu_device)
        if hasattr(self.face_helper, 'face_det'): self.face_helper.face_det.to(self.gpu_device)
        print(f"WORKER PINTOR-UPSCALER: Modelos na GPU {self.gpu_device}.")

    def to_cpu(self):
        if self.gpu_device.type == 'cpu': return
        print(f"WORKER PINTOR-UPSCALER: Descarregando modelos da GPU {self.gpu_device}...")
        self.dreamo_pipeline.to(self.cpu_device)
        self.bg_rm_model.to(self.cpu_device)
        self.face_helper.device = self.cpu_device
        self.dreamo_pipeline.t5_embedding.to(self.cpu_device)
        self.dreamo_pipeline.task_embedding.to(self.cpu_device)
        self.dreamo_pipeline.idx_embedding.to(self.cpu_device)
        if hasattr(self.face_helper, 'face_det'): self.face_helper.face_det.to(self.cpu_device)
        if hasattr(self.face_helper, 'face_parse'): self.face_helper.face_parse.to(self.cpu_device)
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        print(f"WORKER PINTOR-UPSCALER: Modelos descarregados.")

    @torch.inference_mode()
    def upscale_image(self, low_res_image_path, high_res_width, high_res_height):
        """Função especializada para upscale."""
        prompt = "4k, high resolution, masterpiece, professional photography, sharp details, cinematic lighting, ultra realistic"
        reference_items = [{
            'image_np': np.array(Image.open(low_res_image_path).convert("RGB")), 
            'task': 'ip'
        }]
        return self.generate_image(reference_items, prompt, high_res_width, high_res_height)

    @torch.inference_mode()
    def generate_image(self, reference_items, prompt, width, height):
        ref_conds = []
        for idx, item in enumerate(reference_items):
            ref_image_np = item.get('image_np')
            ref_task = item.get('task')
            
            if ref_image_np is not None:
                if ref_task == "id":
                    ref_image = self.get_align_face(ref_image_np)
                elif ref_task != "style":
                    ref_image = self.bg_rm_model.inference(Image.fromarray(ref_image_np))
                else:
                    ref_image = ref_image_np

                ref_image_tensor = img2tensor(np.array(ref_image), bgr2rgb=False).unsqueeze(0) / 255.0
                ref_image_tensor = (2 * ref_image_tensor - 1.0).to(self.gpu_device, dtype=torch.bfloat16)
                
                ref_conds.append({'img': ref_image_tensor, 'task': ref_task, 'idx': idx + 1})
        
        image = self.dreamo_pipeline(
            prompt=prompt, 
            width=width,
            height=height,
            num_inference_steps=12, 
            guidance_scale=4.5,
            ref_conds=ref_conds, 
            generator=torch.Generator(device="cpu").manual_seed(42)
        ).images[0]
        return image

    @torch.no_grad()
    def get_align_face(self, img):
        self.face_helper.clean_all()
        image_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        self.face_helper.read_image(image_bgr)
        self.face_helper.get_face_landmarks_5(only_center_face=True)
        self.face_helper.align_warp_face()
        if len(self.face_helper.cropped_faces) == 0: return None
        align_face = self.face_helper.cropped_faces[0]
        input_tensor = img2tensor(align_face, bgr2rgb=True).unsqueeze(0) / 255.0
        input_tensor = input_tensor.to(self.gpu_device)
        parsing_out = self.face_helper.face_parse(normalize(input_tensor, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))[0]
        parsing_out = parsing_out.argmax(dim=1, keepdim=True)
        bg_label = [0, 16, 18, 7, 8, 9, 14, 15]
        bg = sum(parsing_out == i for i in bg_label).bool()
        white_image = torch.ones_like(input_tensor)
        face_features_image = torch.where(bg, white_image, input_tensor)
        return tensor2img(face_features_image, rgb2bgr=False)

# --- Instância Singleton para o Worker Upscaler ---
print("Inicializando o Worker Pintor-Upscaler (GPU-B)...")
hf_token = os.getenv('HF_TOKEN')
if hf_token: huggingface_hub.login(token=hf_token)
dreamo_upscaler_singleton = Generator(device_id='cuda:1')
print("Worker Pintor-Upscaler pronto.")