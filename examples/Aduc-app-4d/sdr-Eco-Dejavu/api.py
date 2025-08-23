# api.py
# Servidor FastAPI para o framework Deformes4D.
# Copyright (C) 2025 Carlos Rodrigues dos Santos

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
import logging
import os
import random
import time

from aduc_orchestrator import AducOrchestrator
from config import config # Assumindo que a config foi movida para um arquivo config.py

# --- Configuração ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="Deformes4D API", description="API para Cirurgia de Realidade Latente com ADUC.")
aduc = AducOrchestrator(workspace_dir=config['application']['workspace_dir'])

# --- Modelos de Dados (Validação com Pydantic) ---

class GenerateRequest(BaseModel):
    prompt: str = Field(..., example="a majestic lion walking on the savanna")
    num_frames: int = Field(49, ge=25, le=241)
    width: int = Field(512, ge=256, le=1024)
    height: int = Field(512, ge=256, le=1024)
    cfg: float = Field(1.0, ge=1.0, le=5.0)
    steps: int = Field(4, ge=4, le=30)

class RegenerateRequest(BaseModel):
    quantum_twin_path: str = Field(..., description="Caminho para o arquivo de latentes (.pt) no servidor.")
    x_cut: int = Field(..., description="Frame exato do ponto de corte.")
    n_cut: int = Field(..., description="Número de frames a serem removidos.")
    n_eco: int = Field(..., description="Número de frames para o eco de memória.")

# --- Endpoints da API ---

@app.post("/generate", summary="Gera um vídeo base e seu gêmeo latente")
def generate_base_video(request: GenerateRequest):
    """
    Cria um vídeo inicial a partir de um prompt e retorna os caminhos
    para o vídeo gerado e seu arquivo de latentes correspondente.
    """
    logger.info(f"API: Recebida requisição de geração para o prompt: '{request.prompt}'")
    try:
        # A UI Gradio precisa de um objeto 'progress', a API não.
        # Podemos criar um mock simples se a função exigir.
        class MockProgress:
            def __call__(self, *args, **kwargs):
                pass
        
        video_path, latent_path = aduc.task_generate_base(
            prompt=request.prompt,
            num_frames=request.num_frames,
            width=request.width,
            height=request.height,
            cfg=request.cfg,
            steps=request.steps,
            progress=MockProgress()
        )
        return {"video_path": video_path, "latent_path": latent_path}
    except Exception as e:
        logger.error(f"API: Erro na geração - {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/regenerate", summary="Executa a cirurgia latente e retorna o vídeo final")
def regenerate_video(request: RegenerateRequest):
    """
    Recebe o caminho para um arquivo de latentes e os parâmetros da edição,
    executa a regeneração e retorna o vídeo final como um arquivo.
    """
    logger.info(f"API: Recebida requisição de regeneração para: '{request.quantum_twin_path}'")
    if not os.path.exists(request.quantum_twin_path):
        raise HTTPException(status_code=404, detail="Arquivo de latentes não encontrado no servidor.")
    
    try:
        class MockProgress:
            def __call__(self, *args, **kwargs):
                pass

        final_video_path = aduc.task_regenerate_video(
            x_cut=request.x_cut,
            n_cut=request.n_cut,
            n_eco=request.n_eco,
            progress=MockProgress()
        )
        return FileResponse(final_video_path, media_type="video/mp4", filename=os.path.basename(final_video_path))
    except Exception as e:
        logger.error(f"API: Erro na regeneração - {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# --- Execução do Servidor ---
# Para rodar: uvicorn api:app --reload