# optimization.py
# Focado apenas na otimização estável de quantização FP8.

import torch
import logging
from torchao.quantization import quantize_, float8_dynamic_activation_float8_weight

# Usamos type hints com strings para evitar importações circulares
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ltx_manager_helpers import LtxWorker

logger = logging.getLogger(__name__)

def can_optimize_fp8():
    """Verifica se a GPU atual suporta otimizações FP8."""
    if not torch.cuda.is_available():
        return False
    
    major, _ = torch.cuda.get_device_capability()
    
    if major >= 9: # Arquitetura Hopper
        logger.info(f"GPU com arquitetura Hopper ou superior (CC {major}.x) detectada. Ativando quantização FP8.")
        return True
    
    if major == 8:
        device_name = torch.cuda.get_device_name(0).lower()
        if "h100" in device_name or "l40" in device_name or "rtx 40" in device_name: # Arquitetura Ada Lovelace
            logger.info(f"GPU com arquitetura Ada Lovelace (CC 8.9, Nome: {device_name}) detectada. Ativando quantização FP8.")
            return True
        
    logger.warning(f"A GPU atual (CC {major}.x) não tem suporte otimizado para FP8. Pulando quantização.")
    return False

@torch.no_grad()
def optimize_ltx_worker(worker: "LtxWorker"):
    """Aplica quantização FP8 ao transformador do pipeline LTX."""
    pipeline = worker.pipeline
    device = worker.device
    
    logger.info(f"Iniciando quantização FP8 do transformador LTX no dispositivo {device}...")
    quantize_(pipeline.transformer, float8_dynamic_activation_float8_weight())
    
    torch.cuda.empty_cache()
    logger.info(f"Quantização FP8 do LTX Worker no dispositivo {device} concluída com sucesso!")