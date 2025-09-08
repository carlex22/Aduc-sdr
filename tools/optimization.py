# optimization.py
# AducSdr: Uma implementação aberta e funcional da arquitetura ADUC-SDR
# Copyright (C) 4 de Agosto de 2025  Carlos Rodrigues dos Santos
#
# Contato:
# Carlos Rodrigues dos Santos
# carlex22@gmail.com
# Rua Eduardo Carlos Pereira, 4125, B1 Ap32, Curitiba, PR, Brazil, CEP 8102025
#
# Repositórios e Projetos Relacionados:
# GitHub: https://github.com/carlex22/Aduc-sdr
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License...
# PENDING PATENT NOTICE: Please see NOTICE.md.

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