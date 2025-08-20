# hardware_manager.py
# Gerencia a detecção e alocação de GPUs para os especialistas.
# Copyright (C) 2025 Carlos Rodrigues dos Santos

import torch
import logging

logger = logging.getLogger(__name__)

class HardwareManager:
    def __init__(self):
        self.gpus = []
        self.allocated_gpus = set()
        if torch.cuda.is_available():
            self.gpus = [f'cuda:{i}' for i in range(torch.cuda.device_count())]
        logger.info(f"Hardware Manager: Encontradas {len(self.gpus)} GPUs disponíveis: {self.gpus}")

    def allocate_gpus(self, specialist_name: str, num_required: int) -> list[str]:
        if not self.gpus or num_required == 0:
            logger.warning(f"Nenhuma GPU disponível ou solicitada para '{specialist_name}'. Alocando para CPU.")
            return ['cpu']

        available_gpus = [gpu for gpu in self.gpus if gpu not in self.allocated_gpus]
        
        if len(available_gpus) < num_required:
            error_msg = f"Recursos de GPU insuficientes para '{specialist_name}'. Solicitado: {num_required}, Disponível: {len(available_gpus)}."
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        allocated = available_gpus[:num_required]
        self.allocated_gpus.update(allocated)
        logger.info(f"Hardware Manager: Alocando GPUs {allocated} para o especialista '{specialist_name}'.")
        return allocated

hardware_manager = HardwareManager()