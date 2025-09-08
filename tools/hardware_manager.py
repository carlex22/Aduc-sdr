# tools/hardware_manager.py
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
#
# Version 1.0.1

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