#Uma implementação aberta e funcional da arquitetura ADUC-SDR para geração de vídeo coerente.
# Copyright (C) 4 de Agosto de 2025  Carlos Rodrigues dos Santos
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