# ltx_upscaler_manager_helpers.py
# Gerente de Pool para o revezamento de workers de Upscaling.
# Este arquivo é parte do projeto Euia-AducSdr e está sob a licença AGPL v3.
# Copyright (C) 4 de Agosto de 2025  Carlos Rodrigues dos Santos

import torch
import gc
import os
import threading
from ltx_worker_upscaler import LtxUpscaler

class LtxUpscalerPoolManager:
    """
    Gerencia um pool de LtxUpscalerWorkers, orquestrando um revezamento entre GPUs
    para a tarefa de upscaling.
    """
    def __init__(self, device_ids=['cuda:2', 'cuda:3']):
        print(f"LTX UPSCALER POOL MANAGER: Criando workers para os dispositivos: {device_ids}")
        self.workers = [LtxUpscaler(device_id) for device_id in device_ids]
        self.current_worker_index = 0
        self.lock = threading.Lock()
        self.last_cleanup_thread = None

    def _cleanup_worker(self, worker):
        """Função alvo para a thread de limpeza em background."""
        print(f"UPSCALER CLEANUP THREAD: Iniciando limpeza da GPU {worker.device}...")
        worker.to_cpu()
        print(f"UPSCALER CLEANUP THREAD: Limpeza da GPU {worker.device} concluída.")

    def upscale_video_fragment(self, video_path_low_res: str, output_path: str, video_fps: int):
        """
        Seleciona um worker livre, faz o upscale de um fragmento e limpa o worker anterior.
        """
        worker_to_use = None
        try:
            with self.lock:
                if self.last_cleanup_thread and self.last_cleanup_thread.is_alive():
                    print("UPSCALER POOL MANAGER: Aguardando limpeza da GPU anterior...")
                    self.last_cleanup_thread.join()

                worker_to_use = self.workers[self.current_worker_index]
                previous_worker_index = (self.current_worker_index - 1 + len(self.workers)) % len(self.workers)
                worker_to_cleanup = self.workers[previous_worker_index]

                cleanup_thread = threading.Thread(target=self._cleanup_worker, args=(worker_to_cleanup,))
                cleanup_thread.start()
                self.last_cleanup_thread = cleanup_thread
                
                worker_to_use.to_gpu()
                
                self.current_worker_index = (self.current_worker_index + 1) % len(self.workers)
            
            print(f"UPSCALER POOL MANAGER: Worker em {worker_to_use.device} iniciando upscale de {os.path.basename(video_path_low_res)}...")
            worker_to_use.upscale_video_fragment(video_path_low_res, output_path, video_fps)
            print(f"UPSCALER POOL MANAGER: Upscale de {os.path.basename(video_path_low_res)} concluído.")

        finally:
            # A limpeza do worker_to_use será feita na próxima chamada
            pass

# --- Instância Singleton do Gerenciador de Upscaling ---
ltx_upscaler_manager_singleton = LtxUpscalerPoolManager(device_ids=['cuda:2', 'cuda:3'])