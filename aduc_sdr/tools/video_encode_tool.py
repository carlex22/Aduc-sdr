# tools/video_encode_tool.py
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
#
# This file defines the VideoEncodeTool specialist. Its purpose is to abstract away
# the underlying command-line tools (like FFmpeg) used for video manipulation tasks
# such as concatenation and creating transitions. By encapsulating this logic, the core
# Deformes4D engine can remain agnostic to the specific tool being used, allowing for easier
# maintenance and future replacement with other libraries or tools.

import os
import subprocess
import logging
import random
import time
import shutil
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

class VideoToolError(Exception):
    """Custom exception for errors originating from the VideoEncodeTool."""
    pass

class VideoEncodeTool:
    """
    A specialist for handling video encoding and manipulation tasks.
    Currently uses FFmpeg as the backend.
    """

    def create_transition_bridge(self, start_image_path: str, end_image_path: str, 
                                 duration: float, fps: int, target_resolution: Tuple[int, int],
                                 workspace_dir: str, effect: Optional[str] = None) -> str:
        """
        Creates a short video clip that transitions between two static images using FFmpeg's xfade filter.
        This is useful for creating a "bridge" during a hard "cut" decided by the cinematic director.

        Args:
            start_image_path (str): The file path to the starting image.
            end_image_path (str): The file path to the ending image.
            duration (float): The desired duration of the transition in seconds.
            fps (int): The frames per second for the output video.
            target_resolution (Tuple[int, int]): The (width, height) of the output video.
            workspace_dir (str): The directory to save the output video.
            effect (Optional[str], optional): The specific xfade effect to use. If None, a random
                                              effect is chosen. Defaults to None.

        Returns:
            str: The file path to the generated transition video clip.
            
        Raises:
            VideoToolError: If the FFmpeg command fails.
        """
        output_path = os.path.join(workspace_dir, f"bridge_{int(time.time())}.mp4")
        width, height = target_resolution
        
        fade_effects = [
            "fade", "wipeleft", "wiperight", "wipeup", "wipedown", "dissolve", 
            "fadeblack", "fadewhite", "radial", "rectcrop", "circleopen", 
            "circleclose", "horzopen", "horzclose"
        ]
        
        selected_effect = effect if effect and effect.strip() else random.choice(fade_effects)
        
        transition_duration = max(0.1, duration)

        cmd = (
            f"ffmpeg -y -v error -loop 1 -t {transition_duration} -i \"{start_image_path}\" -loop 1 -t {transition_duration} -i \"{end_image_path}\" "
            f"-filter_complex \"[0:v]scale={width}:{height},setsar=1[v0];[1:v]scale={width}:{height},setsar=1[v1];"
            f"[v0][v1]xfade=transition={selected_effect}:duration={transition_duration}:offset=0[out]\" "
            f"-map \"[out]\" -c:v libx264 -r {fps} -pix_fmt yuv420p \"{output_path}\""
        )
        
        logger.info(f"Creating FFmpeg transition bridge with effect: '{selected_effect}' | Duration: {transition_duration}s")
        
        try:
            subprocess.run(cmd, shell=True, check=True, text=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg bridge creation failed. Return code: {e.returncode}")
            logger.error(f"FFmpeg command: {cmd}")
            logger.error(f"FFmpeg stderr: {e.stderr}")
            raise VideoToolError(f"Failed to create transition video. Details: {e.stderr}")
            
        return output_path

    def concatenate_videos(self, video_paths: List[str], output_path: str, workspace_dir: str):
        """
        Concatenates multiple video clips into a single file without re-encoding.

        Args:
            video_paths (List[str]): A list of absolute paths to the video clips to be concatenated.
            output_path (str): The absolute path for the final output video.
            workspace_dir (str): The directory to use for temporary files, like the concat list.
            
        Raises:
            VideoToolError: If no video paths are provided or if the FFmpeg command fails.
        """
        if not video_paths:
            raise VideoToolError("VideoEncodeTool: No video fragments provided for concatenation.")

        if len(video_paths) == 1:
            logger.info("Only one video clip found. Skipping concatenation and just copying the file.")
            shutil.copy(video_paths[0], output_path)
            return

        list_file_path = os.path.join(workspace_dir, "concat_list.txt")

        try:
            with open(list_file_path, 'w', encoding='utf-8') as f:
                for path in video_paths:
                    f.write(f"file '{os.path.abspath(path)}'\n")

            cmd_list = ['ffmpeg', '-y', '-f', 'concat', '-safe', '0', '-i', list_file_path, '-c', 'copy', output_path]
            
            logger.info(f"Concatenating {len(video_paths)} video clips into {output_path} using FFmpeg...")
            
            subprocess.run(cmd_list, check=True, capture_output=True, text=True)
            
            logger.info(f"FFmpeg concatenation successful. Final video is at: {output_path}")

        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg concatenation failed. Return code: {e.returncode}")
            logger.error(f"FFmpeg stderr: {e.stderr}")
            raise VideoToolError(f"Failed to assemble the final video using FFmpeg. Details: {e.stderr}")
        except Exception as e:
            logger.error(f"An unexpected error occurred during video concatenation: {e}", exc_info=True)
            raise VideoToolError("An unexpected error occurred during the final video assembly.")
        finally:
            if os.path.exists(list_file_path):
                os.remove(list_file_path)


# --- Singleton Instance ---
# We create a single instance of the tool to be imported by other modules.
video_encode_tool_singleton = VideoEncodeTool()