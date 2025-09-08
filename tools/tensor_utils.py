# tools/tensor_utils.py
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
# Version: 1.0.1
#
# This module provides utility functions for tensor manipulation, specifically for
# image and video processing tasks. The functions here, such as wavelet reconstruction,
# are internalized within the ADUC-SDR framework to ensure stability and reduce
# reliance on specific external library structures.
#
# The wavelet_reconstruction code is adapted from the SeedVR project.

import torch
from torch import Tensor
from torch.nn import functional as F
from typing import Tuple

def wavelet_blur(image: Tensor, radius: int) -> Tensor:
    """
    Apply wavelet blur to the input tensor.
    """
    if image.ndim != 4: # Expects (B, C, H, W)
        raise ValueError(f"wavelet_blur expects a 4D tensor, but got shape {image.shape}")
        
    b, c, h, w = image.shape
    
    # convolution kernel
    kernel_vals = [
        [0.0625, 0.125, 0.0625],
        [0.125, 0.25, 0.125],
        [0.0625, 0.125, 0.0625],
    ]
    kernel = torch.tensor(kernel_vals, dtype=image.dtype, device=image.device)
    kernel = kernel[None, None] # (1, 1, 3, 3)
    
    # repeat the kernel across all input channels for grouped convolution
    kernel = kernel.repeat(c, 1, 1, 1) # (C, 1, 3, 3)
    
    image = F.pad(image, (radius, radius, radius, radius), mode='replicate')
    
    # apply convolution with groups=c to process each channel independently
    output = F.conv2d(image, kernel, groups=c, dilation=radius)
    return output

def wavelet_decomposition(image: Tensor, levels=5) -> Tuple[Tensor, Tensor]:
    """
    Apply wavelet decomposition to the input tensor.
    This function returns both the high frequency and low frequency components.
    """
    # Ensure tensor is 4D (B, C, H, W)
    is_video_frame = image.ndim == 5 # (B, C, F, H, W)
    if is_video_frame:
        b, c, f, h, w = image.shape
        image = image.permute(0, 2, 1, 3, 4).reshape(b * f, c, h, w)

    high_freq = torch.zeros_like(image)
    low_freq = image
    for i in range(levels):
        radius = 2 ** i
        blurred = wavelet_blur(low_freq, radius)
        high_freq += (low_freq - blurred)
        low_freq = blurred

    if is_video_frame:
        high_freq = high_freq.view(b, f, c, h, w).permute(0, 2, 1, 3, 4)
        low_freq = low_freq.view(b, f, c, h, w).permute(0, 2, 1, 3, 4)
        
    return high_freq, low_freq

def wavelet_reconstruction(content_feat: Tensor, style_feat: Tensor) -> Tensor:
    """
    Applies wavelet decomposition to transfer the color/style (low-frequency components)
    from a style feature to the details (high-frequency components) of a content feature.
    This works for both images (4D) and videos (5D).

    Args:
        content_feat (Tensor): The tensor containing the structural details.
        style_feat (Tensor): The tensor containing the desired color and lighting style.
    
    Returns:
        Tensor: The reconstructed tensor with content details and style colors.
    """
    # calculate the wavelet decomposition of the content feature
    content_high_freq, _ = wavelet_decomposition(content_feat)
    
    # calculate the wavelet decomposition of the style feature
    _, style_low_freq = wavelet_decomposition(style_feat)
    
    # reconstruct the content feature with the style's low frequency (color/lighting)
    return content_high_freq + style_low_freq