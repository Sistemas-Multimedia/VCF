"""
Estimación de movimiento bidireccional usando block matching.

Este módulo implementa la estimación de movimiento bidireccional necesaria
para MCTF (Motion Compensated Temporal Filtering).

Referencias:
    - Ohm, J.R. (1994). "Three-dimensional subband coding with motion compensation"
      IEEE Transactions on Image Processing, 9:559-571
    - Choi, S.-J., Woods, J.W. (1999). "Motion compensated 3-D subband coding of video"
      IEEE Transactions on Image Processing, 2:155-167
    - González-Ruiz, V. "MCTF"
      https://github.com/vicente-gonzalez-ruiz/motion_compensated_temporal_filtering
"""

import numpy as np
from typing import Tuple
import logging


def block_matching_bidirectional(
    frame_current: np.ndarray,
    frame_prev: np.ndarray,
    frame_next: np.ndarray,
    block_size: int = 16,
    search_range: int = 16
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimación de movimiento bidireccional usando block matching.
    
    Implementa Full Search Block Matching con métrica SAD (Sum of Absolute Differences)
    para encontrar los mejores vectores de movimiento hacia adelante y hacia atrás.
    
    Args:
        frame_current: Frame actual (el que se está prediciendo)
        frame_prev: Frame anterior (referencia hacia atrás)
        frame_next: Frame siguiente (referencia hacia adelante)
        block_size: Tamaño del bloque NxN (default: 16)
        search_range: Rango de búsqueda ±pixels (default: 16)
        
    Returns:
        mv_forward: Vectores de movimiento hacia adelante (current→next) [blocks_h, blocks_w, 2]
        mv_backward: Vectores de movimiento hacia atrás (current→prev) [blocks_h, blocks_w, 2]
    """
    
    height, width = frame_current.shape[:2]
    
    # Número de bloques en cada dimensión
    blocks_h = height // block_size
    blocks_w = width // block_size
    
    # Inicializar campos de vectores de movimiento
    mv_forward = np.zeros((blocks_h, blocks_w, 2), dtype=np.float32)
    mv_backward = np.zeros((blocks_h, blocks_w, 2), dtype=np.float32)
    
    logging.debug(f"Block matching: {blocks_h}x{blocks_w} blocks, size={block_size}, range={search_range}")
    
    # Iterar sobre cada bloque
    for by in range(blocks_h):
        for bx in range(blocks_w):
            # Coordenadas del bloque actual
            y_start = by * block_size
            x_start = bx * block_size
            y_end = y_start + block_size
            x_end = x_start + block_size
            
            # Extraer bloque actual
            current_block = frame_current[y_start:y_end, x_start:x_end]
            
            # === Búsqueda hacia adelante (current → next) ===
            mv_forward[by, bx] = _search_best_match(
                current_block, frame_next, 
                y_start, x_start, block_size,
                height, width, search_range
            )
            
            # === Búsqueda hacia atrás (current → prev) ===
            mv_backward[by, bx] = _search_best_match(
                current_block, frame_prev,
                y_start, x_start, block_size,
                height, width, search_range
            )
    
    return mv_forward, mv_backward


def _search_best_match(
    current_block: np.ndarray,
    reference_frame: np.ndarray,
    y_start: int,
    x_start: int,
    block_size: int,
    height: int,
    width: int,
    search_range: int
) -> Tuple[float, float]:
    """
    Busca el mejor match para un bloque en el frame de referencia.
    
    Args:
        current_block: Bloque a buscar
        reference_frame: Frame donde buscar
        y_start, x_start: Posición del bloque en el frame original
        block_size: Tamaño del bloque
        height, width: Dimensiones del frame
        search_range: Rango de búsqueda
        
    Returns:
        (dx, dy): Vector de movimiento que minimiza SAD
    """
    
    min_sad = float('inf')
    best_mv = (0, 0)
    
    for dy in range(-search_range, search_range + 1):
        for dx in range(-search_range, search_range + 1):
            # Coordenadas en frame de referencia
            ref_y = y_start + dy
            ref_x = x_start + dx
            
            # Verificar límites
            if (ref_y >= 0 and ref_y + block_size <= height and
                ref_x >= 0 and ref_x + block_size <= width):
                
                # Extraer bloque de referencia
                ref_block = reference_frame[
                    ref_y:ref_y + block_size,
                    ref_x:ref_x + block_size
                ]
                
                # Calcular SAD (Sum of Absolute Differences) - vectorizado
                sad = np.sum(np.abs(
                    current_block.astype(np.float32) - 
                    ref_block.astype(np.float32)
                ))
                
                # Actualizar mejor match
                if sad < min_sad:
                    min_sad = sad
                    best_mv = (dx, dy)
    
    return best_mv

