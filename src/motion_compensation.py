"""
Compensación de movimiento para MCTF.

Este módulo implementa la compensación de movimiento que aplica vectores
de movimiento a frames de referencia para generar predicciones.

Referencias:
    - González-Ruiz, V. "Motion Compensation"
      https://github.com/vicente-gonzalez-ruiz/motion_compensation
    - Pesquet-Popescu, B., Bottreau, V. (2001). "Three-dimensional lifting 
      schemes for motion compensated video compression"
"""

import numpy as np
import logging


def motion_compensate(
    frame: np.ndarray,
    motion_vectors: np.ndarray,
    block_size: int = 16
) -> np.ndarray:
    """
    Aplica compensación de movimiento a un frame.
    
    Genera un frame compensado desplazando bloques según los vectores
    de movimiento proporcionados.
    
    Args:
        frame: Frame de referencia
        motion_vectors: Campo de vectores de movimiento [blocks_h, blocks_w, 2]
        block_size: Tamaño del bloque (default: 16)
        
    Returns:
        compensated_frame: Frame compensado por movimiento
    """
    
    height, width = frame.shape[:2]
    compensated_frame = np.zeros_like(frame, dtype=np.float32)
    
    blocks_h, blocks_w = motion_vectors.shape[:2]
    
    logging.debug(f"Motion compensate: {blocks_h}x{blocks_w} blocks, size={block_size}")
    
    for by in range(blocks_h):
        for bx in range(blocks_w):
            # Coordenadas del bloque destino
            y_start = by * block_size
            x_start = bx * block_size
            y_end = min(y_start + block_size, height)
            x_end = min(x_start + block_size, width)
            
            # Vector de movimiento
            dx, dy = motion_vectors[by, bx]
            
            # Coordenadas en frame de referencia
            ref_y = int(y_start + dy)
            ref_x = int(x_start + dx)
            
            # Verificar límites
            if (ref_y >= 0 and ref_y + block_size <= height and
                ref_x >= 0 and ref_x + block_size <= width):
                
                # Copiar bloque compensado
                compensated_frame[y_start:y_end, x_start:x_end] = \
                    frame[ref_y:ref_y + (y_end - y_start), 
                          ref_x:ref_x + (x_end - x_start)]
            else:
                # Si está fuera de límites, copiar bloque original
                compensated_frame[y_start:y_end, x_start:x_end] = \
                    frame[y_start:y_end, x_start:x_end]
    
    return compensated_frame


def motion_compensate_bidirectional(
    frame_prev: np.ndarray,
    frame_next: np.ndarray,
    mv_backward: np.ndarray,
    mv_forward: np.ndarray,
    block_size: int = 16
) -> np.ndarray:
    """
    Compensación de movimiento bidireccional para frames B.
    
    Genera una predicción bidireccional promediando las compensaciones
    desde el frame anterior y el siguiente.
    
    Args:
        frame_prev: Frame anterior (referencia backward)
        frame_next: Frame siguiente (referencia forward)
        mv_backward: Vectores de movimiento hacia atrás
        mv_forward: Vectores de movimiento hacia adelante
        block_size: Tamaño del bloque (default: 16)
        
    Returns:
        prediction: Frame de predicción bidireccional
    """
    
    # Compensar desde frame anterior
    mc_prev = motion_compensate(frame_prev, mv_backward, block_size)
    
    # Compensar desde frame siguiente
    mc_next = motion_compensate(frame_next, mv_forward, block_size)
    
    # Predicción bidireccional (promedio)
    prediction = (mc_prev + mc_next) / 2.0
    
    return prediction

