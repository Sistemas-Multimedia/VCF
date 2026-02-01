"""
Filtrado temporal usando lifting scheme con compensación de movimiento.

Este módulo implementa el filtrado temporal wavelet para MCTF usando
el esquema de lifting con pasos Predict y Update.

Referencias:
    - Pesquet-Popescu, B., Bottreau, V. (2001). "Three-dimensional lifting 
      schemes for motion compensated video compression"
    - Secker, A., Taubman, D. (2003). "Lifting-based invertible motion 
      adaptive transform (LIMAT) framework"
    - González-Ruiz, V. "MCTF"
      https://github.com/vicente-gonzalez-ruiz/motion_compensated_temporal_filtering
"""

import numpy as np
from typing import List, Tuple
import logging

from motion_compensation import motion_compensate


# Coeficientes de wavelets para lifting scheme
WAVELET_COEFFICIENTS = {
    'haar': {
        'predict': 1.0,
        'update': 0.5
    },
    '5/3': {
        'predict': 0.5,
        'update': 0.25
    },
    '9/7': {
        'predict': 1.586134342,
        'update': 0.052980118
    }
}


def get_wavelet_coefficients(wavelet_type: str) -> Tuple[float, float]:
    """
    Obtiene los coeficientes de predicción y actualización para un wavelet.
    
    Args:
        wavelet_type: Tipo de wavelet ('haar', '5/3', '9/7')
        
    Returns:
        (predict_coef, update_coef): Coeficientes del lifting scheme
        
    Raises:
        ValueError: Si el tipo de wavelet no está soportado
    """
    if wavelet_type not in WAVELET_COEFFICIENTS:
        raise ValueError(
            f"Wavelet type '{wavelet_type}' not supported. "
            f"Supported types: {list(WAVELET_COEFFICIENTS.keys())}"
        )
    
    coeffs = WAVELET_COEFFICIENTS[wavelet_type]
    return coeffs['predict'], coeffs['update']


def temporal_filter_lifting(
    frames: List[np.ndarray],
    motion_vectors_forward: List[np.ndarray],
    motion_vectors_backward: List[np.ndarray],
    wavelet_type: str = '5/3',
    block_size: int = 16
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Aplica filtrado temporal usando lifting scheme con compensación de movimiento.
    
    Implementa el esquema IBB... donde los frames pares son low-pass (L)
    y los impares generan high-pass (H) como residuos de predicción.
    
    Args:
        frames: Lista de frames a filtrar
        motion_vectors_forward: Lista de MVs hacia adelante
        motion_vectors_backward: Lista de MVs hacia atrás
        wavelet_type: Tipo de wavelet ('haar', '5/3', '9/7')
        block_size: Tamaño de bloque para MC
        
    Returns:
        low_pass: Frames de baja frecuencia temporal (L)
        high_pass: Frames de alta frecuencia temporal (H/residuos)
    """
    
    n_frames = len(frames)
    predict_coef, update_coef = get_wavelet_coefficients(wavelet_type)
    
    logging.info(f"Temporal filtering {n_frames} frames with {wavelet_type} wavelet")
    
    low_pass = []
    high_pass = []
    
    # Procesar pares de frames (even, odd)
    for i in range(0, n_frames - 1, 2):
        frame_even = frames[i].astype(np.float32)      # Frame par (t=0,2,4...)
        frame_odd = frames[i + 1].astype(np.float32)   # Frame impar (t=1,3,5...)
        
        # === PREDICT STEP ===
        # Predecir frame odd usando MC desde frames even vecinos
        
        # MC desde frame anterior (even actual)
        mc_prev = motion_compensate(
            frame_even,
            motion_vectors_backward[i] if i < len(motion_vectors_backward) else np.zeros_like(motion_vectors_backward[0]),
            block_size
        )
        
        # MC desde frame siguiente (even i+2) si existe
        if i + 2 < n_frames:
            mc_next = motion_compensate(
                frames[i + 2].astype(np.float32),
                motion_vectors_forward[i + 1] if i + 1 < len(motion_vectors_forward) else np.zeros_like(motion_vectors_forward[0]),
                block_size
            )
        else:
            mc_next = mc_prev
        
        # Predicción bidireccional
        prediction = (mc_prev + mc_next) * predict_coef / 2.0
        
        # Residuo de alta frecuencia (H)
        h_frame = frame_odd - prediction
        high_pass.append(h_frame)
        
        # === UPDATE STEP ===
        # Actualizar frame even con información del residuo
        mc_residual = motion_compensate(
            h_frame,
            motion_vectors_forward[i] if i < len(motion_vectors_forward) else np.zeros_like(motion_vectors_forward[0]),
            block_size
        )
        
        # Actualización (L)
        l_frame = frame_even + mc_residual * update_coef
        low_pass.append(l_frame)
    
    # Si hay número impar de frames, el último pasa como low-pass
    if n_frames % 2 != 0:
        low_pass.append(frames[-1].astype(np.float32))
    
    logging.info(f"Temporal filtering complete: {len(low_pass)} L frames, {len(high_pass)} H frames")

    return low_pass, high_pass


def inverse_temporal_filter_lifting(
    low_pass: List[np.ndarray],
    high_pass: List[np.ndarray],
    motion_vectors_forward: List[np.ndarray],
    motion_vectors_backward: List[np.ndarray],
    wavelet_type: str = '5/3',
    block_size: int = 16
) -> List[np.ndarray]:
    """
    Reconstruye frames desde la descomposición temporal.

    Aplica el lifting scheme inverso para recuperar los frames originales
    desde los componentes L (low-pass) y H (high-pass).

    Args:
        low_pass: Frames L (baja frecuencia temporal)
        high_pass: Frames H (alta frecuencia temporal)
        motion_vectors_forward: MVs hacia adelante
        motion_vectors_backward: MVs hacia atrás
        wavelet_type: Tipo de wavelet usado en la codificación
        block_size: Tamaño de bloque para MC

    Returns:
        reconstructed_frames: Lista de frames reconstruidos
    """

    predict_coef, update_coef = get_wavelet_coefficients(wavelet_type)

    n_low = len(low_pass)
    n_high = len(high_pass)

    logging.info(f"Inverse temporal filtering: {n_low} L frames, {n_high} H frames")

    reconstructed_frames = []

    for i in range(n_high):
        l_frame = low_pass[i].astype(np.float32)
        h_frame = high_pass[i].astype(np.float32)

        # === INVERSE UPDATE ===
        # Recuperar frame even original
        mc_residual = motion_compensate(
            h_frame,
            motion_vectors_forward[2 * i] if 2 * i < len(motion_vectors_forward) else np.zeros_like(motion_vectors_forward[0]),
            block_size
        )

        frame_even = l_frame - mc_residual * update_coef
        reconstructed_frames.append(frame_even)

        # === INVERSE PREDICT ===
        # Recuperar frame odd original

        # MC para predicción
        mc_prev = motion_compensate(
            frame_even,
            motion_vectors_backward[2 * i] if 2 * i < len(motion_vectors_backward) else np.zeros_like(motion_vectors_backward[0]),
            block_size
        )

        if i + 1 < n_low:
            mc_next = motion_compensate(
                low_pass[i + 1].astype(np.float32),
                motion_vectors_forward[2 * i + 1] if 2 * i + 1 < len(motion_vectors_forward) else np.zeros_like(motion_vectors_forward[0]),
                block_size
            )
        else:
            mc_next = mc_prev

        # Predicción bidireccional
        prediction = (mc_prev + mc_next) * predict_coef / 2.0

        # Inverse predict
        frame_odd = h_frame + prediction
        reconstructed_frames.append(frame_odd)

    # Si hubo frame extra en low_pass (número impar de frames)
    if n_low > n_high:
        reconstructed_frames.append(low_pass[-1].astype(np.float32))

    logging.info(f"Inverse temporal filtering complete: {len(reconstructed_frames)} frames")

    return reconstructed_frames

