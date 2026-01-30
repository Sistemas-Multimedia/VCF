import os
import matplotlib.pyplot as plt
import cv2
import numpy as np

def show_images(original_path, encoded_path, decoded_path, title):
    """
    Función para visualizar y comparar las imágenes original, codificada y decodificada.

    Parámetros:
        original_path: ruta a la imagen original
        encoded_path: ruta a la imagen codificada (coeficientes)
        decoded_path: ruta a la imagen decodificada
        title: título para la visualización
    """
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(title, fontsize=16)

    # Mostrar imagen original
    if os.path.exists(original_path):
        img = cv2.cvtColor(cv2.imread(original_path), cv2.COLOR_BGR2RGB)
        axs[0].imshow(img)
        axs[0].set_title(f'Original\nTamaño: {os.path.getsize(original_path)/1024:.1f} KB')
    else:
        axs[0].text(0.5, 0.5, 'No encontrada', ha='center')
    axs[0].axis('off')

    # Mostrar imagen codificada (Coeficientes)
    # Nota: Se muestra el archivo de coeficientes como imagen. Puede parecer ruido o bloques.
    if os.path.exists(encoded_path):
        try:
            # Intentar leer con OpenCV (maneja archivos TIF)
            img_enc = cv2.imread(encoded_path, cv2.IMREAD_UNCHANGED)
            if img_enc is not None:
                if len(img_enc.shape) == 3:
                     # Convertir BGR a RGB si es imagen de color
                     img_enc = cv2.cvtColor(img_enc, cv2.COLOR_BGR2RGB)
                # Mostrar los coeficientes sin normalización
                axs[1].imshow(img_enc, cmap='gray')
            else:
                 axs[1].text(0.5, 0.5, 'No se pudo leer TIF', ha='center')
        except Exception as e:
             axs[1].text(0.5, 0.5, f'Error: {e}', ha='center')

        # Calcular y mostrar tamaños de archivos
        weights_path = encoded_path.replace('.tif', '_weights.bin')
        w_size = os.path.getsize(weights_path) if os.path.exists(weights_path) else 0
        enc_size = os.path.getsize(encoded_path)
        total_size = enc_size + w_size
        axs[1].set_title(f'Coeficientes + Pesos\nTIF: {enc_size/1024:.1f} KB | Pesos: {w_size/1024:.1f} KB\nTotal: {total_size/1024:.1f} KB')
    else:
        axs[1].text(0.5, 0.5, 'No encontrada', ha='center')
    axs[1].axis('off')

    # Mostrar imagen decodificada (reconstruida)
    if os.path.exists(decoded_path):
        img_dec = cv2.cvtColor(cv2.imread(decoded_path), cv2.COLOR_BGR2RGB)
        axs[2].imshow(img_dec)
        axs[2].set_title(f'Decodificada\nTamaño: {os.path.getsize(decoded_path)/1024:.1f} KB')
    else:
        axs[2].text(0.5, 0.5, 'No encontrada', ha='center')
    axs[2].axis('off')

    plt.tight_layout()
    plt.show()

def load_rgb(path):
    """Carga una imagen en formato RGB."""
    if not os.path.exists(path): return np.zeros((10,10,3), dtype=np.uint8)
    img = cv2.imread(path)
    if img is None: return np.zeros((10,10,3), dtype=np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def show_results_bm3d(img_name, original_path, filtered_paths):
    """Muestra los resultados: Original + 3 filtradas (BM3D)."""
    # Configuración de niveles
    levels = ['Baja', 'Media', 'Alta']

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # Imagen Original
    axes[0].imshow(load_rgb(original_path))
    axes[0].set_title(f"Original ({img_name})", fontsize=12, fontweight='bold')
    axes[0].axis('on')
    axes[0].set_xticks([]); axes[0].set_yticks([])

    # Resultados Filtrados (BM3D)
    for i in range(3):
        if i < len(filtered_paths):
            axes[i+1].imshow(load_rgb(filtered_paths[i]))
            axes[i+1].set_title(f"BM3D - {levels[i]}", fontsize=12)
            axes[i+1].axis('on')
            axes[i+1].set_xticks([]); axes[i+1].set_yticks([])

    plt.tight_layout()
    plt.suptitle(f"Resultados: {img_name}", fontsize=16, y=1.05)
    plt.show()

def show_results_nlm(img_name, original_path, filtered_paths):
    """Muestra los resultados: Original + 3 filtradas (NLM)."""
    # Configuración de niveles
    levels = ['Baja', 'Media', 'Alta']

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # Imagen Original
    axes[0].imshow(load_rgb(original_path))
    axes[0].set_title(f"Original ({img_name})", fontsize=12, fontweight='bold')
    axes[0].axis('on')
    axes[0].set_xticks([]); axes[0].set_yticks([])

    # Resultados Filtrados (NLM)
    for i in range(3):
        if i < len(filtered_paths):
            axes[i+1].imshow(load_rgb(filtered_paths[i]))
            axes[i+1].set_title(f"NLM - {levels[i]}", fontsize=12)
            axes[i+1].axis('on')
            axes[i+1].set_xticks([]); axes[i+1].set_yticks([])

    plt.tight_layout()
    plt.suptitle(f"Resultados: {img_name}", fontsize=16, y=1.05)
    plt.show()
