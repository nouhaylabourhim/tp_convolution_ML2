"""
TP Convolution - Filtres de convolution sur les images
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from pathlib import Path


# FONCTIONS DE BASE 

def convolve_channel(image, kernel):
   
    # Assertions pour valider les dimensions
    assert image.shape[0] >= kernel.shape[0], "L'image est trop petite pour le noyau en hauteur"
    assert image.shape[1] >= kernel.shape[1], "L'image est trop petite pour le noyau en largeur"
    
    # Dimensions de l'image et du noyau
    img_height, img_width = image.shape
    kernel_height, kernel_width = kernel.shape
    pad_height, pad_width = kernel_height // 2, kernel_width // 2
    
    # Ajouter un padding pour gérer les bords
    padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), 
                          mode='constant', constant_values=0)
    output = np.zeros_like(image, dtype=np.float32)
    
    # Appliquer la convolution
    for i in range(img_height):
        for j in range(img_width):
            region = padded_image[i:i + kernel_height, j:j + kernel_width]
            output[i, j] = np.sum(region * kernel)
    
    # Vérification de la taille de la sortie
    assert output.shape == image.shape, "La taille de la sortie ne correspond pas à l'image d'entrée"
    
    return output


def apply_convolution(image, kernel):
    
    # Assertions pour valider les entrées
    assert isinstance(image, np.ndarray), "L'image doit être un tableau NumPy"
    assert isinstance(kernel, np.ndarray), "Le noyau doit être un tableau NumPy"
    assert len(kernel.shape) == 2, "Le noyau doit être une matrice 2D"
    assert kernel.shape[0] == kernel.shape[1], "Le noyau doit être carré"
    assert kernel.shape[0] % 2 == 1, "Le noyau doit avoir une taille impaire"
    assert len(image.shape) in [2, 3], "L'image doit être en niveaux de gris (2D) ou RGB (3D)"
    
    if len(image.shape) == 3:  # Image RGB
        assert image.shape[2] == 3, "L'image RGB doit avoir 3 canaux"
        output = np.zeros_like(image, dtype=np.float32)
        # Appliquer le filtre sur chaque canal
        for c in range(image.shape[2]):
            output[:, :, c] = convolve_channel(image[:, :, c], kernel)
    else:  # Image en niveaux de gris
        output = convolve_channel(image, kernel)
    
    # Normaliser pour éviter des valeurs hors plage [0, 255]
    output = np.clip(output, 0, 255).astype(np.uint8)
    return output


def load_image(image_path, mode='auto'):
 
    # Charger l'image en BGR
    img_bgr = cv2.imread(image_path)
    assert img_bgr is not None, f"Erreur: Impossible de charger l'image '{image_path}'"
    
    if mode == 'gray':
        gray_image = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        assert len(gray_image.shape) == 2, "L'image doit être en niveaux de gris"
        return gray_image
    elif mode == 'rgb':
        rgb_image = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        assert len(rgb_image.shape) == 3 and rgb_image.shape[2] == 3, "L'image doit être RGB"
        return rgb_image
    else:  # mode 'auto'
        if len(img_bgr.shape) == 2:
            return img_bgr
        else:
            return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def create_kernel_blur(size=3):
   
    assert size % 2 == 1, "La taille doit être impaire"
    return np.ones((size, size), dtype=np.float32) / (size * size)


def create_kernel_gaussian_blur(size=3, sigma=1.0):
   
    assert size % 2 == 1, "La taille doit être impaire"
    kernel = np.zeros((size, size), dtype=np.float32)
    center = size // 2
    
    for i in range(size):
        for j in range(size):
            x = i - center
            y = j - center
            kernel[i, j] = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    
    return kernel / np.sum(kernel)


def create_kernel_sobel_horizontal():
    """Crée le noyau Sobel pour détection des contours horizontaux"""
    return np.array([[-1, 0, 1],
                     [-2, 0, 2],
                     [-1, 0, 1]], dtype=np.float32)


def create_kernel_sobel_vertical():
    """Crée le noyau Sobel pour détection des contours verticaux"""
    return np.array([[-1, -2, -1],
                     [0, 0, 0],
                     [1, 2, 1]], dtype=np.float32)


def create_kernel_prewitt_horizontal():
    """Crée le noyau Prewitt pour détection des contours horizontaux"""
    return np.array([[-1, 0, 1],
                     [-1, 0, 1],
                     [-1, 0, 1]], dtype=np.float32)


def create_kernel_prewitt_vertical():
    """Crée le noyau Prewitt pour détection des contours verticaux"""
    return np.array([[-1, -1, -1],
                     [0, 0, 0],
                     [1, 1, 1]], dtype=np.float32)


def create_kernel_laplacian():
    """Crée le noyau Laplacien pour détection des contours"""
    return np.array([[0, -1, 0],
                     [-1, 4, -1],
                     [0, -1, 0]], dtype=np.float32)


def create_kernel_sharpen():
    """Crée le noyau de netteté (sharpening)"""
    return np.array([[0, -1, 0],
                     [-1, 5, -1],
                     [0, -1, 0]], dtype=np.float32)


def create_kernel_emboss():
    """Crée le noyau Emboss pour effet de relief"""
    return np.array([[-2, -1, 0],
                     [-1, 1, 1],
                     [0, 1, 2]], dtype=np.float32)


def create_kernel_random(size=3, seed=42):

    assert size % 2 == 1, "La taille doit être impaire"
    np.random.seed(seed)
    kernel = np.random.randn(size, size)
    # Normalisation pour éviter des valeurs trop grandes
    kernel = kernel / np.sum(np.abs(kernel))
    return kernel


def get_all_kernels():
  
    kernels = {
        # Flous
        'Flou 3x3': create_kernel_blur(3),
        'Flou 5x5': create_kernel_blur(5),
        'Flou 7x7': create_kernel_blur(7),
        'Flou 9x9': create_kernel_blur(9),
        'Flou Gaussien 3x3': create_kernel_gaussian_blur(3, 1.0),
        'Flou Gaussien 5x5': create_kernel_gaussian_blur(5, 1.5),
        
        # Détection de contours
        'Sobel Horizontal': create_kernel_sobel_horizontal(),
        'Sobel Vertical': create_kernel_sobel_vertical(),
        'Prewitt Horizontal': create_kernel_prewitt_horizontal(),
        'Prewitt Vertical': create_kernel_prewitt_vertical(),
        'Laplacien': create_kernel_laplacian(),
        
        # Autres effets
        'Netteté': create_kernel_sharpen(),
        'Emboss': create_kernel_emboss(),
        
        # Filtres aléatoires
        'Aléatoire 3x3': create_kernel_random(3, 42),
        'Aléatoire 5x5': create_kernel_random(5, 42),
        'Aléatoire 7x7': create_kernel_random(7, 42),
    }
    return kernels


def display_images_grid(original, filtered_images, titles, figsize=(15, 10)):
   
    n_filters = len(filtered_images)
    fig, axes = plt.subplots(1, n_filters + 1, figsize=figsize)
    
    # Afficher l'image originale
    cmap = 'gray' if len(original.shape) == 2 else None
    axes[0].imshow(original, cmap=cmap)
    axes[0].set_title("Originale", fontsize=12)
    axes[0].axis('off')
    
    # Afficher les images filtrées
    for i, (filtered, title) in enumerate(zip(filtered_images, titles)):
        cmap = 'gray' if len(filtered.shape) == 2 else None
        axes[i + 1].imshow(filtered, cmap=cmap)
        axes[i + 1].set_title(title, fontsize=10)
        axes[i + 1].axis('off')
    
    plt.tight_layout()
    plt.show()


def display_comparison_figure(original, filtered_dict, title_figure="Comparaison des filtres"):
  
    n_filters = len(filtered_dict)
    fig, axes = plt.subplots(2, (n_filters + 1) // 2 + 1, figsize=(15, 10))
    axes = axes.flatten()
    
    # Afficher l'image originale
    cmap = 'gray' if len(original.shape) == 2 else None
    axes[0].imshow(original, cmap=cmap)
    axes[0].set_title("Originale", fontsize=12)
    axes[0].axis('off')
    
    # Afficher les images filtrées
    for i, (name, filtered) in enumerate(filtered_dict.items()):
        cmap = 'gray' if len(filtered.shape) == 2 else None
        axes[i + 1].imshow(filtered, cmap=cmap)
        axes[i + 1].set_title(name, fontsize=10)
        axes[i + 1].axis('off')
    
    # Cacher les axes inutilisés
    for i in range(len(filtered_dict) + 1, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle(title_figure, fontsize=14)
    plt.tight_layout()
    plt.show()


def save_filtered_image(image, filename, output_dir="results"):
   
    os.makedirs(output_dir, exist_ok=True)
    
    if len(image.shape) == 3:  # RGB
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f'{output_dir}/{filename}', image_bgr)
    else:  # Grayscale
        cv2.imwrite(f'{output_dir}/{filename}', image)


def compute_gradient_magnitude(sobel_h, sobel_v):
   
    magnitude = np.sqrt(sobel_h.astype(np.float32)**2 + sobel_v.astype(np.float32)**2)
    magnitude = np.clip(magnitude, 0, 255).astype(np.uint8)
    return magnitude