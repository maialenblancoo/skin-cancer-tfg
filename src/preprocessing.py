import cv2
import numpy as np


def apply_dullrazor(image: np.ndarray) -> np.ndarray:
    """
    Elimina el vello de imágenes dermatoscópicas.
    Entrada/salida: imagen BGR (numpy array).
    """
    # Convertir a escala de grises
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Kernel para detectar estructuras alargadas (pelos)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))

    # Blackhat: detecta estructuras oscuras sobre fondo claro (pelos oscuros)
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

    # Umbral para crear máscara de vello
    _, mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)

    # Inpainting: rellena las zonas de vello con el contexto alrededor
    result = cv2.inpaint(image, mask, inpaintRadius=6, flags=cv2.INPAINT_TELEA)

    return result


def apply_color_constancy(image: np.ndarray, power: float = 6.0) -> np.ndarray:
    """
    Normalización de color (Shades of Gray).
    Reduce el efecto de la iluminación variable entre cámaras/consultas.
    Entrada/salida: imagen BGR (numpy array).
    """
    # Trabajar en float
    img = image.astype(np.float32)

    # Calcular la norma de Minkowski por canal
    norms = np.power(np.mean(np.power(img, power), axis=(0, 1)), 1.0 / power)

    # Factor de escala para igualar los canales a la norma media
    scale = np.mean(norms) / (norms + 1e-6)

    # Aplicar y recortar a rango válido
    result = np.clip(img * scale, 0, 255).astype(np.uint8)

    return result


def preprocess_image(image: np.ndarray, mode: str) -> np.ndarray:
    """
    Aplica el preprocesado según el modo del experimento.

    mode:
        "none"           -> sin preprocesado
        "dullrazor"      -> solo DullRazor
        "colorconstancy" -> solo Color Constancy
        "both"           -> DullRazor + Color Constancy
    """
    if mode == "none":
        return image

    if mode == "dullrazor":
        return apply_dullrazor(image)

    if mode == "colorconstancy":
        return apply_color_constancy(image)

    if mode == "both":
        image = apply_dullrazor(image)
        image = apply_color_constancy(image)
        return image

    raise ValueError(f"Modo de preprocesado desconocido: '{mode}'. "
                     f"Usa: none | dullrazor | colorconstancy | both")