import os


class SamEnv:
    """
    - `points_per_side` define el número de puntos que se utilizan para generar las máscaras. Reducir este número puede acelerar el proceso de segmentación, pero también podría resultar en máscaras menos precisas.
    - `pred_iou_thresh` define el umbral para la intersección sobre la unión (IOU) en las predicciones. Aumentar este valor puede resultar en menos regiones propuestas, acelerando el proceso de segmentación.
    - Aumentar `stability_score_thresh` puede resultar en menos regiones propuestas.
    - `crop_n_layers` define el número de capas en el recorte de la imagen. Reducir este número puede acelerar el proceso de segmentación, pero podría afectar negativamente la precisión del modelo.
    - Aumentar `crop_n_points_downscale_factor` puede resultar en menos puntos en el recorte de la imagen, acelerando el proceso de segmentación.
    - Aumentar el área mínima de la región de la máscara: Aumentar min_mask_region_area puede resultar en menos regiones propuestas, acelerando el proceso de segmentación.
    """

    points_per_side = 32
    pred_iou_thresh = 0.86
    stability_score_thresh = 0.92
    crop_n_layers = 1
    crop_n_points_downscale_factor = 2
    min_mask_region_area = 40 * 40


class Data:
    def generation_path(name):
        return os.path.sep.join(["data", "images", "generation", name])
