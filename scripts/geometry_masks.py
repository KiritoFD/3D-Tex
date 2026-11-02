"""Map explicit foreground masks onto the recorded model grid."""
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def load_masks(scene, cameras, source="none", mask_dir=None):
    masks = []
    for frame in cameras:
        size = (frame["model_width"], frame["model_height"])
        if source == "none":
            masks.append(np.ones(size[::-1], dtype=bool))
            continue
        path = (scene / "images" / frame["file"] if source == "alpha" else
                Path(mask_dir) / (Path(frame["file"]).stem + ".png"))
        with Image.open(path) as image:
            if source == "alpha":
                if "A" not in image.getbands():
                    raise ValueError(f"Missing alpha channel: {path}")
                image = image.getchannel("A")
            else:
                image = image.convert("L")
            mask = np.asarray(image)
        affine = np.asarray(frame["pixel_affine_original_to_model"])
        mapped = cv2.warpAffine(mask, affine[:2], size, flags=cv2.INTER_NEAREST)
        masks.append(mapped > 127)
    return np.stack(masks)
