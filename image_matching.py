"""
image_matching.py - Large-area scan and real-image relocation matching tools
"""

import cv2
import numpy as np

from afm_relocation import match_template_candidates, to_grayscale_u8


def find_template(search_image, template, method=cv2.TM_CCOEFF_NORMED):
    """Return template match top-left coordinates and score inside a search image."""
    if search_image is None or template is None:
        return None, None, 0.0

    search_image = to_grayscale_u8(search_image)
    template = to_grayscale_u8(template)

    result = cv2.matchTemplate(search_image, template, method)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)
    return int(max_loc[0]), int(max_loc[1]), float(max_val)


def match_reference_template(sample, template, center_x, center_y, half_range=600, top_k=3):
    """
    Search around the current position for a previously saved reference FOV.
    Returns the best matching FOV top-left coordinate and score.
    """
    if sample is None or template is None:
        return None

    sample_h, sample_w = sample.shape[:2]
    template_h, template_w = template.shape[:2]
    if template_h >= sample_h or template_w >= sample_w:
        return None

    start_x = int(max(0, center_x - half_range))
    start_y = int(max(0, center_y - half_range))
    end_x = int(min(sample_w - template_w, center_x + half_range))
    end_y = int(min(sample_h - template_h, center_y + half_range))

    if end_x < start_x or end_y < start_y:
        return None

    search_image = sample[start_y : end_y + template_h, start_x : end_x + template_w]
    candidates = match_template_candidates(search_image, template, top_k=top_k)
    if not candidates:
        return None
    best = candidates[0]
    score_gap = float(best["score"] - candidates[1]["score"]) if len(candidates) > 1 else float(best["score"])

    return {
        "x": start_x + int(best["x"]),
        "y": start_y + int(best["y"]),
        "score": float(best["score"]),
        "score_gap": score_gap,
        "candidates": [
            {
                "x": start_x + int(candidate["x"]),
                "y": start_y + int(candidate["y"]),
                "score": float(candidate["score"]),
            }
            for candidate in candidates
        ],
        "search_origin": (start_x, start_y),
    }
