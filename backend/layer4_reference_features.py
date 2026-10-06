"""Reference-image difference features for Layer 4.

The functions in this file compare a submitted/edited image with a known
reference image. They use image structure and pixel differences only; there
are no product names, Chinese phrases, or case-specific rules.
"""

from pathlib import Path

import cv2
import numpy as np


REFERENCE_FEATURE_NAMES = [
    "structural_difference",
    "mean_gray_difference",
    "mean_color_difference",
    "p99_5_color_difference",
    "changed_pixel_ratio",
    "largest_change_ratio",
]

ANALYSIS_MAX_SIDE = 1024


class Layer4ReferenceError(RuntimeError):
    """Raised when a target/reference pair cannot be compared."""


def load_cv_image(image_path: str | Path) -> np.ndarray:
    """Load one local image as an OpenCV BGR array."""

    path = Path(image_path)

    if not path.is_file():
        raise Layer4ReferenceError(f"Image file not found: {path}")

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)

    if image is None:
        raise Layer4ReferenceError(f"Image could not be decoded: {path}")

    return image


def _resize_for_analysis(image: np.ndarray) -> tuple[np.ndarray, float]:
    """Limit comparison size while retaining the scale for result boxes."""

    height, width = image.shape[:2]
    largest_side = max(height, width)

    if largest_side <= ANALYSIS_MAX_SIDE:
        return image, 1.0

    scale = ANALYSIS_MAX_SIDE / largest_side
    resized = cv2.resize(
        image,
        (round(width * scale), round(height * scale)),
        interpolation=cv2.INTER_AREA,
    )
    return resized, scale


def _calculate_ssim(gray_a: np.ndarray, gray_b: np.ndarray) -> float:
    """Calculate mean structural similarity without an extra dependency."""

    first = gray_a.astype(np.float64)
    second = gray_b.astype(np.float64)

    constant_1 = (0.01 * 255) ** 2
    constant_2 = (0.03 * 255) ** 2

    mean_first = cv2.GaussianBlur(first, (11, 11), 1.5)
    mean_second = cv2.GaussianBlur(second, (11, 11), 1.5)

    mean_first_squared = mean_first * mean_first
    mean_second_squared = mean_second * mean_second
    mean_product = mean_first * mean_second

    variance_first = (
        cv2.GaussianBlur(first * first, (11, 11), 1.5)
        - mean_first_squared
    )
    variance_second = (
        cv2.GaussianBlur(second * second, (11, 11), 1.5)
        - mean_second_squared
    )
    covariance = (
        cv2.GaussianBlur(first * second, (11, 11), 1.5)
        - mean_product
    )

    numerator = (
        (2 * mean_product + constant_1)
        * (2 * covariance + constant_2)
    )
    denominator = (
        (mean_first_squared + mean_second_squared + constant_1)
        * (variance_first + variance_second + constant_2)
    )

    similarity_map = numerator / np.maximum(denominator, 1e-12)
    return float(np.clip(np.mean(similarity_map), -1.0, 1.0))


def _round(value: float) -> float:
    return round(float(value), 6)


def compare_reference_arrays(
    target_image: np.ndarray,
    reference_image: np.ndarray,
) -> dict:
    """Compare two decoded BGR images and return features plus changed boxes."""

    if target_image is None or reference_image is None:
        raise Layer4ReferenceError("Target and reference images are required.")

    original_reference_height, original_reference_width = (
        reference_image.shape[:2]
    )
    original_target_size = (
        int(target_image.shape[1]),
        int(target_image.shape[0]),
    )
    original_reference_size = (
        int(reference_image.shape[1]),
        int(reference_image.shape[0]),
    )

    resized_to_reference = target_image.shape[:2] != reference_image.shape[:2]

    if resized_to_reference:
        target_image = cv2.resize(
            target_image,
            (original_reference_width, original_reference_height),
            interpolation=cv2.INTER_AREA,
        )

    target_small, analysis_scale = _resize_for_analysis(target_image)
    reference_small = cv2.resize(
        reference_image,
        (target_small.shape[1], target_small.shape[0]),
        interpolation=cv2.INTER_AREA,
    )

    # A small blur suppresses isolated JPEG noise while retaining purposeful
    # text, price, logo, and object changes.
    target_blurred = cv2.GaussianBlur(target_small, (3, 3), 0)
    reference_blurred = cv2.GaussianBlur(reference_small, (3, 3), 0)

    target_gray = cv2.cvtColor(target_blurred, cv2.COLOR_BGR2GRAY)
    reference_gray = cv2.cvtColor(reference_blurred, cv2.COLOR_BGR2GRAY)

    gray_difference = cv2.absdiff(target_gray, reference_gray)
    color_difference = cv2.absdiff(target_blurred, reference_blurred)
    strongest_channel_difference = np.max(color_difference, axis=2)

    structural_similarity = _calculate_ssim(target_gray, reference_gray)

    # Estimate the pair's noise level rather than using a product- or
    # case-specific pixel threshold. The small lower bound prevents tiny codec
    # variation from being marked as editing when the robust spread is zero.
    median_difference = float(np.median(strongest_channel_difference))
    median_absolute_deviation = float(
        np.median(
            np.abs(strongest_channel_difference - median_difference)
        )
    )
    noise_scale = 1.4826 * median_absolute_deviation
    pixel_threshold = float(
        np.clip(median_difference + 6.0 * noise_scale, 12.0, 80.0)
    )

    changed_mask = (
        strongest_channel_difference > pixel_threshold
    ).astype(np.uint8) * 255

    # Remove isolated noise and join adjacent strokes into usable regions.
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    changed_mask = cv2.morphologyEx(
        changed_mask,
        cv2.MORPH_OPEN,
        open_kernel,
    )
    changed_mask = cv2.morphologyEx(
        changed_mask,
        cv2.MORPH_CLOSE,
        close_kernel,
    )

    total_pixels = float(changed_mask.shape[0] * changed_mask.shape[1])
    changed_pixel_ratio = float(cv2.countNonZero(changed_mask) / total_pixels)

    contours, _ = cv2.findContours(
        changed_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    minimum_region_area = max(20.0, total_pixels * 0.00002)
    contour_areas = [float(cv2.contourArea(item)) for item in contours]
    meaningful = [
        (contour, area)
        for contour, area in zip(contours, contour_areas)
        if area >= minimum_region_area
    ]
    meaningful.sort(key=lambda item: item[1], reverse=True)

    largest_change_ratio = (
        meaningful[0][1] / total_pixels
        if meaningful
        else 0.0
    )

    inverse_scale = 1.0 / analysis_scale
    changed_regions = []

    for contour, area in meaningful[:10]:
        left, top, width, height = cv2.boundingRect(contour)
        changed_regions.append(
            {
                "left": round(left * inverse_scale),
                "top": round(top * inverse_scale),
                "right": round((left + width) * inverse_scale),
                "bottom": round((top + height) * inverse_scale),
                "area_ratio": _round(area / total_pixels),
            }
        )

    features = {
        "structural_difference": _round(1.0 - structural_similarity),
        "mean_gray_difference": _round(
            float(np.mean(gray_difference)) / 255.0
        ),
        "mean_color_difference": _round(
            float(np.mean(color_difference)) / 255.0
        ),
        "p99_5_color_difference": _round(
            float(np.percentile(strongest_channel_difference, 99.5))
            / 255.0
        ),
        "changed_pixel_ratio": _round(changed_pixel_ratio),
        "largest_change_ratio": _round(largest_change_ratio),
    }

    return {
        "features": features,
        "reference_similarity": _round(structural_similarity),
        "pixel_threshold": round(pixel_threshold, 2),
        "changed_regions": changed_regions,
        "changed_region_count": len(meaningful),
        "target_size": original_target_size,
        "reference_size": original_reference_size,
        "resized_to_reference": resized_to_reference,
        "analysis_scale": _round(analysis_scale),
    }


def compare_reference_images(
    target_path: str | Path,
    reference_path: str | Path,
) -> dict:
    """Load and compare a submitted image with its known reference image."""

    return compare_reference_arrays(
        load_cv_image(target_path),
        load_cv_image(reference_path),
    )


__all__ = [
    "Layer4ReferenceError",
    "REFERENCE_FEATURE_NAMES",
    "compare_reference_arrays",
    "compare_reference_images",
    "load_cv_image",
]
