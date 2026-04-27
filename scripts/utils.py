"""
Utility functions for Anchor-Conditioned Compositional Control

This module contains helper functions for:
- Image processing and validation
- Model loading and configuration
- Evaluation metrics
- Pipeline orchestration
"""

import os
import torch
import numpy as np
from PIL import Image
import cv2
from typing import Dict, List, Tuple, Optional, Any
import json
from pathlib import Path


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)


def save_config(config: Dict[str, Any], config_path: str) -> None:
    """Save configuration to JSON file."""
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)


def validate_image(image_path: str) -> bool:
    """Validate that an image file exists and is readable."""
    if not os.path.exists(image_path):
        return False

    try:
        with Image.open(image_path) as img:
            img.verify()
        return True
    except Exception:
        return False


def resize_image(image: Image.Image, size: Tuple[int, int], method: str = 'bicubic') -> Image.Image:
    """Resize image with specified method."""
    methods = {
        'bicubic': Image.BICUBIC,
        'bilinear': Image.BILINEAR,
        'nearest': Image.NEAREST,
        'lanczos': Image.LANCZOS
    }

    return image.resize(size, methods.get(method, Image.BICUBIC))


def normalize_image(image: np.ndarray) -> np.ndarray:
    """Normalize image to [-1, 1] range for diffusion models."""
    return (image.astype(np.float32) / 127.5) - 1.0


def denormalize_image(image: np.ndarray) -> np.ndarray:
    """Denormalize image from [-1, 1] to [0, 255] range."""
    return ((image + 1.0) * 127.5).clip(0, 255).astype(np.uint8)


def compute_horizon_metrics(pred_horizon: float, target_horizon: float) -> Dict[str, float]:
    """Compute horizon detection metrics."""
    deviation = abs(pred_horizon - target_horizon)

    # Rule of thirds alignment (ideal positions: 0.333, 0.667)
    rule_of_thirds_positions = [1/3, 2/3]
    alignment = min([abs(pred_horizon - pos) for pos in rule_of_thirds_positions])

    return {
        'horizon_deviation': deviation,
        'rule_of_thirds_alignment': 1.0 - min(alignment * 3, 1.0),  # Normalize to [0, 1]
        'is_rule_of_thirds': alignment < 0.1  # Within 10% of rule of thirds
    }


def compute_image_sharpness(image: np.ndarray) -> float:
    """Compute image sharpness using Laplacian variance."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def setup_device() -> torch.device:
    """Setup the appropriate device (CUDA if available, else CPU)."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using CUDA device: {torch.cuda.get_device_name()}")
    else:
        device = torch.device('cpu')
        print("Using CPU device")

    return device


def get_model_size(model_path: str) -> str:
    """Get human-readable model size."""
    if os.path.isdir(model_path):
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(model_path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                total_size += os.path.getsize(filepath)
    else:
        total_size = os.path.getsize(model_path)

    # Convert to human readable
    for unit in ['B', 'KB', 'MB', 'GB']:
        if total_size < 1024.0:
            return ".1f"
        total_size /= 1024.0

    return ".1f"


def create_anchor_token(anchor_vector: np.ndarray, device: torch.device) -> torch.Tensor:
    """
    Create anchor token from Fourier-encoded anchor vector.

    Args:
        anchor_vector: Fourier-encoded 4D anchor (shape: [128])
        device: Target device

    Returns:
        Anchor token tensor of shape [1, 1, 768]
    """
    # This would be replaced with the actual AnchorProjModel
    # For now, return a placeholder
    return torch.randn(1, 1, 768, device=device)


def tile_image(image: np.ndarray, tile_size: Tuple[int, int],
               overlap: Tuple[int, int]) -> List[Dict[str, Any]]:
    """
    Divide image into overlapping tiles.

    Args:
        image: Input image array
        tile_size: (height, width) of each tile
        overlap: (vertical_overlap, horizontal_overlap)

    Returns:
        List of tile dictionaries with coordinates and data
    """
    height, width = image.shape[:2]
    tile_h, tile_w = tile_size
    overlap_v, overlap_h = overlap

    tiles = []

    # Calculate step sizes
    step_v = tile_h - overlap_v
    step_h = tile_w - overlap_h

    y = 0
    while y < height:
        x = 0
        while x < width:
            # Calculate tile boundaries
            y_end = min(y + tile_h, height)
            x_end = min(x + tile_w, width)

            # Adjust start if we're at the edge
            y_start = max(0, y_end - tile_h)
            x_start = max(0, x_end - tile_w)

            tile = {
                'data': image[y_start:y_end, x_start:x_end],
                'coords': (y_start, x_start, y_end, x_end),
                'local_horizon_offset': y_start / height
            }

            tiles.append(tile)

            x += step_h

        y += step_v

    return tiles


def blend_tiles(tiles: List[Dict[str, Any]], original_size: Tuple[int, int],
                overlap: Tuple[int, int]) -> np.ndarray:
    """
    Blend overlapping tiles back into a single image.

    Args:
        tiles: List of tile dictionaries
        original_size: (height, width) of output image
        overlap: (vertical_overlap, horizontal_overlap)

    Returns:
        Blended image array
    """
    height, width = original_size
    blended = np.zeros((height, width, 3), dtype=np.float32)
    weight_mask = np.zeros((height, width), dtype=np.float32)

    overlap_v, overlap_h = overlap

    for tile in tiles:
        y_start, x_start, y_end, x_end = tile['coords']
        tile_data = tile['data']

        # Create Gaussian weight mask for blending
        tile_h, tile_w = tile_data.shape[:2]

        # Create coordinate grids
        y_coords = np.arange(tile_h)
        x_coords = np.arange(tile_w)

        # Distance from edges
        y_dist = np.minimum(y_coords, tile_h - 1 - y_coords)
        x_dist = np.minimum(x_coords, tile_w - 1 - x_coords)

        # Gaussian weights based on distance from edges
        sigma = min(overlap_v, overlap_h) / 3.0
        y_weights = np.exp(-0.5 * (y_dist / sigma) ** 2)
        x_weights = np.exp(-0.5 * (x_dist / sigma) ** 2)

        weights = y_weights[:, np.newaxis] * x_weights[np.newaxis, :]

        # Apply weights
        blended[y_start:y_end, x_start:x_end] += tile_data * weights[:, :, np.newaxis]
        weight_mask[y_start:y_end, x_start:x_end] += weights

    # Normalize by weights
    weight_mask = np.maximum(weight_mask, 1e-8)  # Avoid division by zero
    blended /= weight_mask[:, :, np.newaxis]

    return blended.astype(np.uint8)


class MetricsTracker:
    """Track and compute evaluation metrics."""

    def __init__(self):
        self.metrics = {
            'horizon_detection_rate': [],
            'horizon_deviation': [],
            'rule_of_thirds_alignment': [],
            'sharpness': [],
            'clip_score': [],
            'niqe': []
        }

    def add_sample(self, metrics_dict: Dict[str, float]) -> None:
        """Add metrics for a single sample."""
        for key, value in metrics_dict.items():
            if key in self.metrics:
                self.metrics[key].append(value)

    def get_summary(self) -> Dict[str, float]:
        """Get summary statistics for all metrics."""
        summary = {}
        for key, values in self.metrics.items():
            if values:
                summary[f'{key}_mean'] = np.mean(values)
                summary[f'{key}_std'] = np.std(values)
                summary[f'{key}_min'] = np.min(values)
                summary[f'{key}_max'] = np.max(values)
            else:
                summary[f'{key}_mean'] = 0.0

        return summary

    def reset(self) -> None:
        """Reset all metrics."""
        for key in self.metrics:
            self.metrics[key] = []


if __name__ == "__main__":
    # Example usage
    device = setup_device()
    print(f"Device: {device}")

    # Test anchor token creation
    dummy_anchor = np.random.rand(128)
    token = create_anchor_token(dummy_anchor, device)
    print(f"Anchor token shape: {token.shape}")