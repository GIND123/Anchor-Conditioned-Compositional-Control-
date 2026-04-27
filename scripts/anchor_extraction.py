"""
Anchor Extraction Module

Extracts 4D compositional anchor vectors from landscape images.
Based on the methodology described in the Anchor-Conditioned Compositional Control paper.

Anchor vector components:
- horizon_y: Normalized horizon position [0, 1]
- horizon_conf: Horizon detection confidence [0, 1]
- avg_saliency: Average saliency score [0, 1]
- fg_ratio: Foreground ratio estimate [0, 1]
"""

import cv2
import numpy as np
from typing import Tuple, List, Dict, Any
import json
import os


class AnchorExtractor:
    """Extracts compositional anchor vectors from images."""

    def __init__(self):
        self.hough_threshold = 100
        self.min_line_length = 50
        self.max_line_gap = 10

    def extract_anchor(self, image_path: str) -> Dict[str, float]:
        """
        Extract 4D anchor vector from a single image.

        Args:
            image_path: Path to the image file

        Returns:
            Dictionary containing the 4D anchor vector components
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")

        height, width = image.shape[:2]

        # Extract horizon position and confidence
        horizon_y, horizon_conf = self._extract_horizon(image)

        # Extract average saliency
        avg_saliency = self._extract_saliency(image)

        # Extract foreground ratio
        fg_ratio = self._extract_foreground_ratio(image)

        return {
            'horizon_y': horizon_y,
            'horizon_conf': horizon_conf,
            'avg_saliency': avg_saliency,
            'fg_ratio': fg_ratio
        }

    def _extract_horizon(self, image: np.ndarray) -> Tuple[float, float]:
        """
        Extract horizon position using Hough line transform.

        Returns:
            Tuple of (normalized_horizon_y, confidence)
        """
        height, width = image.shape[:2]

        # Convert to grayscale and apply Canny edge detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Focus on upper 60% of image for horizon detection
        upper_region = gray[:int(height * 0.6), :]

        # Apply Canny edge detection
        edges = cv2.Canny(upper_region, 50, 150, apertureSize=3)

        # Apply Hough line transform
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=self.hough_threshold,
                               minLineLength=self.min_line_length, maxLineGap=self.max_line_gap)

        if lines is None or len(lines) == 0:
            # No horizon detected
            return 0.5, 0.0

        # Find the longest horizontal line
        best_line = None
        max_length = 0

        for line in lines:
            x1, y1, x2, y2 = line[0]

            # Check if line is approximately horizontal (slope < 0.1)
            if abs(y2 - y1) / max(abs(x2 - x1), 1) < 0.1:
                length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                if length > max_length:
                    max_length = length
                    best_line = line[0]

        if best_line is None:
            return 0.5, 0.0

        # Calculate horizon position (average y-coordinate)
        x1, y1, x2, y2 = best_line
        horizon_y_local = (y1 + y2) / 2

        # Normalize to full image height
        horizon_y_normalized = horizon_y_local / height

        # Calculate confidence based on line length relative to image width
        confidence = min(max_length / width, 1.0)

        return horizon_y_normalized, confidence

    def _extract_saliency(self, image: np.ndarray) -> float:
        """
        Extract average saliency using spectral residual approach.
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)

        # Compute FFT
        fft = np.fft.fft2(gray)
        fft_shift = np.fft.fftshift(fft)

        # Get magnitude and phase
        magnitude = np.abs(fft_shift)
        phase = np.angle(fft_shift)

        # Compute spectral residual
        log_magnitude = np.log(magnitude + 1e-8)
        mean_log = cv2.blur(log_magnitude, (3, 3))
        spectral_residual = log_magnitude - mean_log

        # Inverse FFT
        residual_fft = np.exp(spectral_residual + 1j * phase)
        saliency_map = np.abs(np.fft.ifft2(np.fft.ifftshift(residual_fft)))

        # Normalize
        saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min() + 1e-8)

        # Return average saliency
        return float(np.mean(saliency_map))

    def _extract_foreground_ratio(self, image: np.ndarray) -> float:
        """
        Estimate foreground ratio using simple heuristics.
        This is a coarse estimate of visually prominent foreground content.
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Use adaptive thresholding to segment foreground
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY_INV, 11, 2)

        # Calculate ratio of foreground pixels
        fg_pixels = np.sum(thresh > 0)
        total_pixels = thresh.size

        return fg_pixels / total_pixels

    def extract_batch(self, image_paths: List[str], cache_path: str = None) -> Dict[str, Dict[str, float]]:
        """
        Extract anchors for a batch of images.

        Args:
            image_paths: List of image file paths
            cache_path: Optional path to save/load cache

        Returns:
            Dictionary mapping image paths to anchor vectors
        """
        if cache_path and os.path.exists(cache_path):
            print(f"Loading cached anchors from {cache_path}")
            with open(cache_path, 'r') as f:
                return json.load(f)

        anchors = {}
        for i, path in enumerate(image_paths):
            if i % 100 == 0:
                print(f"Processing image {i+1}/{len(image_paths)}")

            try:
                anchors[path] = self.extract_anchor(path)
            except Exception as e:
                print(f"Error processing {path}: {e}")
                # Use default anchor for failed images
                anchors[path] = {
                    'horizon_y': 0.5,
                    'horizon_conf': 0.0,
                    'avg_saliency': 0.5,
                    'fg_ratio': 0.5
                }

        if cache_path:
            print(f"Saving anchors to {cache_path}")
            with open(cache_path, 'w') as f:
                json.dump(anchors, f, indent=2)

        return anchors


def fourier_encode_anchor(anchor: Dict[str, float], num_freqs: int = 16) -> np.ndarray:
    """
    Apply Fourier encoding to the 4D anchor vector.

    Args:
        anchor: Dictionary with horizon_y, horizon_conf, avg_saliency, fg_ratio
        num_freqs: Number of frequency bands (default 16)

    Returns:
        Fourier-encoded vector of shape (4 * 2 * num_freqs,)
    """
    values = np.array([
        anchor['horizon_y'],
        anchor['horizon_conf'],
        anchor['avg_saliency'],
        anchor['fg_ratio']
    ])

    # Frequencies from 2^0 to 2^(num_freqs-1)
    frequencies = 2 ** np.arange(num_freqs)

    # Apply sin and cos transformations
    encoded = []
    for freq in frequencies:
        encoded.extend(np.sin(values * freq))
        encoded.extend(np.cos(values * freq))

    return np.array(encoded)


if __name__ == "__main__":
    # Example usage
    extractor = AnchorExtractor()

    # Test on a single image (you would replace with actual image path)
    # anchor = extractor.extract_anchor("path/to/image.jpg")
    # print("Anchor vector:", anchor)

    # Fourier encoding example
    sample_anchor = {
        'horizon_y': 0.333,
        'horizon_conf': 0.9,
        'avg_saliency': 0.7,
        'fg_ratio': 0.4
    }
    encoded = fourier_encode_anchor(sample_anchor)
    print(f"Fourier encoded shape: {encoded.shape}")  # Should be (128,) for 16 freqs