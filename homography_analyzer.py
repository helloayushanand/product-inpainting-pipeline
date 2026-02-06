"""Extract perspective information from homography matrices."""
import numpy as np
from typing import Tuple


class HomographyAnalyzer:
    """Analyzes homography matrices to extract perspective parameters."""
    
    def extract_perspective_angle(self, homography: np.ndarray) -> float:
        """
        Extract the viewing angle from a homography matrix.
        
        The homography encodes the perspective transformation. We can
        estimate the horizontal viewing angle by analyzing how the
        transformation affects corners.
        
        Args:
            homography: 3x3 homography matrix
        
        Returns:
            Estimated viewing angle in degrees (0 = front view, 45 = angled)
        """
        if homography is None:
            return 0.0
        
        # Define corners of a unit square
        corners = np.array([
            [0, 0, 1],
            [1, 0, 1],
            [1, 1, 1],
            [0, 1, 1]
        ], dtype=np.float32).T  # 3x4
        
        # Transform corners through homography
        transformed = homography @ corners
        
        # Convert from homogeneous coordinates
        transformed = transformed / transformed[2, :]
        
        # Analyze the transformation to estimate viewing angle
        # Look at how left and right edges are transformed
        
        # Original left edge: (0,0) to (0,1)
        # Original right edge: (1,0) to (1,1)
        
        left_top = transformed[:2, 0]     # (0,0) transformed
        left_bottom = transformed[:2, 3]  # (0,1) transformed
        right_top = transformed[:2, 1]    # (1,0) transformed
        right_bottom = transformed[:2, 2] # (1,1) transformed
        
        # Calculate edge lengths after transformation
        left_edge_length = np.linalg.norm(left_bottom - left_top)
        right_edge_length = np.linalg.norm(right_bottom - right_top)
        
        # Calculate width at top and bottom
        top_width = np.linalg.norm(right_top - left_top)
        bottom_width = np.linalg.norm(right_bottom - left_bottom)
        
        # Perspective causes trapezoid: one edge shorter than the other
        # The ratio tells us the viewing angle
        width_ratio = min(top_width, bottom_width) / max(top_width, bottom_width)
        
        # Convert ratio to approximate angle
        # ratio = 1.0 → angle = 0° (no perspective)
        # ratio = 0.5 → angle ≈ 45° (strong perspective)
        # ratio = 0.3 → angle ≈ 60° (very strong)
        
        if width_ratio > 0.95:
            angle = 0.0  # Nearly front view
        elif width_ratio > 0.8:
            angle = 15.0  # Mild perspective
        elif width_ratio > 0.6:
            angle = 30.0  # Moderate perspective
        elif width_ratio > 0.4:
            angle = 45.0  # Strong perspective
        else:
            angle = 60.0  # Very strong perspective
        
        return angle
    
    def analyze_homography(self, homography: np.ndarray) -> dict:
        """
        Comprehensive analysis of homography matrix.
        
        Args:
            homography: 3x3 homography matrix
        
        Returns:
            Dictionary with analysis results
        """
        if homography is None:
            return {
                'viewing_angle': 0.0,
                'has_perspective': False,
                'transform_type': 'none'
            }
        
        angle = self.extract_perspective_angle(homography)
        
        return {
            'viewing_angle': angle,
            'has_perspective': angle > 10.0,
            'transform_type': 'perspective' if angle > 10.0 else 'affine'
        }


def test_homography_analyzer():
    """Test the homography analyzer."""
    print("Testing Homography Analyzer...")
    
    analyzer = HomographyAnalyzer()
    
    # Test 1: Identity matrix (no transformation)
    print("\n1. Identity matrix (front view):")
    identity = np.eye(3, dtype=np.float32)
    angle = analyzer.extract_perspective_angle(identity)
    print(f"   Extracted angle: {angle:.1f}°")
    
    # Test 2: Perspective transformation (trapezoid)
    print("\n2. Perspective transformation:")
    # Simulate a perspective transform
    perspective = np.array([
        [1.0, 0.1, 0],
        [0, 1.2, 0],
        [0.001, 0.002, 1.0]
    ], dtype=np.float32)
    angle = analyzer.extract_perspective_angle(perspective)
    print(f"   Extracted angle: {angle:.1f}°")
    
    # Test 3: Strong perspective
    print("\n3. Strong perspective transformation:")
    strong_perspective = np.array([
        [0.8, 0.2, 0],
        [0, 1.5, 0],
        [0.003, 0.005, 1.0]
    ], dtype=np.float32)
    angle = analyzer.extract_perspective_angle(strong_perspective)
    print(f"   Extracted angle: {angle:.1f}°")
    
    print("\n✓ Homography analyzer test complete!")


if __name__ == "__main__":
    test_homography_analyzer()
