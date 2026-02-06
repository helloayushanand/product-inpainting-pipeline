"""Scene perspective estimation for 3D rendering."""
import cv2
import numpy as np
from PIL import Image
from typing import Tuple, Optional


class PerspectiveEstimator:
    """
    Estimates camera perspective from generated scene images.
    Analyzes table/surface angles to determine viewing angle.
    """
    
    def estimate_perspective(self, scene_image: Image.Image) -> Tuple[float, float]:
        """
        Estimate camera angle and elevation from scene.
        
        Args:
            scene_image: Generated scene image
        
        Returns:
            Tuple of (horizontal_angle, elevation_angle) in degrees
        """
        print(f"  🎯 Estimating scene perspective...")
        
        # Convert to numpy
        scene_np = np.array(scene_image.convert("RGB"))
        
        # Detect table/surface angle
        angle, elevation = self._analyze_surface_perspective(scene_np)
        
        print(f"     Camera angle: {angle:.1f}°, elevation: {elevation:.1f}°")
        
        return angle, elevation
    
    def _analyze_surface_perspective(self, image: np.ndarray) -> Tuple[float, float]:
        """
        Analyze table/surface in image to estimate perspective.
        
        Args:
            image: Scene image as numpy array
        
        Returns:
            Tuple of (horizontal_angle, elevation_angle)
        """
        h, w = image.shape[:2]
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Detect edges (table edges, surface lines)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # Detect lines using Hough Transform
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi/180,
            threshold=100,
            minLineLength=100,
            maxLineGap=10
        )
        
        if lines is None or len(lines) == 0:
            # No significant lines detected - assume front view
            print(f"     No perspective lines detected, assuming front view")
            return 0.0, 15.0
        
        # Analyze line angles
        horizontal_lines = []
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Calculate line angle
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            
            # Focus on nearly horizontal lines (table edges)
            if abs(angle) < 30 or abs(angle) > 150:
                horizontal_lines.append((x1, y1, x2, y2, angle))
        
        if not horizontal_lines:
            print(f"     No horizontal table lines found, assuming front view")
            return 0.0, 15.0
        
        # Estimate perspective from horizontal lines
        # Lines that converge indicate perspective
        angles = [line[4] for line in horizontal_lines]
        avg_angle = np.mean(angles)
        angle_variance = np.var(angles)
        
        # High variance = strong perspective (angled view)
        # Low variance = weak perspective (front view)
        
        if angle_variance > 50:
            # Strong perspective - table at angle
            horizontal_angle = 45.0  # Approximate 45° view
            elevation = 20.0
        elif angle_variance > 10:
            # Moderate perspective
            horizontal_angle = 25.0
            elevation = 15.0
        else:
            # Minimal perspective - front view
            horizontal_angle = 0.0
            elevation = 10.0
        
        print(f"     Line variance: {angle_variance:.1f} → angle estimate: {horizontal_angle:.1f}°")
        
        return horizontal_angle, elevation
    
    def should_use_3d_rendering(
        self,
        horizontal_angle: float,
        product_shape: str
    ) -> bool:
        """
        Decide if 3D rendering is needed based on perspective and shape.
        
        Args:
            horizontal_angle: Estimated camera angle in degrees
            product_shape: Product shape (FLAT, CYLINDER, BOX)
        
        Returns:
            True if 3D rendering should be used
        """
        # Always use 2D for flat products
        if product_shape == "FLAT":
            print(f"  📋 Decision: Use 2D (flat product)")
            return False
        
        # Use 3D for volumetric products with perspective
        if product_shape in ["CYLINDER", "BOX"] and horizontal_angle > 15:
            print(f"  🎨 Decision: Use 3D (volumetric + perspective)")
            return True
        
        # For mild perspective, 2D is often good enough
        print(f"  📋 Decision: Use 2D (mild perspective)")
        return False


def test_perspective_estimator():
    """Test the perspective estimator."""
    print("Testing Perspective Estimator...")
    
    estimator = PerspectiveEstimator()
    
    # Create test image
    test_img = Image.new('RGB', (1024, 1024), color='white')
    
    print("\nTesting with sample image:")
    angle, elevation = estimator.estimate_perspective(test_img)
    
    print("\nTesting decision logic:")
    print("  FLAT product:")
    should_use_3d = estimator.should_use_3d_rendering(angle, "FLAT")
    
    print("  CYLINDER product with angle:")
    should_use_3d = estimator.should_use_3d_rendering(45.0, "CYLINDER")
    
    print("  CYLINDER product front view:")
    should_use_3d = estimator.should_use_3d_rendering(5.0, "CYLINDER")
    
    print("\n✓ Perspective estimator test complete!")


if __name__ == "__main__":
    test_perspective_estimator()
