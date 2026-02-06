"""Pseudo-3D product renderer using OpenCV perspective transforms.

This approach simulates 3D rendering without requiring OpenGL/pyrender.
Works reliably on macOS without dependencies issues.
"""
import cv2
import numpy as np
from PIL import Image
from typing import Tuple


class Pseudo3DRenderer:
    """
    Simulates 3D product rendering using 2D perspective transforms.
    Less realistic than true 3D but works immediately on all platforms.
    """
    
    def __init__(self):
        """Initialize the pseudo-3D renderer."""
        pass
    
    def render_cylinder_view(
        self,
        product_image: Image.Image,
        viewing_angle: float = 0.0
    ) -> Image.Image:
        """
        Simulate cylindrical product from an angle using perspective warping.
        
        Args:
            product_image: Flat product image
            viewing_angle: Viewing angle in degrees (0=front, 45=angle, 90=side)
        
        Returns:
            Warped product image simulating 3D cylinder view
        """
        print(f"  🎨 Applying pseudo-3D cylinder transform: angle={viewing_angle}°")
        
        # Convert to numpy
        img_np = np.array(product_image)
        h, w = img_np.shape[:2]
        
        # For mild angles (< 30°), use standard 2D homography
        if abs(viewing_angle) < 30:
            print(f"     Using 2D perspective (mild angle)")
            return product_image
        
        # For stronger angles, apply radial distortion to simulate cylinder
        # This makes the product appear to wrap around a cylindrical surface
        
        # Create a slight perspective tilt
        angle_factor = min(abs(viewing_angle) / 90.0, 0.6)  # Cap at 60% warp
        
        # Define source points (original rectangle)
        src_pts = np.float32([
            [0, 0],
            [w, 0],
            [w, h],
            [0, h]
        ])
        
        # Define destination points (perspective-warped trapezoid)
        # For cylinder, the far side should be narrower
        width_reduction = int(w * angle_factor * 0.3)  # Reduce width on far side
        
        if viewing_angle > 0:
            # Viewing from right, compress right side
            dst_pts = np.float32([
                [0, 0],
                [w - width_reduction, 0],
                [w - width_reduction, h],
                [0, h]
            ])
        else:
            # Viewing from left, compress left side
            dst_pts = np.float32([
                [width_reduction, 0],
                [w, 0],
                [w, h],
                [width_reduction, h]
            ])
        
        # Apply perspective transform
        matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped = cv2.warpPerspective(
            img_np,
            matrix,
            (w, h),
            flags=cv2.INTER_LANCZOS4,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0) if img_np.shape[2] == 4 else (255, 255, 255)
        )
        
        print(f"     Applied perspective warp (reduction: {width_reduction}px)")
        
        return Image.fromarray(warped)
    
    def render_box_view(
        self,
        product_image: Image.Image,
        viewing_angle: float = 0.0
    ) -> Image.Image:
        """
        Simulate box-shaped product from an angle.
        
        Args:
            product_image: Flat product image
            viewing_angle: Viewing angle in degrees
        
        Returns:
            Warped product image simulating 3D box view
        """
        print(f"  📦 Applying pseudo-3D box transform: angle={viewing_angle}°")
        
        img_np = np.array(product_image)
        h, w = img_np.shape[:2]
        
        # For mild angles, use standard 2D
        if abs(viewing_angle) < 20:
            return product_image
        
        # Apply perspective for box (similar to cylinder but more aggressive)
        angle_factor = min(abs(viewing_angle) / 90.0, 0.7)
        
        src_pts = np.float32([
            [0, 0],
            [w, 0],
            [w, h],
            [0, h]
        ])
        
        # Create perspective effect
        width_reduction = int(w * angle_factor * 0.4)
        height_reduction = int(h * angle_factor * 0.1)  # Slight vertical perspective
        
        if viewing_angle > 0:
            dst_pts = np.float32([
                [0, height_reduction],
                [w - width_reduction, 0],
                [w - width_reduction, h],
                [0, h - height_reduction]
            ])
        else:
            dst_pts = np.float32([
                [width_reduction, 0],
                [w, height_reduction],
                [w, h - height_reduction],
                [width_reduction, h]
            ])
        
        matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped = cv2.warpPerspective(
            img_np,
            matrix,
            (w, h),
            flags=cv2.INTER_LANCZOS4,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0) if img_np.shape[2] == 4 else (255, 255, 255)
        )
        
        return Image.fromarray(warped)


def test_pseudo3d():
    """Test the pseudo-3D renderer."""
    print("Testing Pseudo-3D Renderer...")
    
    # Create test image
    test_img = Image.new('RGB', (400, 800), color='lightblue')
    
    renderer = Pseudo3DRenderer()
    
    print("\n1. Cylinder - Front view (0°):")
    front = renderer.render_cylinder_view(test_img, viewing_angle=0)
    
    print("\n2. Cylinder - Angled view (45°):")
    angled = renderer.render_cylinder_view(test_img, viewing_angle=45)
    
    print("\n3. Box - Angled view (30°):")
    box = renderer.render_box_view(test_img, viewing_angle=30)
    
    print("\n✓ Pseudo-3D renderer test complete!")


if __name__ == "__main__":
    test_pseudo3d()
