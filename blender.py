"""Advanced blending techniques for seamless product compositing."""
import cv2
import numpy as np
from PIL import Image
from typing import Tuple, Optional


class Blender:
    """Handles various blending techniques for product compositing."""
    
    def __init__(self, method: str = "feathered", remove_bg: bool = True):
        """
        Initialize blender.
        
        Args:
            method: Blending method - "simple", "feathered", "poisson", "multiband"
            remove_bg: Whether to automatically remove background from product
        """
        self.method = method
        self.remove_bg = remove_bg
    
    def remove_background(self, image: Image.Image, threshold: int = 240) -> Image.Image:
        """
        Remove white/light background from product image.
        
        Args:
            image: Product image with background
            threshold: Brightness threshold for background (0-255)
        
        Returns:
            Image with transparent background (RGBA)
        """
        # Convert to numpy
        img_np = np.array(image.convert("RGB"))
        
        # Create mask: anything brighter than threshold = background
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
        
        # Clean up mask with morphological operations
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Create RGBA image
        img_rgba = cv2.cvtColor(img_np, cv2.COLOR_RGB2RGBA)
        img_rgba[:, :, 3] = mask
        
        return Image.fromarray(img_rgba)
    
    def blend(
        self,
        product: Image.Image,
        scene: Image.Image,
        bbox: Tuple[int, int, int, int],
        homography: Optional[np.ndarray] = None
    ) -> Image.Image:
        """
        Blend product into scene using selected method.
        
        Args:
            product: Original product image
            scene: Generated scene image
            bbox: Bounding box (x, y, w, h) where product should be placed
            homography: Optional homography matrix for perspective warping
        
        Returns:
            Blended image
        """
        # Remove background if enabled
        if self.remove_bg:
            product = self.remove_background(product)
        
        if self.method == "simple":
            return self._simple_blend(product, scene, bbox)
        elif self.method == "feathered":
            return self._feathered_blend(product, scene, bbox, homography)
        elif self.method == "poisson":
            return self._poisson_blend(product, scene, bbox, homography)
        elif self.method == "multiband":
            return self._multiband_blend(product, scene, bbox, homography)
        else:
            raise ValueError(f"Unknown blending method: {self.method}")
    
    def _simple_blend(
        self,
        product: Image.Image,
        scene: Image.Image,
        bbox: Tuple[int, int, int, int]
    ) -> Image.Image:
        """Simple alpha blending (current implementation)."""
        x, y, w, h = bbox
        
        # Convert to numpy
        scene_np = np.array(scene.convert("RGB"))
        product_np = np.array(product.convert("RGBA"))
        
        # Resize product to bbox
        product_resized = cv2.resize(product_np, (w, h), interpolation=cv2.INTER_LANCZOS4)
        
        # Extract alpha or create mask
        if product_resized.shape[2] == 4:
            alpha = product_resized[:, :, 3] / 255.0
            product_rgb = product_resized[:, :, :3]
        else:
            product_rgb = product_resized
            gray = cv2.cvtColor(product_rgb, cv2.COLOR_RGB2GRAY)
            _, alpha = cv2.threshold(gray, 240, 1.0, cv2.THRESH_BINARY_INV)
        
        # Ensure bounds
        y_end = min(y + h, scene_np.shape[0])
        x_end = min(x + w, scene_np.shape[1])
        actual_h = y_end - y
        actual_w = x_end - x
        
        product_rgb = product_rgb[:actual_h, :actual_w]
        alpha = alpha[:actual_h, :actual_w]
        
        # Blend
        alpha_3d = np.stack([alpha] * 3, axis=2)
        scene_np[y:y_end, x:x_end] = (
            alpha_3d * product_rgb + 
            (1 - alpha_3d) * scene_np[y:y_end, x:x_end]
        )
        
        return Image.fromarray(scene_np.astype(np.uint8))
    
    def _feathered_blend(
        self,
        product: Image.Image,
        scene: Image.Image,
        bbox: Tuple[int, int, int, int],
        homography: Optional[np.ndarray] = None
    ) -> Image.Image:
        """
        Feathered edge blending with optional perspective warping.
        Creates soft edges while keeping center sharp.
        """
        x, y, w, h = bbox
        
        # Convert to numpy
        scene_np = np.array(scene.convert("RGB")).astype(np.float32)
        
        # Handle RGBA (with alpha) or RGB product
        if product.mode == 'RGBA':
            product_np = np.array(product).astype(np.float32)
            has_alpha = True
        else:
            product_np = np.array(product.convert("RGB")).astype(np.float32)
            has_alpha = False
        
        # Resize product to bbox
        product_resized = cv2.resize(product_np, (w, h), interpolation=cv2.INTER_LANCZOS4)
        
        # Extract RGB and alpha
        if has_alpha:
            product_rgb = product_resized[:, :, :3]
            alpha_channel = product_resized[:, :, 3] / 255.0
        else:
            product_rgb = product_resized
            alpha_channel = np.ones((h, w), dtype=np.float32)
        
        # Create feathered mask from alpha channel
        feather_mask = self._create_feathered_mask(alpha_channel, feather_amount=20)
        
        # Ensure bounds
        y_end = min(y + h, scene_np.shape[0])
        x_end = min(x + w, scene_np.shape[1])
        actual_h = y_end - y
        actual_w = x_end - x
        
        product_rgb = product_rgb[:actual_h, :actual_w]
        feather_mask = feather_mask[:actual_h, :actual_w]
        
        # Apply feathered blending
        feather_mask_3d = np.stack([feather_mask] * 3, axis=2)
        scene_np[y:y_end, x:x_end] = (
            feather_mask_3d * product_rgb + 
            (1 - feather_mask_3d) * scene_np[y:y_end, x:x_end]
        )
        
        return Image.fromarray(scene_np.astype(np.uint8))
    
    def _create_feathered_mask(self, mask: np.ndarray, feather_amount: int = 20) -> np.ndarray:
        """
        Create a feathered mask with soft edges.
        
        Args:
            mask: Binary mask
            feather_amount: Pixels to feather from edge
        
        Returns:
            Feathered mask (0-1 float)
        """
        # Distance transform from edges
        dist_transform = cv2.distanceTransform(
            (mask > 0.5).astype(np.uint8), 
            cv2.DIST_L2, 
            5
        )
        
        # Normalize to 0-1 based on feather amount
        feathered = np.clip(dist_transform / feather_amount, 0, 1)
        
        return feathered.astype(np.float32)
    
    def _poisson_blend(
        self,
        product: Image.Image,
        scene: Image.Image,
        bbox: Tuple[int, int, int, int],
        homography: Optional[np.ndarray] = None
    ) -> Image.Image:
        """
        Poisson blending (seamless cloning).
        Best for natural lighting integration.
        """
        x, y, w, h = bbox
        
        # Convert to numpy
        scene_np = np.array(scene.convert("RGB"))
        product_np = np.array(product.convert("RGB"))
        
        # Resize product
        product_resized = cv2.resize(product_np, (w, h), interpolation=cv2.INTER_LANCZOS4)
        
        # Create mask
        mask = np.ones((h, w), dtype=np.uint8) * 255
        
        # Calculate center point
        center = (x + w // 2, y + h // 2)
        
        # Ensure bounds
        if x < 0 or y < 0 or x + w > scene_np.shape[1] or y + h > scene_np.shape[0]:
            # Fall back to feathered blend if out of bounds
            print("  Poisson blend out of bounds, falling back to feathered")
            return self._feathered_blend(product, scene, bbox, homography)
        
        # Apply Poisson blending
        try:
            result = cv2.seamlessClone(
                product_resized,
                scene_np,
                mask,
                center,
                cv2.NORMAL_CLONE
            )
            return Image.fromarray(result)
        except cv2.error as e:
            print(f"  Poisson blend failed: {e}, falling back to feathered")
            return self._feathered_blend(product, scene, bbox, homography)
    
    def _multiband_blend(
        self,
        product: Image.Image,
        scene: Image.Image,
        bbox: Tuple[int, int, int, int],
        homography: Optional[np.ndarray] = None
    ) -> Image.Image:
        """
        Multi-band (Laplacian pyramid) blending.
        Best quality but slower.
        """
        # For now, use feathered blend as placeholder
        # Full Laplacian pyramid is complex - can implement if needed
        print("  Multi-band blending not fully implemented, using feathered")
        return self._feathered_blend(product, scene, bbox, homography)


if __name__ == "__main__":
    # Test different blending methods
    import config
    from product_matcher import ProductMatcher
    
    product_path = "/Users/ayush/Downloads/390432_media_swatch_0_17-12-25-08-45-19.jpeg"
    scene_path = config.OUTPUT_DIR / "1_generated_scene.jpg"
    
    if not scene_path.exists():
        print("Please run pipeline.py first to generate a scene")
        exit(1)
    
    product = Image.open(product_path)
    scene = Image.open(scene_path)
    
    # Find product location
    matcher = ProductMatcher(method="feature")
    bbox = matcher.find_product_location(product, scene)
    
    if not bbox:
        print("Could not find product")
        exit(1)
    
    print(f"Found product at: {bbox}")
    
    # Test each blending method
    methods = ["simple", "feathered", "poisson"]
    
    for method in methods:
        print(f"\nTesting {method} blend...")
        blender = Blender(method=method)
        result = blender.blend(product, scene, bbox)
        output_path = config.OUTPUT_DIR / f"blend_{method}.jpg"
        result.save(output_path)
        print(f"  Saved to {output_path}")
