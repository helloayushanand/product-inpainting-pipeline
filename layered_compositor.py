"""Multi-resolution layered compositor for product placement."""
import cv2
import numpy as np
from PIL import Image
from typing import Tuple, Optional


class LayeredCompositor:
    """
    Compositor using multi-resolution layering approach.
    
    Strategy:
    1. Create intermediate canvas at 2x target resolution
    2. Place product at full resolution on canvas
    3. Composite canvas onto scene with proper scaling
    4. No upscaling of product = no quality loss
    """
    
    def __init__(self, feather_amount: int = 30):
        """
        Initialize compositor.
        
        Args:
            feather_amount: Pixels to feather at boundaries
        """
        self.feather_amount = feather_amount
    
    def remove_background(self, image: Image.Image, threshold: int = 245) -> Image.Image:
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
        
        # Close small holes
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # Erode slightly to remove white halo
        mask = cv2.erode(mask, kernel, iterations=1)
        
        # Open to remove noise
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Create RGBA image
        img_rgba = cv2.cvtColor(img_np, cv2.COLOR_RGB2RGBA)
        img_rgba[:, :, 3] = mask
        
        return Image.fromarray(img_rgba)
    
    def composite(
        self,
        product: Image.Image,
        scene: Image.Image,
        bbox: Tuple[int, int, int, int],
        homography: Optional[np.ndarray] = None,
        remove_bg: bool = True
    ) -> Image.Image:
        """
        Composite product into scene using dynamic multi-resolution layering.
        
        Strategy:
        - If bbox > product: Create 2x product canvas, place product, upscale to bbox
        - If bbox < product: Use product-sized canvas, place product, downscale to bbox
        - If bbox ≈ product: Direct placement with minimal scaling
        
        Args:
            product: Original product image
            scene: Generated scene image
            bbox: Bounding box (x, y, w, h) in scene coordinates
            homography: Homography matrix for transformation
            remove_bg: Whether to remove background from product
        
        Returns:
            Final composited image
        """
        x, y, w, h = bbox
        scene_np = np.array(scene.convert("RGB"))
        
        # Step 1: Remove background if needed
        if remove_bg:
            product = self.remove_background(product)
            print(f"  Background removed")
        
        product_np = np.array(product).astype(np.float32)
        product_h, product_w = product_np.shape[:2]
        
        # Step 2: Determine canvas strategy based on bbox vs product size
        bbox_max = max(w, h)
        product_max = max(product_w, product_h)
        scale_ratio = bbox_max / product_max
        
        print(f"  Product: {product_w}x{product_h}, Bbox: {w}x{h}, Ratio: {scale_ratio:.2f}x")
        
        if scale_ratio > 1.2:
            # Need significant upscaling - use intermediate canvas
            canvas_scale = 2.0  # Create 2x product canvas
            canvas_w = int(product_w * canvas_scale)
            canvas_h = int(product_h * canvas_scale)
            strategy = "upscale"
            print(f"  Strategy: UPSCALE via {canvas_w}x{canvas_h} canvas (2x product)")
        elif scale_ratio < 0.8:
            # Need downscaling - use product-sized canvas
            canvas_w = product_w
            canvas_h = product_h
            strategy = "downscale"
            print(f"  Strategy: DOWNSCALE from {canvas_w}x{canvas_h} canvas")
        else:
            # Roughly same size - minimal scaling
            canvas_w = product_w
            canvas_h = product_h
            strategy = "direct"
            print(f"  Strategy: DIRECT placement (minimal scaling)")
        
        # Step 3: Create canvas and place product
        canvas = np.zeros((canvas_h, canvas_w, 4), dtype=np.float32)
        
        if homography is not None:
            # Calculate transformation for canvas space
            # We need to map: product -> canvas -> bbox
            
            # Scale factor from canvas to bbox
            canvas_to_bbox_scale_x = w / canvas_w
            canvas_to_bbox_scale_y = h / canvas_h
            
            # Create scaling matrix: canvas -> bbox
            S_canvas_to_bbox = np.array([
                [canvas_to_bbox_scale_x, 0, 0],
                [0, canvas_to_bbox_scale_y, 0],
                [0, 0, 1]
            ], dtype=np.float32)
            
            # Create translation matrix: bbox -> scene
            T_bbox_to_scene = np.array([
                [1, 0, x],
                [0, 1, y],
                [0, 0, 1]
            ], dtype=np.float32)
            
            # The homography maps: product -> scene
            # We want: product -> canvas
            # So: H_canvas = (T * S)^-1 * H_scene
            
            transform_canvas_to_scene = T_bbox_to_scene @ S_canvas_to_bbox
            H_canvas = np.linalg.inv(transform_canvas_to_scene) @ homography
            
            # Warp product onto canvas
            product_warped = cv2.warpPerspective(
                product_np,
                H_canvas,
                (canvas_w, canvas_h),
                flags=cv2.INTER_LINEAR
            )
            
            canvas = product_warped
            print(f"  Applied homography on canvas")
        
        else:
            # Simple center placement on canvas
            offset_x = (canvas_w - product_w) // 2
            offset_y = (canvas_h - product_h) // 2
            
            canvas[offset_y:offset_y+product_h, offset_x:offset_x+product_w] = product_np
            print(f"  Placed product at canvas center")
        
        # Step 4: Scale canvas to bbox size
        if strategy == "upscale":
            # Upscale canvas to bbox
            canvas_scaled = cv2.resize(
                canvas,
                (w, h),
                interpolation=cv2.INTER_LANCZOS4  # High quality upscaling
            )
            print(f"  Upscaled canvas {canvas_w}x{canvas_h} -> {w}x{h}")
        elif strategy == "downscale":
            # Downscale canvas to bbox
            canvas_scaled = cv2.resize(
                canvas,
                (w, h),
                interpolation=cv2.INTER_AREA  # Best for downscaling
            )
            print(f"  Downscaled canvas {canvas_w}x{canvas_h} -> {w}x{h}")
        else:
            # Minimal scaling
            canvas_scaled = cv2.resize(
                canvas,
                (w, h),
                interpolation=cv2.INTER_LINEAR
            )
            print(f"  Scaled canvas {canvas_w}x{canvas_h} -> {w}x{h}")
        
        # Step 5: Extract RGB and alpha
        if canvas_scaled.shape[2] == 4:
            canvas_rgb = canvas_scaled[:, :, :3]
            alpha = canvas_scaled[:, :, 3] / 255.0
        else:
            canvas_rgb = canvas_scaled
            alpha = np.ones((h, w), dtype=np.float32)
        
        # Step 6: Feather alpha for smooth edges
        alpha_feathered = self._feather_alpha(alpha, self.feather_amount)
        
        # Step 7: Composite onto scene
        scene_region = scene_np[y:y+h, x:x+w].astype(np.float32)
        alpha_3d = np.stack([alpha_feathered] * 3, axis=2)
        
        blended = (
            alpha_3d * canvas_rgb +
            (1 - alpha_3d) * scene_region
        )
        
        # Step 8: Place back into scene
        result = scene_np.copy()
        result[y:y+h, x:x+w] = blended.astype(np.uint8)
        
        return Image.fromarray(result.astype(np.uint8))
    
    def _feather_alpha(self, alpha: np.ndarray, amount: int) -> np.ndarray:
        """
        Apply feathering to alpha channel.
        
        Args:
            alpha: Alpha channel (0-1 float)
            amount: Feather amount in pixels
        
        Returns:
            Feathered alpha channel
        """
        if amount <= 0:
            return alpha
        
        # Convert to uint8 for distance transform
        alpha_uint8 = (alpha * 255).astype(np.uint8)
        
        # Distance transform from edges
        dist = cv2.distanceTransform(alpha_uint8, cv2.DIST_L2, 5)
        
        # Normalize to feather amount
        feathered = np.clip(dist / amount, 0, 1)
        
        return feathered.astype(np.float32)
