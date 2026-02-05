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
    
    
    def transfer_lighting(
        self,
        product: np.ndarray,
        scene_region: np.ndarray,
        alpha: np.ndarray
    ) -> np.ndarray:
        """
        Transfer lighting characteristics from scene region to product.
        
        Samples color temperature, brightness, and contrast from the scene
        region where the product will be placed, then applies those
        characteristics to the product.
        
        Args:
            product: Product RGB (float32, 0-255)
            scene_region: Scene region RGB (float32, 0-255)
            alpha: Product alpha mask (float32, 0-1)
        
        Returns:
            Product with adjusted lighting (float32, 0-255)
        """
        # Only analyze non-transparent product pixels
        product_mask = alpha > 0.5
        
        if not np.any(product_mask):
            return product
        
        # Get product pixels (only non-transparent)
        product_pixels = product[product_mask]
        
        # Sample scene region (use all pixels)
        scene_pixels = scene_region.reshape(-1, 3)
        
        # Calculate mean and std for each channel
        product_mean = np.mean(product_pixels, axis=0)
        product_std = np.std(product_pixels, axis=0)
        
        scene_mean = np.mean(scene_pixels, axis=0)
        scene_std = np.std(scene_pixels, axis=0)
        
        print(f"  Product mean RGB: {product_mean.astype(int)}")
        print(f"  Scene mean RGB: {scene_mean.astype(int)}")
        
        # Apply color transfer (Reinhard color transfer)
        # Normalize product to zero mean, unit variance
        product_adjusted = product.copy()
        
        for c in range(3):
            if product_std[c] > 0:
                # Normalize
                product_adjusted[:, :, c] = (product[:, :, c] - product_mean[c]) / product_std[c]
                # Scale to scene statistics
                product_adjusted[:, :, c] = product_adjusted[:, :, c] * scene_std[c] + scene_mean[c]
        
        # Clip to valid range
        product_adjusted = np.clip(product_adjusted, 0, 255)
        
        # Blend adjusted product with original based on alpha
        # Keep original in transparent areas
        alpha_3d = np.stack([alpha] * 3, axis=2)
        product_final = alpha_3d * product_adjusted + (1 - alpha_3d) * product
        
        return product_final
    
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
            # User requested to disable automatic removal for now
            # product = self.remove_background(product)
            print(f"  Background removal SKIPPED (user request)")
        
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
        
        # Step 6: Transfer lighting from scene region to product
        scene_region = scene_np[y:y+h, x:x+w].astype(np.float32)
        canvas_rgb_lit = self.transfer_lighting(canvas_rgb, scene_region, alpha)
        print(f"  Applied lighting transfer from scene region")
        
        # Step 7: Feather alpha for smooth edges
        # Calculate adaptive feathering map based on scene context
        print(f"  Calculating adaptive feathering map (contour-based)...")
        feather_map = self._calculate_adaptive_feather_map(alpha, scene_np, bbox)
        
        # Apply feathering using the map
        alpha_feathered = self._feather_alpha(alpha, feather_map)
        
        # Step 8: Composite onto scene
        alpha_3d = np.stack([alpha_feathered] * 3, axis=2)
        
        blended = (
            alpha_3d * canvas_rgb_lit +
            (1 - alpha_3d) * scene_region
        )
        
        # Step 9: Place back into scene
        result = scene_np.copy()
        result[y:y+h, x:x+w] = blended.astype(np.uint8)
        
        return Image.fromarray(result.astype(np.uint8))
    
    def _calculate_adaptive_feather_map(
        self,
        alpha: np.ndarray,
        scene: np.ndarray,
        bbox: Tuple[int, int, int, int]
    ) -> np.ndarray:
        """
        Create a spatially varying feather map based on scene texture.
        
        Logic:
        1. Find contour of the product.
        2. Walk the contour: at each point, check variance of underlying scene.
           - High Variance (sharp background) -> Low Feather (5px)
           - Low Variance (blurry background) -> High Feather (50px)
        3. Propagate these edge values inward using inpainting/nearest neighbor.
        """
        x, y, w, h = bbox
        
        # 1. Work at lower resolution for performance (e.g. max 512px dim)
        scale = 512.0 / max(w, h)
        if scale >= 1.0:
            scale = 1.0
            
        small_w = int(w * scale)
        small_h = int(h * scale)
        
        # Extract scene region corresponding to bbox
        scene_roi = scene[y:y+h, x:x+w]
        
        # Resize inputs
        alpha_small = cv2.resize(alpha, (small_w, small_h), interpolation=cv2.INTER_NEAREST)
        scene_small = cv2.resize(scene_roi, (small_w, small_h), interpolation=cv2.INTER_AREA)
        
        # Convert scene to gray for variance calc
        if len(scene_small.shape) == 3:
            scene_gray = cv2.cvtColor(scene_small, cv2.COLOR_RGB2GRAY)
        else:
            scene_gray = scene_small
            
        # 2. Find contours
        # Ensure binary
        _, alpha_bin = cv2.threshold(alpha_small, 0.5, 1.0, cv2.THRESH_BINARY)
        alpha_u8 = (alpha_bin * 255).astype(np.uint8)
        
        contours, _ = cv2.findContours(alpha_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        # Initialize sparse map (0 = unknown)
        # We'll use a mask for inpainting: 0=known, 255=unknown to fill
        # But cv2.inpaint expects the mask to be non-zero at pixels TO BE INPAINTED.
        # So: Known pixels = 0 in mask. Unknown = 255.
        
        feather_sparse = np.zeros((small_h, small_w), dtype=np.float32)
        inpaint_mask = np.ones((small_h, small_w), dtype=np.uint8) * 255
        
        min_feather = 5.0
        max_feather = 50.0
        
        # 3. Walk contours
        if contours:
            # Flatten all contour points
            points = np.vstack(contours).squeeze()
            if len(points.shape) == 1: # Handle single point case
                points = np.array([points])
                
            for pt in points:
                px, py = pt
                
                # Check valid bounds
                if px < 0 or px >= small_w or py < 0 or py >= small_h:
                    continue
                
                # Sample window size (e.g. 20px at full res -> scaled down)
                # Ensure at least 3x3 window
                win_r = max(2, int(15 * scale))
                
                x1 = max(0, px - win_r)
                y1 = max(0, py - win_r)
                x2 = min(small_w, px + win_r + 1)
                y2 = min(small_h, py + win_r + 1)
                
                patch = scene_gray[y1:y2, x1:x2]
                
                if patch.size == 0:
                    val = max_feather
                else:
                    # Calculate Laplacian variance (sharpness)
                    variance = cv2.Laplacian(patch, cv2.CV_64F).var()
                    
                    # Map variance to feather amount
                    # High var (>500) -> Min feather
                    # Low var (<50) -> Max feather
                    # Log mapping might be better
                    
                    if variance > 500:
                        val = min_feather
                    elif variance < 50:
                        val = max_feather
                    else:
                        # Linear interp between 50 and 500
                        t = (variance - 50) / 450.0 # 0 to 1
                        val = max_feather - t * (max_feather - min_feather)
                        
                feather_sparse[py, px] = val
                inpaint_mask[py, px] = 0 # Mark as known
                
        # 4. Inpaint to fill the interior/exterior
        # Linear inpainting is fast and creates smooth gradients
        feather_map_small = cv2.inpaint(
            feather_sparse, inpaint_mask, 3, cv2.INPAINT_TELEA
        )
        
        # 5. Upscale back to full resolution
        feather_map = cv2.resize(feather_map_small, (w, h), interpolation=cv2.INTER_LINEAR)
        
        return feather_map

    def _feather_alpha(self, alpha: np.ndarray, amount: np.ndarray) -> np.ndarray:
        """
        Apply feathering to alpha channel.
        
        Args:
            alpha: Alpha channel (0-1 float)
            amount: Feather amount map (float array same size as alpha) OR scalar
        
        Returns:
            Feathered alpha channel
        """
        # Convert to uint8 for distance transform
        alpha_uint8 = (alpha * 255).astype(np.uint8)
        
        # Distance transform from edges (calculates distance to nearest zero pixel)
        dist = cv2.distanceTransform(alpha_uint8, cv2.DIST_L2, 5)
        
        # Max sure amount is not zero to avoid div by zero
        safe_amount = np.maximum(amount, 1e-5)
        
        # Normalize to feather amount
        feathered = np.clip(dist / safe_amount, 0, 1)
        
        return feathered.astype(np.float32)
