"""Crop-Blend-Stitch compositor for isolated blending."""
import cv2
import numpy as np
from PIL import Image
from typing import Tuple, Optional


class CropBlendStitch:
    """Handles crop-blend-stitch workflow for better isolated blending."""
    
    def __init__(self, feather_amount: int = 30):
        """
        Initialize compositor.
        
        Args:
            feather_amount: Pixels to feather at edges for seamless stitching
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
        remove_bg: bool = True,
        paste_only: bool = False
    ) -> Image.Image:
        """
        Composite product into scene using crop-blend-stitch method.
        
        Workflow:
        1. Crop the bounding box region from scene
        2. Blend product into the cropped region
        3. Stitch the blended region back into the full scene
        
        Args:
            product: Original product image
            scene: Generated scene image
            bbox: Bounding box (x, y, w, h) where product should be placed
            homography: Optional homography matrix for precise transformation
            remove_bg: Whether to remove background from product
            paste_only: If True, skip feathering and just paste with alpha
        
        Returns:
            Final composited image
        """
        x, y, w, h = bbox
        
        # Step 1: Remove background from product if needed
        if remove_bg:
            product = self.remove_background(product)
            # Debug: save background-removed product
            import config
            debug_path = config.OUTPUT_DIR / "debug_product_no_bg.png"
            product.save(debug_path)
            print(f"  Saved bg-removed product to: {debug_path}")
        
        # Step 2: Crop the region from scene
        scene_np = np.array(scene.convert("RGB"))
        cropped_region = scene_np[y:y+h, x:x+w].copy()
        
        print(f"  Cropped region: {cropped_region.shape}")
        
        # Step 3: Blend product into cropped region
        blended_region = self._blend_into_crop(
            product, cropped_region, (w, h), 
            homography=homography, bbox_offset=(x, y),
            paste_only=paste_only
        )
        
        # Step 4: Create feathered mask for seamless stitching
        stitch_mask = self._create_stitch_mask(blended_region.shape[:2])
        
        # Step 5: Stitch back into scene with feathering
        result = scene_np.copy()
        
        # Apply feathered blending at boundaries
        stitch_mask_3d = np.stack([stitch_mask] * 3, axis=2)
        result[y:y+h, x:x+w] = (
            stitch_mask_3d * blended_region +
            (1 - stitch_mask_3d) * result[y:y+h, x:x+w]
        )
        
        return Image.fromarray(result.astype(np.uint8))
    
    def _blend_into_crop(
        self,
        product: Image.Image,
        cropped_region: np.ndarray,
        target_size: Tuple[int, int],
        homography: Optional[np.ndarray] = None,
        bbox_offset: Tuple[int, int] = (0, 0),
        paste_only: bool = False
    ) -> np.ndarray:
        """
        Blend product into the cropped region.
        
        Args:
            product: Product image (RGBA with transparency)
            cropped_region: Cropped scene region (RGB numpy array)
            target_size: (width, height) of the cropped region
            homography: Optional homography matrix for precise transformation
            bbox_offset: (x, y) offset of the bounding box in the full scene
            paste_only: If True, skip feathering and just paste with alpha
        
        Returns:
            Blended region (RGB numpy array)
        """
        w, h = target_size
        cropped_float = cropped_region.astype(np.float32)
        
        # Convert product to numpy
        product_np = np.array(product).astype(np.float32)
        
        if homography is not None:
            # Use homography for precise transformation
            print(f"  Using homography for precise placement")
            
            # Create translation matrix to account for bbox offset
            # We need to translate the homography to work within the cropped region
            offset_x, offset_y = bbox_offset
            T_offset = np.array([
                [1, 0, -offset_x],
                [0, 1, -offset_y],
                [0, 0, 1]
            ], dtype=np.float32)
            
            # Adjust homography for the cropped region
            H_adjusted = T_offset @ homography
            
            # Warp product using adjusted homography
            product_warped = cv2.warpPerspective(
                product_np,
                H_adjusted,
                (w, h),
                flags=cv2.INTER_LANCZOS4
            )
            
            # Extract RGB and alpha if RGBA
            if product_warped.shape[2] == 4:
                product_rgb = product_warped[:, :, :3]
                alpha = product_warped[:, :, 3] / 255.0
            else:
                product_rgb = product_warped
                # Create alpha from warped mask
                gray = cv2.cvtColor(product_rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY)
                alpha = (gray > 0).astype(np.float32)
        
        else:
            # Fallback: simple resize
            print(f"  No homography, using simple resize")
            product_resized = cv2.resize(product_np, (w, h), interpolation=cv2.INTER_LANCZOS4)
            
            # Extract RGB and alpha
            if product_resized.shape[2] == 4:
                product_rgb = product_resized[:, :, :3]
                alpha = product_resized[:, :, 3] / 255.0
            else:
                product_rgb = product_resized
                alpha = np.ones((h, w), dtype=np.float32)
        
        # Apply feathering only if not paste-only mode
        if paste_only:
            final_alpha = alpha
        else:
            final_alpha = self._feather_alpha(alpha, feather_amount=15)
        
        # Blend
        alpha_3d = np.stack([final_alpha] * 3, axis=2)
        
        blended = (
            alpha_3d * product_rgb +
            (1 - alpha_3d) * cropped_float
        )
        
        return blended.astype(np.uint8)
    
    def _feather_alpha(self, alpha: np.ndarray, feather_amount: int = 15) -> np.ndarray:
        """
        Feather the alpha channel for soft edges.
        
        Args:
            alpha: Alpha channel (0-1 float)
            feather_amount: Pixels to feather
        
        Returns:
            Feathered alpha (0-1 float)
        """
        # Convert to binary mask
        mask = (alpha > 0.5).astype(np.uint8)
        
        # Distance transform
        dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        
        # Normalize and clip
        feathered = np.clip(dist / feather_amount, 0, 1)
        
        # Multiply with original alpha to preserve transparency
        return feathered * alpha
    
    def _create_stitch_mask(self, region_shape: Tuple[int, int]) -> np.ndarray:
        """
        Create a feathered mask for stitching the region back.
        This prevents hard edges at the boundary.
        
        Args:
            region_shape: (height, width) of the region
        
        Returns:
            Feathered mask (0-1 float)
        """
        h, w = region_shape
        
        # Start with all ones (full blend in center)
        mask = np.ones((h, w), dtype=np.float32)
        
        # Feather at edges
        feather = self.feather_amount
        
        # Top edge
        for i in range(min(feather, h)):
            mask[i, :] *= (i / feather)
        
        # Bottom edge
        for i in range(min(feather, h)):
            mask[h-1-i, :] *= (i / feather)
        
        # Left edge
        for j in range(min(feather, w)):
            mask[:, j] *= (j / feather)
        
        # Right edge
        for j in range(min(feather, w)):
            mask[:, w-1-j] *= (j / feather)
        
        return mask


if __name__ == "__main__":
    # Test the crop-blend-stitch method
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
    print("Finding product location...")
    matcher = ProductMatcher(method="feature")
    bbox = matcher.find_product_location(product, scene)
    
    if not bbox:
        print("Could not find product")
        exit(1)
    
    print(f"Found product at: {bbox}")
    
    # Composite using crop-blend-stitch
    print("\nCompositing with crop-blend-stitch method...")
    compositor = CropBlendStitch(feather_amount=30)
    result = compositor.composite(product, scene, bbox, remove_bg=True)
    
    output_path = config.OUTPUT_DIR / "crop_blend_stitch.jpg"
    result.save(output_path)
    print(f"Saved to: {output_path}")
