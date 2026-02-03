"""Composite original product onto generated scene."""
import cv2
import numpy as np
from PIL import Image
from typing import Tuple

class Compositor:
    def paste_product(
        self,
        original_product: Image.Image,
        generated_scene: Image.Image,
        bbox: Tuple[int, int, int, int]
    ) -> Image.Image:
        """
        Paste the original product onto the generated scene at the detected location.
        
        Args:
            original_product: Original product image with sharp text
            generated_scene: Generated scene with blurry product
            bbox: Bounding box (x, y, w, h) where to paste
        
        Returns:
            Composite image (no blending yet, just simple paste)
        """
        x, y, w, h = bbox
        
        # Convert to numpy for processing
        scene_np = np.array(generated_scene.convert("RGB"))
        product_np = np.array(original_product.convert("RGBA"))
        
        # Resize product to match bounding box
        product_resized = cv2.resize(product_np, (w, h), interpolation=cv2.INTER_LANCZOS4)
        
        # Extract alpha channel if available
        if product_resized.shape[2] == 4:
            alpha = product_resized[:, :, 3] / 255.0
            product_rgb = product_resized[:, :, :3]
        else:
            # No alpha channel, create a simple mask (assume white background)
            product_rgb = product_resized
            # Simple threshold-based mask (you can improve this)
            gray = cv2.cvtColor(product_rgb, cv2.COLOR_RGB2GRAY)
            _, alpha = cv2.threshold(gray, 240, 1.0, cv2.THRESH_BINARY_INV)
        
        # Ensure we don't go out of bounds
        y_end = min(y + h, scene_np.shape[0])
        x_end = min(x + w, scene_np.shape[1])
        
        actual_h = y_end - y
        actual_w = x_end - x
        
        # Crop if needed
        product_rgb = product_rgb[:actual_h, :actual_w]
        alpha = alpha[:actual_h, :actual_w]
        
        # Simple paste (no blending)
        for c in range(3):
            scene_np[y:y_end, x:x_end, c] = (
                alpha * product_rgb[:, :, c] + 
                (1 - alpha) * scene_np[y:y_end, x:x_end, c]
            )
        
        return Image.fromarray(scene_np.astype(np.uint8))

if __name__ == "__main__":
    # Test compositor
    import config
    
    compositor = Compositor()
    
    product_path = config.INPUT_DIR / "product.png"
    scene_path = config.OUTPUT_DIR / "generated_scene.jpg"
    
    if product_path.exists() and scene_path.exists():
        from product_matcher import ProductMatcher
        
        product = Image.open(product_path)
        scene = Image.open(scene_path)
        
        # Find location
        matcher = ProductMatcher(method="template")
        bbox = matcher.find_product_location(product, scene)
        
        if bbox:
            # Composite
            result = compositor.paste_product(product, scene, bbox)
            result.save(config.OUTPUT_DIR / "composited.jpg")
            print(f"Saved composite to {config.OUTPUT_DIR / 'composited.jpg'}")
        else:
            print("Could not find product location")
    else:
        print("Please run gemini_generator.py first")
