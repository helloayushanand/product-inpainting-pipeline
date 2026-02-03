"""Main pipeline orchestrator."""
from pathlib import Path
from PIL import Image

import config
from gemini_generator import GeminiGenerator
from product_matcher import ProductMatcher
from compositor import Compositor


# ============================================
# CONFIGURATION - Edit these values
# ============================================
PRODUCT_IMAGE_PATH = "/Users/ayush/Downloads/390432_media_swatch_0_17-12-25-08-45-19.jpeg"  # Path to your product image
SCENE_PROMPT = "place this on a table near the window. it must look like a real life lifestyle shot of this product."  # Scene description
MATCHING_METHOD = "feature"  # "template" or "feature"
OUTPUT_PATH = None  # None = auto (output/3_final_composite.jpg), or specify custom path
# ============================================


class Pipeline:
    def __init__(self, matching_method: str = "template"):
        """
        Initialize the pipeline.
        
        Args:
            matching_method: "template" or "feature"
        """
        self.generator = GeminiGenerator()
        self.matcher = ProductMatcher(method=matching_method)
        self.compositor = Compositor()
    
    def run(
        self, 
        product_image_path: str, 
        scene_prompt: str,
        output_path: str = None
    ) -> Image.Image:
        """
        Run the full pipeline.
        
        Args:
            product_image_path: Path to original product image
            scene_prompt: Description of desired scene
            output_path: Where to save the result (optional)
        
        Returns:
            Final composited image
        """
        print("=" * 60)
        print("PRODUCT PLACEMENT PIPELINE")
        print("=" * 60)
        
        # Step 1: Generate scene with product
        print("\n[1/3] Generating scene with Gemini...")
        generated_scene = self.generator.generate_scene(product_image_path, scene_prompt)
        print(f"✓ Generated scene: {generated_scene.size}")
        
        # Save intermediate result
        intermediate_path = config.OUTPUT_DIR / "1_generated_scene.jpg"
        generated_scene.save(intermediate_path)
        print(f"  Saved to: {intermediate_path}")
        
        # Step 2: Find product location
        print("\n[2/3] Finding product location...")
        original_product = Image.open(product_image_path)
        bbox = self.matcher.find_product_location(original_product, generated_scene)
        
        if bbox is None:
            print("✗ Could not find product in generated scene!")
            print("  Try adjusting the prompt or using a different matching method.")
            return generated_scene
        
        x, y, w, h = bbox
        print(f"✓ Found product at: x={x}, y={y}, w={w}, h={h}")
        
        # Save detection visualization
        import cv2
        import numpy as np
        scene_with_box = np.array(generated_scene.copy())
        cv2.rectangle(scene_with_box, (x, y), (x+w, y+h), (0, 255, 0), 3)
        detection_path = config.OUTPUT_DIR / "2_detected_location.jpg"
        Image.fromarray(scene_with_box).save(detection_path)
        print(f"  Saved detection to: {detection_path}")
        
        # Step 3: Composite original product
        print("\n[3/3] Compositing original product...")
        final_image = self.compositor.paste_product(original_product, generated_scene, bbox)
        print(f"✓ Composite complete: {final_image.size}")
        
        # Save final result
        if output_path is None:
            output_path = config.OUTPUT_DIR / "3_final_composite.jpg"
        
        final_image.save(output_path)
        print(f"  Saved to: {output_path}")
        
        print("\n" + "=" * 60)
        print("PIPELINE COMPLETE!")
        print("=" * 60)
        
        return final_image


def main():
    # Validate product image exists
    product_path = Path(PRODUCT_IMAGE_PATH)
    if not product_path.exists():
        print(f"Error: Product image not found at {PRODUCT_IMAGE_PATH}")
        print(f"Please place your product image at: {config.INPUT_DIR / 'product.png'}")
        return
    
    # Run pipeline with hardcoded config
    print(f"Product: {PRODUCT_IMAGE_PATH}")
    print(f"Prompt: {SCENE_PROMPT}")
    print(f"Method: {MATCHING_METHOD}")
    print()
    
    pipeline = Pipeline(matching_method=MATCHING_METHOD)
    pipeline.run(str(product_path), SCENE_PROMPT, OUTPUT_PATH)


if __name__ == "__main__":
    main()
