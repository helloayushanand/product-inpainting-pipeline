"""Pipeline with text layer extraction for preserving product text fidelity."""
from pathlib import Path
from PIL import Image

import config
from gemini_generator import GeminiGenerator
from product_matcher import ProductMatcher
from compositor import Compositor
from text_layer_extractor import TextLayerExtractor, TextLayerCompositor


# ============================================
# CONFIGURATION - Edit these values
# ============================================
# PRODUCT_IMAGE_PATH = "/Users/ayush/Downloads/390432_media_swatch_0_17-12-25-08-45-19__1_-removebg-preview.png"

PRODUCT_IMAGE_PATH = '/Users/ayush/Downloads/Untitled design (69).png'
SCENE_PROMPT = "place this on a table near the window. it must look like a real life lifestyle shot of this product. do not change any deatails of the product. The product should be kept in the center of the image. only front still shot."
MATCHING_METHOD = "feature"  # "template" or "feature"
USE_TEXT_EXTRACTION = True  # Enable text layer extraction
OUTPUT_PATH = None  # None = auto, or specify custom path
# ============================================


class TextPreservingPipeline:
    """
    Enhanced pipeline that preserves text/logo fidelity.
    
    Workflow:
    1. Extract text layer from product
    2. Generate clean product base (text inpainted)
    3. Generate scene with clean product
    4. Find product location + homography
    5. Composite clean product
    6. Composite text layer with same homography
    """
    
    def __init__(self, matching_method: str = "feature"):
        """
        Initialize the pipeline.
        
        Args:
            matching_method: "template" or "feature"
        """
        self.generator = GeminiGenerator()
        self.matcher = ProductMatcher(method=matching_method)
        from layered_compositor import LayeredCompositor
        self.compositor = LayeredCompositor(feather_amount=30)
        self.text_extractor = TextLayerExtractor(languages=['en'], use_gpu=False)
        self.text_compositor = TextLayerCompositor()
    
    def run(
        self, 
        product_image_path: str, 
        scene_prompt: str,
        use_text_extraction: bool = True,
        output_path: str = None
    ) -> Image.Image:
        """
        Run the full pipeline with text preservation.
        
        Args:
            product_image_path: Path to original product image
            scene_prompt: Description of desired scene
            use_text_extraction: Whether to extract and preserve text
            output_path: Where to save the result (optional)
        
        Returns:
            Final composited image
        """
        print("=" * 60)
        print("TEXT-PRESERVING PRODUCT PLACEMENT PIPELINE")
        print("=" * 60)
        
        original_product = Image.open(product_image_path)
        
        # Conditional: Extract text layer or use original product
        if use_text_extraction:
            # Step 0: Extract text layer
            print("\n[0/5] Extracting text layer from product...")
            text_layer, clean_product, text_mask = self.text_extractor.extract(
                original_product,
                include_edges=False,  # Set to True to also extract logos
                inpaint_method="telea"
            )
            
            # Save intermediate results
            text_layer.save(config.OUTPUT_DIR / "0a_text_layer.png")
            clean_product.save(config.OUTPUT_DIR / "0b_clean_product.png")
            Image.fromarray(text_mask).save(config.OUTPUT_DIR / "0c_text_mask.png")
            print(f"  Text layer saved to: {config.OUTPUT_DIR / '0a_text_layer.png'}")
            
            product_for_generation = clean_product
        else:
            print("\n[0/5] Using original product (text extraction disabled)")
            product_for_generation = original_product
            text_layer = None
        
        # Step 1: Generate scene with product (clean or original)
        print("\n[1/5] Generating 4K scene with Gemini...")
        generated_scene = self.generator.generate_scene(
            product_for_generation, 
            scene_prompt
        )
        print(f"✓ Generated scene: {generated_scene.size}")
        
        # Save intermediate result
        intermediate_path = config.OUTPUT_DIR / "1_generated_scene.jpg"
        generated_scene.save(intermediate_path)
        print(f"  Saved to: {intermediate_path}")
        
        # Step 2: Find product location with homography
        # IMPORTANT: Use ORIGINAL product for matching (has more features)
        # Then apply same transform to clean product + text layer
        print("\n[2/5] Finding product location...")
        result = self.matcher.find_product_location(original_product, generated_scene)

        
        if result is None:
            print("✗ Could not find product in generated scene!")
            print("  Try adjusting the prompt or using a different matching method.")
            return generated_scene
        
        bbox, homography = result
        x, y, w, h = bbox
        print(f"✓ Found product at: x={x}, y={y}, w={w}, h={h}")
        if homography is not None:
            print(f"  Homography matrix available for precise placement")
        
        # Save detection visualization
        import cv2
        import numpy as np
        scene_with_box = np.array(generated_scene.copy())
        cv2.rectangle(scene_with_box, (x, y), (x+w, y+h), (0, 255, 0), 3)
        detection_path = config.OUTPUT_DIR / "2_detected_location.jpg"
        Image.fromarray(scene_with_box).save(detection_path)
        print(f"  Saved detection to: {detection_path}")
        
        # Step 3: Composite product (ONLY if text extraction is disabled)
        # When text extraction is enabled, Gemini already placed the clean product!
        # We just need to add the text layer back.
        if use_text_extraction:
            print("\n[3/5] Skipping product composite (Gemini already placed it)...")
            scene_with_product = generated_scene
            print(f"✓ Using Gemini's generated scene directly")
        else:
            print("\n[3/5] Compositing product...")
            scene_with_product = self.compositor.composite(
                original_product, 
                generated_scene, 
                bbox,
                homography=homography,
                remove_bg=False
            )
            print(f"✓ Product composite complete")
        
        # Save intermediate result (if we composited)
        if not use_text_extraction:
            product_composite_path = config.OUTPUT_DIR / "3_product_composite.jpg"
            scene_with_product.save(product_composite_path)
            print(f"  Saved to: {product_composite_path}")

        
        # Step 4: Composite text layer (if extracted)
        if use_text_extraction and text_layer is not None and homography is not None:
            print("\n[4/5] Compositing text layer with perspective...")
            final_image = self.text_compositor.composite_text_layer(
                scene_with_product,
                text_layer,
                homography,
                bbox
            )
            print(f"✓ Text layer composite complete")
        else:
            print("\n[4/5] Skipping text layer composite (not extracted or no homography)")
            final_image = scene_with_product
        
        # Save final result
        if output_path is None:
            output_path = config.OUTPUT_DIR / "4_final_composite.jpg"
        
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
    
    # Run pipeline with text extraction
    print(f"Product: {PRODUCT_IMAGE_PATH}")
    print(f"Prompt: {SCENE_PROMPT}")
    print(f"Method: {MATCHING_METHOD}")
    print(f"Text Extraction: {USE_TEXT_EXTRACTION}")
    print()
    
    pipeline = TextPreservingPipeline(matching_method=MATCHING_METHOD)
    pipeline.run(
        str(product_path), 
        SCENE_PROMPT,
        use_text_extraction=USE_TEXT_EXTRACTION,
        output_path=OUTPUT_PATH
    )


if __name__ == "__main__":
    main()
