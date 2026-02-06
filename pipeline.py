"""Main pipeline orchestrator."""
from pathlib import Path
from PIL import Image

import config
from gemini_generator import GeminiGenerator
from product_matcher import ProductMatcher
from compositor import Compositor
from decision_engine import DecisionEngine
from renderer_3d import Pseudo3DRenderer
from homography_analyzer import HomographyAnalyzer


# ============================================
# CONFIGURATION - Edit these values
# ============================================
PRODUCT_IMAGE_PATH = "/Users/ayush/Downloads/Untitled design (69).png"  # Path to your product image
SCENE_PROMPT = "place this on a table near the window. it must look like a real life lifestyle shot of this product. do not change any deatails of the product. The product should be kept in the center of the image."  # Scene description
MATCHING_METHOD = "feature"  # "template" or "feature"
OUTPUT_PATH = None  # None = auto (output/3_final_composite.jpg), or specify custom path
# ============================================


class Pipeline:
    def __init__(self, matching_method: str = "feature"):
        """
        Initialize the pipeline.
        
        Args:
            matching_method: "template" or "feature"
        """
        self.generator = GeminiGenerator()
        self.matcher = ProductMatcher(method=matching_method)
        # Use LayeredCompositor for better quality
        from layered_compositor import LayeredCompositor
        self.compositor = LayeredCompositor(feather_amount=10)  # Reduced from 30 for sharper edges
        # Add 3D rendering system
        self.decision_engine = DecisionEngine()
        self.renderer_3d = Pseudo3DRenderer()
        self.homography_analyzer = HomographyAnalyzer()
    
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
        
        # Step 1: Generate scene with product (4K)
        print("\n[1/3] Generating 4K scene with Gemini...")
        generated_scene = self.generator.generate_scene(product_image_path, scene_prompt)
        print(f"✓ Generated scene: {generated_scene.size}")
        
        # Save intermediate result
        intermediate_path = config.OUTPUT_DIR / "1_generated_scene.jpg"
        generated_scene.save(intermediate_path)
        print(f"  Saved to: {intermediate_path}")
        
        # Step 2: Decide rendering method (2D vs 3D)
        print("\n[2/4] Analyzing product and scene...")
        original_product = Image.open(product_image_path)
        
        try:
            rendering_method, product_shape, viewing_angle = self.decision_engine.decide_rendering_method(
                original_product,
                generated_scene
            )
        except Exception as e:
            print(f"⚠ Decision engine failed: {e}, falling back to 2D")
            rendering_method = "2D"
            product_shape = "UNKNOWN"
            viewing_angle = 0.0
        
        # Step 3: Find product location
        print(f"\n[3/4] Finding product location (using {rendering_method} method)...")
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
        
        # Step 4: Extract perspective angle from homography (if available)
        if homography is not None and rendering_method == "3D":
            # Extract actual viewing angle from homography matrix
            homography_angle = self.homography_analyzer.extract_perspective_angle(homography)
            print(f"  Extracted perspective from homography: {homography_angle:.1f}°")
            # Use homography angle instead of estimated angle for better matching
            viewing_angle = homography_angle
        
        # Step 5: Apply 3D transform if needed, then composite
        print(f"\n[4/4] Compositing with {rendering_method} rendering...")
        
        # Apply 3D transform if selected
        if rendering_method == "3D":
            try:
                print(f"  Applying pseudo-3D transform for {product_shape} at {viewing_angle:.1f}°...")
                if product_shape == "CYLINDER":
                    transformed_product = self.renderer_3d.render_cylinder_view(
                        original_product,
                        viewing_angle=viewing_angle
                    )
                elif product_shape == "BOX":
                    transformed_product = self.renderer_3d.render_box_view(
                        original_product,
                        viewing_angle=viewing_angle
                    )
                else:
                    transformed_product = original_product
                
                # Strategy: Apply subtle 3D transform, then let homography handle placement
                # - If viewing_angle < 15°: Skip 3D, use homography only
                # - If viewing_angle >= 15°: Apply 3D, but DISABLE homography to avoid conflict
                
                if viewing_angle < 15:
                    # Mild perspective - just use standard homography
                    print(f"  Angle too mild ({viewing_angle:.1f}°), using 2D homography only")
                    final_image = self.compositor.composite(
                        original_product,
                        generated_scene,
                        bbox,
                        homography=homography,
                        remove_bg=True
                    )
                else:
                    # Strong perspective - use 3D transform WITHOUT homography
                    # The 3D transform already matches the viewing angle from homography
                    print(f"  Using 3D transform (no homography - angle already matched)")
                    final_image = self.compositor.composite(
                        transformed_product,
                        generated_scene,
                        bbox,
                        homography=None,  # Disable to avoid double-transform
                        remove_bg=True
                    )
                
                print(f"✓ 3D composite complete: {final_image.size}")
                
            except Exception as e:
                print(f"⚠ 3D rendering failed: {e}")
                print(f"  Falling back to 2D homography...")
                # Fallback to 2D
                final_image = self.compositor.composite(
                    original_product,
                    generated_scene,
                    bbox,
                    homography=homography,
                    remove_bg=True
                )
                print(f"✓ 2D composite complete (fallback): {final_image.size}")
        else:
            # Use standard 2D homography
            final_image = self.compositor.composite(
                original_product,
                generated_scene,
                bbox,
                homography=homography,
                remove_bg=True
            )
            print(f"✓ 2D composite complete: {final_image.size}")
        
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
