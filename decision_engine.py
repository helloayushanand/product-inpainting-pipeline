"""Decision engine for choosing 2D vs 3D rendering approach."""
from PIL import Image
from shape_classifier import ShapeClassifier, ProductShape
from perspective_estimator import PerspectiveEstimator
from typing import Tuple, Literal


RenderingMethod = Literal["2D", "3D"]


class DecisionEngine:
    """
    Intelligently decides whether to use 2D or 3D rendering
    based on product shape and scene perspective.
    """
    
    def __init__(self):
        """Initialize the decision engine."""
        self.shape_classifier = ShapeClassifier()
        self.perspective_estimator = PerspectiveEstimator()
    
    def decide_rendering_method(
        self,
        product_image: Image.Image,
        scene_image: Image.Image
    ) -> Tuple[RenderingMethod, ProductShape, float]:
        """
        Decide which rendering method to use.
        
        Args:
            product_image: Original product image
            scene_image: Generated scene image
        
        Returns:
            Tuple of (method, product_shape, viewing_angle)
            - method: "2D" or "3D"
            - product_shape: Detected shape
            - viewing_angle: Estimated camera angle for 3D rendering
        """
        print("\n🤖 Decision Engine: Analyzing product and scene...")
        
        # Step 1: Classify product shape
        product_shape = self.shape_classifier.classify(product_image)
        
        # Step 2: Estimate scene perspective
        camera_angle, elevation = self.perspective_estimator.estimate_perspective(scene_image)
        
        # Step 3: Decision logic
        method = self._make_decision(product_shape, camera_angle)
        
        print(f"\n✓ Decision: Use {method} rendering")
        print(f"  Product: {product_shape}, Scene angle: {camera_angle:.1f}°")
        
        return method, product_shape, camera_angle
    
    def _make_decision(
        self,
        product_shape: ProductShape,
        camera_angle: float
    ) -> RenderingMethod:
        """
        Apply decision logic.
        
        Decision Rules:
        1. FLAT products → always use 2D (homography works great)
        2. 3D products + mild angle (< 20°) → use 2D (good enough)
        3. 3D products + strong angle (≥ 20°) → use 3D (avoid distortion)
        """
        # Rule 1: Flat products always use 2D
        if product_shape == "FLAT":
            print(f"  → Rule: Flat product → 2D homography")
            return "2D"
        
        # Rule 2: Mild perspective, 2D is sufficient
        if camera_angle < 20:
            print(f"  → Rule: Mild perspective ({camera_angle:.1f}°) → 2D is sufficient")
            return "2D"
        
        # Rule 3: Strong perspective + volumetric product → need 3D
        if product_shape in ["CYLINDER", "BOX"]:
            print(f"  → Rule: {product_shape} + strong angle ({camera_angle:.1f}°) → 3D required")
            return "3D"
        
        # Default: use 2D for safety
        print(f"  → Rule: Default fallback → 2D")
        return "2D"


def test_decision_engine():
    """Test the decision engine."""
    print("Testing Decision Engine...")
    
    engine = DecisionEngine()
    
    # Test cases
    print("\n1. Tall product (bottle-like):")
    tall_img = Image.new('RGB', (400, 1000), color='white')
    scene_img = Image.new('RGB', (1024, 1024), color='lightgray')
    
    method, shape, angle = engine.decide_rendering_method(tall_img, scene_img)
    print(f"   Result: {method} method selected")
    
    print("\n2. Wide product (card-like):")
    wide_img = Image.new('RGB', (1000, 400), color='white')
    
    method, shape, angle = engine.decide_rendering_method(wide_img, scene_img)
    print(f"   Result: {method} method selected")
    
    print("\n✓ Decision engine test complete!")


if __name__ == "__main__":
    test_decision_engine()
