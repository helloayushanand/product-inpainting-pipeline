"""Debug script to test product matching."""
from PIL import Image
from product_matcher import ProductMatcher
import cv2
import numpy as np

# Load images
clean_product = Image.open("output/0b_clean_product.png")
generated_scene = Image.open("output/1_generated_scene.jpg")

print("Testing Product Matching")
print("=" * 60)
print(f"Clean product size: {clean_product.size}")
print(f"Generated scene size: {generated_scene.size}")

# Try feature matching
matcher = ProductMatcher(method="feature")
result = matcher.find_product_location(clean_product, generated_scene)

if result is None:
    print("\n❌ Feature matching FAILED - no product found")
    print("\nTrying template matching...")
    
    # Try template matching as fallback
    matcher_template = ProductMatcher(method="template")
    result = matcher_template.find_product_location(clean_product, generated_scene)
    
    if result is None:
        print("❌ Template matching also FAILED")
    else:
        bbox, homography = result
        print(f"✓ Template matching SUCCESS!")
        print(f"  Bbox: {bbox}")
else:
    bbox, homography = result
    print(f"\n✓ Feature matching SUCCESS!")
    print(f"  Bbox: {bbox}")
    print(f"  Homography: {'Available' if homography is not None else 'None'}")

# Visualize
if result:
    x, y, w, h = bbox
    scene_np = np.array(generated_scene)
    cv2.rectangle(scene_np, (x, y), (x+w, y+h), (0, 255, 0), 5)
    Image.fromarray(scene_np).save("output/debug_detection.jpg")
    print(f"\nSaved visualization to output/debug_detection.jpg")
