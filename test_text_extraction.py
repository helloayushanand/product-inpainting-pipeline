"""Test script for text layer extraction."""
from pathlib import Path
from PIL import Image
from text_layer_extractor import TextLayerExtractor
import config

# Product image path
PRODUCT_PATH = "/Users/ayush/Downloads/390432_media_swatch_0_17-12-25-08-45-19__1_-removebg-preview.png"

def test_text_extraction():
    """Test text extraction on product image."""
    print("Testing Text Layer Extraction")
    print("=" * 60)
    
    # Check if product exists
    if not Path(PRODUCT_PATH).exists():
        print(f"❌ Product image not found: {PRODUCT_PATH}")
        return
    
    print(f"✓ Product image found: {PRODUCT_PATH}")
    
    # Load product
    product = Image.open(PRODUCT_PATH)
    print(f"✓ Product size: {product.size}")
    
    # Create extractor
    print("\nInitializing TextLayerExtractor...")
    extractor = TextLayerExtractor(languages=['en'], use_gpu=False)
    
    # Extract text layer
    print("\nExtracting text layer...")
    try:
        text_layer, clean_product, text_mask = extractor.extract(
            product,
            include_edges=False,
            inpaint_method="telea"
        )
        
        # Save results
        output_dir = Path(config.OUTPUT_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        text_layer_path = output_dir / "test_text_layer.png"
        clean_product_path = output_dir / "test_clean_product.png"
        text_mask_path = output_dir / "test_text_mask.png"
        
        text_layer.save(text_layer_path)
        clean_product.save(clean_product_path)
        Image.fromarray(text_mask).save(text_mask_path)
        
        print(f"\n✓ Results saved:")
        print(f"  - Text layer: {text_layer_path}")
        print(f"  - Clean product: {clean_product_path}")
        print(f"  - Text mask: {text_mask_path}")
        
        print("\n" + "=" * 60)
        print("TEST PASSED! ✓")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Error during extraction: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_text_extraction()
