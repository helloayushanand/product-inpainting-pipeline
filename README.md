# Product Placement Pipeline POC

A proof-of-concept for placing products with text into AI-generated scenes while preserving text quality.

## Strategy

1. **Generate**: Use Gemini to generate a full scene *with* the product
2. **Locate**: Find where the product is in the generated image
3. **Replace**: Paste the original (sharp text) product over the generated (blurry text) product
4. **Blend**: *(Not implemented yet)* Blend edges for natural look

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up your API key:
```bash
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY
export GEMINI_API_KEY="your_key_here"
```

3. Place your product image in the `input/` folder

## Usage

### Basic Usage
```bash
python pipeline.py \
  --product input/product.png \
  --prompt "A bottle on a wooden table in a cozy cafe with warm lighting"
```

### With Feature Matching (more robust)
```bash
python pipeline.py \
  --product input/product.png \
  --prompt "A bottle on a wooden table in a cozy cafe" \
  --method feature
```

### Custom Output Path
```bash
python pipeline.py \
  --product input/product.png \
  --prompt "A bottle on a beach at sunset" \
  --output my_result.jpg
```

## Output

The pipeline creates three intermediate files in `output/`:
- `1_generated_scene.jpg` - The raw Gemini-generated scene
- `2_detected_location.jpg` - Visualization of where the product was found
- `3_final_composite.jpg` - Final result with original product pasted

## Components

- **`gemini_generator.py`** - Generates scenes using Gemini API
- **`product_matcher.py`** - Finds product location (template or feature matching)
- **`compositor.py`** - Pastes original product onto scene
- **`pipeline.py`** - Main orchestrator
- **`config.py`** - Configuration and paths

## Next Steps (Not Implemented)

- [ ] Blending (Laplacian pyramid, histogram matching)
- [ ] Perspective transform for better alignment
- [ ] Shadow preservation
- [ ] Edge refinement
