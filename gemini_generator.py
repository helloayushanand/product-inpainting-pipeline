"""Generate images using Gemini API."""
from google import genai
from google.genai import types
from PIL import Image
import io
import config

class GeminiGenerator:
    def __init__(self, api_key: str = None):
        """Initialize Gemini generator."""
        self.api_key = api_key or config.GEMINI_API_KEY
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not set in environment or config")
        
        self.client = genai.Client(api_key=self.api_key)
    
    def generate_scene(self, product_image_path: str, scene_prompt: str) -> Image.Image:
        """
        Generate a scene with the product in it.
        
        Args:
            product_image_path: Path to the original product image
            scene_prompt: Description of the scene (e.g., "A bottle on a wooden table in a cafe")
        
        Returns:
            PIL Image of the generated scene
        """
        # Load the product image
        product_img = Image.open(product_image_path)
        
        # Create the full prompt
        full_prompt = f"""Generate a photorealistic image based on this description: {scene_prompt}

The product shown in the reference image should be placed naturally in the scene.
Make sure the product is clearly visible and well-lit.
Generate realistic shadows and lighting that match the environment."""
        
        # Generate image using Gemini with 4K resolution
        response = self.client.models.generate_content(
            model=config.GEMINI_MODEL,
            contents=[full_prompt, product_img],
            config=types.GenerateContentConfig(
                response_modalities=['IMAGE'],
                image_config=types.ImageConfig(image_size="4K"),
            )
        )
        
        # Extract image from response
        if response.candidates and response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                if hasattr(part, 'inline_data') and part.inline_data:
                    image_data = part.inline_data.data
                    generated_image = Image.open(io.BytesIO(image_data))
                    return generated_image
        
        raise ValueError("No image generated in response")

if __name__ == "__main__":
    # Test the generator
    generator = GeminiGenerator()
    
    # Example usage
    test_product = config.INPUT_DIR / "product.png"
    if test_product.exists():
        scene = generator.generate_scene(
            str(test_product),
            "A bottle on a wooden table in a cozy cafe with warm lighting"
        )
        scene.save(config.OUTPUT_DIR / "generated_scene.jpg")
        print(f"Generated scene saved to {config.OUTPUT_DIR / 'generated_scene.jpg'}")
    else:
        print(f"Please place a product image at {test_product}")
