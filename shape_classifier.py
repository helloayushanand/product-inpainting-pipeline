"""Product shape classification for determining rendering method."""
import cv2
import numpy as np
from PIL import Image
from typing import Literal


ProductShape = Literal["FLAT", "CYLINDER", "BOX", "UNKNOWN"]


class ShapeClassifier:
    """
    Classifies product images into shape categories.
    Uses simple heuristics based on aspect ratio and edge detection.
    """
    
    def classify(self, product_image: Image.Image) -> ProductShape:
        """
        Classify product shape.
        
        Args:
            product_image: Product image to classify
        
        Returns:
            Product shape: FLAT, CYLINDER, BOX, or UNKNOWN
        """
        # Convert to numpy array
        img_np = np.array(product_image.convert("RGB"))
        h, w = img_np.shape[:2]
        aspect_ratio = h / w
        
        print(f"  📐 Analyzing product shape...")
        print(f"     Size: {w}x{h}, Aspect ratio: {aspect_ratio:.2f}")
        
        # Rule 1: Very tall and narrow = likely cylinder (bottle, can)
        if aspect_ratio > 1.5 and w < h * 0.6:
            # Check for circular top (bottle cap)
            has_circular_top = self._detect_circular_top(img_np)
            if has_circular_top:
                print(f"     ✓ Detected: CYLINDER (tall + circular top)")
                return "CYLINDER"
        
        # Rule 2: Square-ish aspect ratio = likely box
        if 0.7 < aspect_ratio < 1.4:
            # Check for rectangular edges
            has_rectangular_shape = self._detect_rectangular_edges(img_np)
            if has_rectangular_shape:
                print(f"     ✓ Detected: BOX (square aspect + rectangular edges)")
                return "BOX"
        
        # Rule 3: Very wide and short = likely flat (card, poster)
        if aspect_ratio < 0.7 or aspect_ratio > 2.5:
            print(f"     ✓ Detected: FLAT (extreme aspect ratio)")
            return "FLAT"
        
        # Default: treat as flat for safety (2D homography works best)
        print(f"     ⚠ Uncertain shape, defaulting to: FLAT")
        return "FLAT"
    
    def _detect_circular_top(self, image: np.ndarray) -> bool:
        """
        Detect if product has a circular top (like bottle cap).
        
        Args:
            image: RGB image as numpy array
        
        Returns:
            True if circular top detected
        """
        # Look at top 20% of image
        h, w = image.shape[:2]
        top_region = image[:int(h * 0.2), :]
        
        # Convert to grayscale
        gray = cv2.cvtColor(top_region, cv2.COLOR_RGB2GRAY)
        
        # Detect circles using Hough Circle Transform
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=20,
            param1=50,
            param2=30,
            minRadius=10,
            maxRadius=min(w, h) // 4
        )
        
        has_circle = circles is not None and len(circles) > 0
        return has_circle
    
    def _detect_rectangular_edges(self, image: np.ndarray) -> bool:
        """
        Detect if product has rectangular/box-like edges.
        
        Args:
            image: RGB image as numpy array
        
        Returns:
            True if rectangular edges detected
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return False
        
        # Get largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Approximate contour to polygon
        epsilon = 0.02 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)
        
        # Rectangular shapes typically have 4 corners
        num_corners = len(approx)
        is_rectangular = 4 <= num_corners <= 6
        
        return is_rectangular


def test_shape_classifier():
    """Test the shape classifier."""
    print("Testing Shape Classifier...")
    
    classifier = ShapeClassifier()
    
    # Test with different aspect ratios
    print("\n1. Testing tall image (bottle-like):")
    tall_img = Image.new('RGB', (400, 1000), color='white')
    shape1 = classifier.classify(tall_img)
    
    print("\n2. Testing square image (box-like):")
    square_img = Image.new('RGB', (600, 600), color='white')
    shape2 = classifier.classify(square_img)
    
    print("\n3. Testing wide image (card-like):")
    wide_img = Image.new('RGB', (1000, 400), color='white')
    shape3 = classifier.classify(wide_img)
    
    print(f"\n✓ Shape classification test complete!")
    print(f"   Results: Tall={shape1}, Square={shape2}, Wide={shape3}")


if __name__ == "__main__":
    test_shape_classifier()
