"""Find and match product location in generated image."""
import cv2
import numpy as np
from PIL import Image
from typing import Tuple, Optional

class ProductMatcher:
    def __init__(self, method: str = "template"):
        """
        Initialize product matcher.
        
        Args:
            method: Matching method - "template" or "feature" (SIFT/ORB)
        """
        self.method = method
    
    def find_product_location(
        self, 
        original_product: Image.Image, 
        generated_scene: Image.Image
    ) -> Optional[Tuple[Tuple[int, int, int, int], Optional[np.ndarray]]]:
        """
        Find the location of the product in the generated scene.
        
        Args:
            original_product: Original product image (with alpha channel if available)
            generated_scene: Generated scene containing the product
        
        Returns:
            Tuple of (bounding_box, homography_matrix) or None if not found
            - bounding_box: (x, y, w, h)
            - homography_matrix: 3x3 transformation matrix (only for feature matching)
        """
        if self.method == "template":
            bbox = self._template_matching(original_product, generated_scene)
            return (bbox, None) if bbox else None
        elif self.method == "feature":
            return self._feature_matching(original_product, generated_scene)
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def _template_matching(
        self, 
        template_img: Image.Image, 
        scene_img: Image.Image
    ) -> Optional[Tuple[int, int, int, int]]:
        """
        Use template matching to find product.
        Simple but works if the product hasn't changed much.
        """
        # Convert to numpy arrays
        template = np.array(template_img.convert("RGB"))
        scene = np.array(scene_img.convert("RGB"))
        
        # Convert to grayscale for matching
        template_gray = cv2.cvtColor(template, cv2.COLOR_RGB2GRAY)
        scene_gray = cv2.cvtColor(scene, cv2.COLOR_RGB2GRAY)
        
        # Try multiple scales
        best_match = None
        best_val = -1
        
        for scale in np.linspace(0.3, 2.0, 20):
            # Resize template
            w = int(template_gray.shape[1] * scale)
            h = int(template_gray.shape[0] * scale)
            
            if w > scene_gray.shape[1] or h > scene_gray.shape[0]:
                continue
            
            resized_template = cv2.resize(template_gray, (w, h))
            
            # Perform template matching
            result = cv2.matchTemplate(scene_gray, resized_template, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            
            if max_val > best_val:
                best_val = max_val
                best_match = (max_loc[0], max_loc[1], w, h)
        
        # Return best match if confidence is high enough
        if best_val > 0.5:
            return best_match
        
        return None
    
    def _feature_matching(
        self, 
        product_img: Image.Image, 
        scene_img: Image.Image
    ) -> Optional[Tuple[Tuple[int, int, int, int], np.ndarray]]:
        """
        Use feature matching (ORB) to find product.
        More robust to perspective changes.
        
        Returns:
            Tuple of (bounding_box, homography_matrix) or None if not found
        """
        # Convert to numpy arrays
        product = np.array(product_img.convert("RGB"))
        scene = np.array(scene_img.convert("RGB"))
        
        # Convert to grayscale
        product_gray = cv2.cvtColor(product, cv2.COLOR_RGB2GRAY)
        scene_gray = cv2.cvtColor(scene, cv2.COLOR_RGB2GRAY)
        
        # Initialize ORB detector with more features
        orb = cv2.ORB_create(nfeatures=5000)  # Increased from 2000
        
        # Find keypoints and descriptors
        kp1, des1 = orb.detectAndCompute(product_gray, None)
        kp2, des2 = orb.detectAndCompute(scene_gray, None)
        
        print(f"  Found {len(kp1)} keypoints in product, {len(kp2)} in scene")
        
        if des1 is None or des2 is None:
            print("  No descriptors found")
            return None
        
        # Match features using BFMatcher
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        matches = bf.knnMatch(des1, des2, k=2)
        
        # Apply ratio test (Lowe's ratio test)
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.75 * n.distance:  # More lenient threshold
                    good_matches.append(m)
        
        print(f"  Found {len(good_matches)} good matches")
        
        # Need at least 4 matches for homography
        if len(good_matches) < 4:
            print("  Not enough good matches (need at least 4)")
            return None
        
        # Get matched keypoints
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        # Find homography with RANSAC
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        if M is None:
            print("  Could not compute homography")
            return None
        
        # Count inliers
        inliers = np.sum(mask)
        print(f"  Homography inliers: {inliers}/{len(good_matches)}")
        
        # Get bounding box of product in scene
        h, w = product_gray.shape
        pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)
        
        # Calculate bounding box
        x_coords = dst[:, 0, 0]
        y_coords = dst[:, 0, 1]
        
        x = int(np.min(x_coords))
        y = int(np.min(y_coords))
        w = int(np.max(x_coords) - x)
        h = int(np.max(y_coords) - y)
        
        # Sanity check: bbox should be within scene bounds
        if x < 0 or y < 0 or x + w > scene_gray.shape[1] or y + h > scene_gray.shape[0]:
            print(f"  Bounding box out of bounds: ({x}, {y}, {w}, {h})")
            # Clip to scene bounds
            x = max(0, x)
            y = max(0, y)
            w = min(w, scene_gray.shape[1] - x)
            h = min(h, scene_gray.shape[0] - y)
            print(f"  Clipped to: ({x}, {y}, {w}, {h})")
        
        return ((x, y, w, h), M)

if __name__ == "__main__":
    # Test the matcher
    import config
    
    matcher = ProductMatcher(method="template")
    
    product_path = config.INPUT_DIR / "product.png"
    scene_path = config.OUTPUT_DIR / "generated_scene.jpg"
    
    if product_path.exists() and scene_path.exists():
        product = Image.open(product_path)
        scene = Image.open(scene_path)
        
        bbox = matcher.find_product_location(product, scene)
        
        if bbox:
            x, y, w, h = bbox
            print(f"Found product at: x={x}, y={y}, w={w}, h={h}")
            
            # Draw bounding box for visualization
            scene_cv = np.array(scene)
            cv2.rectangle(scene_cv, (x, y), (x+w, y+h), (0, 255, 0), 3)
            result = Image.fromarray(scene_cv)
            result.save(config.OUTPUT_DIR / "detected_location.jpg")
            print(f"Saved detection result to {config.OUTPUT_DIR / 'detected_location.jpg'}")
        else:
            print("Could not find product in scene")
    else:
        print("Please run gemini_generator.py first")
