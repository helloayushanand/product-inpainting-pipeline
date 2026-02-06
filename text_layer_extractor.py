"""Extract text/logos from product images for separate compositing."""
import cv2
import numpy as np
from PIL import Image
from typing import Tuple, Optional
import easyocr


class TextLayerExtractor:
    """
    Extracts text and design elements from product images.
    
    This uses OCR to detect text regions, then creates:
    1. A text layer mask (what to extract)
    2. A clean product base (with text inpainted)
    
    The text layer can then be composited separately to preserve fidelity.
    """
    
    def __init__(self, languages: list = ['en'], use_gpu: bool = False):
        """
        Initialize text extractor.
        
        Args:
            languages: Languages for OCR detection
            use_gpu: Whether to use GPU for OCR
        """
        self.reader = easyocr.Reader(languages, gpu=use_gpu)
    
    def detect_text_regions(
        self, 
        image: Image.Image,
        expand_margin: int = 5,
        refine_mask: bool = True
    ) -> np.ndarray:
        """
        Detect all text regions in the image.
        
        Args:
            image: Product image
            expand_margin: Pixels to expand text bounding boxes
            refine_mask: If True, refine mask to only actual text pixels
        
        Returns:
            Binary mask (uint8, 255=text region)
        """
        # Convert to numpy
        img_np = np.array(image.convert("RGB"))
        h, w = img_np.shape[:2]
        
        # Detect text
        print("  Detecting text regions with OCR...")
        results = self.reader.readtext(img_np)
        
        # Create mask
        if refine_mask:
            # Use precise pixel-level masking
            mask = self._create_precise_text_mask(img_np, results, expand_margin)
        else:
            # Use bounding boxes (old method)
            mask = np.zeros((h, w), dtype=np.uint8)
            for (bbox, text, conf) in results:
                pts = np.array(bbox, dtype=np.int32)
                center = pts.mean(axis=0)
                pts_expanded = center + (pts - center) * (1 + expand_margin / 100)
                pts_expanded = pts_expanded.astype(np.int32)
                cv2.fillPoly(mask, [pts_expanded], 255)
                print(f"    Found: '{text}' (conf: {conf:.2f})")
            
            # Dilate mask to capture full text + shadows
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=2)
        
        print(f"  Detected {len(results)} text regions")
        return mask
    
    def _create_precise_text_mask(
        self,
        img_np: np.ndarray,
        ocr_results: list,
        expand_margin: int
    ) -> np.ndarray:
        """
        Create precise text mask by segmenting actual text pixels within OCR boxes.
        
        Args:
            img_np: Image as numpy array (RGB)
            ocr_results: OCR results from EasyOCR
            expand_margin: Margin to expand boxes before segmentation
        
        Returns:
            Refined binary mask with only text pixels
        """
        h, w = img_np.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        
        for (bbox, text, conf) in ocr_results:
            print(f"    Found: '{text}' (conf: {conf:.2f})")
            
            # Get bounding box
            pts = np.array(bbox, dtype=np.int32)
            
            # Get axis-aligned bounding box
            x_min = max(0, int(np.min(pts[:, 0])))
            x_max = min(w, int(np.max(pts[:, 0])))
            y_min = max(0, int(np.min(pts[:, 1])))
            y_max = min(h, int(np.max(pts[:, 1])))
            
            # Extract region
            if x_max > x_min and y_max > y_min:
                region = gray[y_min:y_max, x_min:x_max]
                
                # Apply adaptive thresholding to isolate text
                # Text is typically darker than background
                thresh = cv2.adaptiveThreshold(
                    region,
                    255,
                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY_INV,  # INV because text is dark
                    blockSize=11,
                    C=5
                )
                
                # Clean up noise
                kernel = np.ones((2, 2), np.uint8)
                thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
                thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
                
                # Place refined mask into full mask
                mask[y_min:y_max, x_min:x_max] = cv2.bitwise_or(
                    mask[y_min:y_max, x_min:x_max],
                    thresh
                )
        
        # Slight dilation to ensure we don't lose thin text strokes
        kernel = np.ones((2, 2), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
        
        print(f"  Refined mask to only text pixels")
        return mask
    
    def detect_edges_as_design(
        self,
        image: Image.Image,
        edge_threshold: int = 100
    ) -> np.ndarray:
        """
        Detect strong edges as design elements (logos, patterns).
        
        This catches non-text design elements like logos, icons, etc.
        
        Args:
            image: Product image
            edge_threshold: Threshold for edge detection
        
        Returns:
            Binary mask (uint8, 255=design region)
        """
        img_np = np.array(image.convert("RGB"))
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        
        # Detect edges
        edges = cv2.Canny(gray, edge_threshold // 2, edge_threshold)
        
        # Dilate to connect nearby edges
        kernel = np.ones((3, 3), np.uint8)
        edges_dilated = cv2.dilate(edges, kernel, iterations=1)
        
        return edges_dilated
    
    def create_text_layer_mask(
        self,
        image: Image.Image,
        include_edges: bool = False
    ) -> np.ndarray:
        """
        Create complete text+design layer mask.
        
        Args:
            image: Product image
            include_edges: Whether to include edge-detected designs
        
        Returns:
            Binary mask (uint8, 255=text/design to extract)
        """
        # Get text regions
        text_mask = self.detect_text_regions(image)
        
        if include_edges:
            # Get design elements
            edge_mask = self.detect_edges_as_design(image)
            
            # Combine masks
            combined_mask = cv2.bitwise_or(text_mask, edge_mask)
            print(f"  Combined text + edge masks")
            return combined_mask
        
        return text_mask
    
    def inpaint_clean_product(
        self,
        image: Image.Image,
        text_mask: np.ndarray,
        method: str = "telea"
    ) -> Image.Image:
        """
        Remove text from product to create clean base.
        
        Uses inpainting to fill in text regions with surrounding texture.
        
        Args:
            image: Original product image
            text_mask: Binary mask of text regions to remove
            method: "telea" or "ns" (Navier-Stokes)
        
        Returns:
            Clean product image with text removed
        """
        img_np = np.array(image.convert("RGB"))
        
        # Choose inpainting method
        if method == "telea":
            inpaint_flag = cv2.INPAINT_TELEA
        else:
            inpaint_flag = cv2.INPAINT_NS
        
        print(f"  Inpainting text regions (method: {method})...")
        inpainted = cv2.inpaint(
            img_np,
            text_mask,
            inpaintRadius=7,
            flags=inpaint_flag
        )
        
        return Image.fromarray(inpainted)
    
    def extract(
        self,
        image: Image.Image,
        include_edges: bool = False,
        inpaint_method: str = "telea"
    ) -> Tuple[Image.Image, Image.Image, np.ndarray]:
        """
        Full extraction pipeline.
        
        Args:
            image: Original product image
            include_edges: Whether to extract edge-based designs
            inpaint_method: Inpainting algorithm
        
        Returns:
            Tuple of:
            - text_layer: RGBA image with only text/designs (transparent elsewhere)
            - clean_product: RGB image with text removed
            - text_mask: Binary mask of extracted regions
        """
        print("=" * 60)
        print("TEXT LAYER EXTRACTION")
        print("=" * 60)
        
        # Step 1: Create mask
        print("\n[1/3] Creating text layer mask...")
        text_mask = self.create_text_layer_mask(image, include_edges)
        
        # Step 2: Extract text layer as RGBA
        print("\n[2/3] Extracting text layer...")
        img_np = np.array(image.convert("RGB"))
        text_layer_rgba = np.zeros(
            (img_np.shape[0], img_np.shape[1], 4),
            dtype=np.uint8
        )
        text_layer_rgba[:, :, :3] = img_np
        text_layer_rgba[:, :, 3] = text_mask
        text_layer = Image.fromarray(text_layer_rgba)
        
        # Step 3: Inpaint clean product
        print("\n[3/3] Creating clean product base...")
        clean_product = self.inpaint_clean_product(
            image,
            text_mask,
            method=inpaint_method
        )
        
        print("\n" + "=" * 60)
        print("EXTRACTION COMPLETE!")
        print("=" * 60)
        
        return text_layer, clean_product, text_mask


class TextLayerCompositor:
    """
    Composites extracted text layer back onto generated scene.
    
    Applies the same homography transformation to text that was
    applied to the product placement.
    """
    
    def composite_text_layer(
        self,
        scene: Image.Image,
        text_layer: Image.Image,
        homography: np.ndarray,
        bbox: Tuple[int, int, int, int],
        apply_lighting: bool = True
    ) -> Image.Image:
        """
        Composite text layer onto scene with perspective transform and lighting.
        
        Args:
            scene: Generated scene with product placement
            text_layer: RGBA text layer from extraction
            homography: Homography matrix from product matching
            bbox: Detected product bounding box (x, y, w, h)
            apply_lighting: Whether to adjust text lighting to match scene
        
        Returns:
            Scene with text composited
        """
        scene_np = np.array(scene.convert("RGB")).astype(np.float32)
        text_np = np.array(text_layer)
        
        # Warp text layer using homography
        h, w = scene_np.shape[:2]
        text_warped = cv2.warpPerspective(
            text_np,
            homography,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0)
        )
        
        # Extract RGB and alpha
        if text_warped.shape[2] == 4:
            text_rgb = text_warped[:, :, :3].astype(np.float32)
            text_alpha = text_warped[:, :, 3] / 255.0
        else:
            text_rgb = text_warped.astype(np.float32)
            text_alpha = np.ones((h, w), dtype=np.float32)
        
        # Apply lighting adjustment to text
        if apply_lighting:
            print("  Applying lighting transfer to text layer...")
            text_rgb = self._adjust_text_lighting(text_rgb, scene_np, text_alpha, bbox)
        
        # Alpha blend
        text_alpha_3d = np.stack([text_alpha] * 3, axis=2)
        result = (
            text_alpha_3d * text_rgb +
            (1 - text_alpha_3d) * scene_np
        )
        
        return Image.fromarray(result.astype(np.uint8))
    
    def _adjust_text_lighting(
        self,
        text_rgb: np.ndarray,
        scene: np.ndarray,
        text_alpha: np.ndarray,
        bbox: Tuple[int, int, int, int]
    ) -> np.ndarray:
        """
        Adjust text lighting to match the scene.
        
        Args:
            text_rgb: Text RGB colors (float32, 0-255)
            scene: Scene RGB (float32, 0-255)
            text_alpha: Text alpha mask (float32, 0-1)
            bbox: Bounding box to sample lighting from
        
        Returns:
            Text RGB with adjusted lighting
        """
        x, y, w, h = bbox
        
        # Sample scene lighting from product region
        scene_region = scene[y:y+h, x:x+w]
        
        # Get text pixels (where alpha > 0)
        text_mask = text_alpha > 0.1
        if not np.any(text_mask):
            return text_rgb
        
        # Calculate scene lighting statistics in the region
        scene_mean = np.mean(scene_region.reshape(-1, 3), axis=0)
        scene_brightness = np.mean(scene_mean)
        
        # Calculate text statistics (only non-transparent pixels)
        text_pixels = text_rgb[text_mask]
        text_mean = np.mean(text_pixels, axis=0) if len(text_pixels) > 0 else np.array([128, 128, 128])
        text_brightness = np.mean(text_mean)
        
        # Adjust text brightness to match scene
        # Use a softer adjustment (blend 70% scene, 30% original)
        brightness_ratio = scene_brightness / max(text_brightness, 1)
        brightness_adjustment = 0.7 * brightness_ratio + 0.3 * 1.0
        
        text_adjusted = text_rgb * brightness_adjustment
        
        # Apply gentle color temperature shift
        # Shift text colors slightly toward scene colors
        color_shift = (scene_mean - text_mean) * 0.15  # Subtle shift
        text_adjusted = text_adjusted + color_shift
        
        # Clip to valid range
        text_adjusted = np.clip(text_adjusted, 0, 255)
        
        print(f"    Scene brightness: {scene_brightness:.1f}, Adjustment: {brightness_adjustment:.2f}x")
        
        return text_adjusted

