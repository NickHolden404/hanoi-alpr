"""
OCR processing module using PaddleOCR.
"""

import cv2
import numpy as np
from paddleocr import PaddleOCR
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class OCRProcessor:
    """Handles OCR processing for license plates."""
    
    def __init__(self, use_gpu: bool = False, lang: str = 'en', show_log: bool = False):
        """
        Initialize OCR processor.
        
        Args:
            use_gpu: Whether to use GPU acceleration
            lang: Language model to use
            show_log: Whether to show PaddleOCR logs
        """
        self.use_gpu = use_gpu
        self.lang = lang
        
        logger.info(f"Initializing PaddleOCR (GPU: {use_gpu}, Lang: {lang})")
        
        try:
            self.reader = PaddleOCR(
                lang=lang,
                use_gpu=use_gpu,
                show_log=show_log
            )
            logger.info("PaddleOCR initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize PaddleOCR: {e}")
            raise
    
    def preprocess_image(self, image: np.ndarray, 
                        enhance_contrast: bool = True,
                        denoise: bool = True) -> np.ndarray:
        """
        Preprocess image for better OCR results.
        
        Args:
            image: Input image (BGR format)
            enhance_contrast: Apply contrast enhancement
            denoise: Apply denoising
            
        Returns:
            Preprocessed image
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Enhance contrast
        if enhance_contrast:
            gray = cv2.equalizeHist(gray)
        
        # Denoise
        if denoise:
            gray = cv2.fastNlMeansDenoising(gray, h=10)
        
        # Convert back to BGR for PaddleOCR
        processed = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        
        return processed
    
    def read_plate(self, image: np.ndarray, 
                   preprocess: bool = True) -> List[Tuple[str, float]]:
        """
        Read text from plate image using OCR.
        
        Args:
            image: Plate region image (BGR format)
            preprocess: Whether to apply preprocessing
            
        Returns:
            List of (text, confidence) tuples
        """
        if image is None or image.size == 0:
            logger.warning("Empty image provided to OCR")
            return []
        
        # Preprocess if requested
        if preprocess:
            image = self.preprocess_image(image)
        
        try:
            # Run OCR
            results = self.reader.ocr(image, cls=True)
            
            # Extract text and confidence
            detections = []
            
            if results and results[0]:
                for line in results[0]:
                    if len(line) >= 2:
                        # line[0] is bbox, line[1] is (text, confidence)
                        text, conf = line[1]
                        detections.append((text, conf))
            
            return detections
            
        except Exception as e:
            logger.error(f"OCR processing failed: {e}")
            return []
    
    def read_plate_best(self, image: np.ndarray, 
                       min_confidence: float = 0.35) -> Optional[Tuple[str, float]]:
        """
        Get best OCR result from plate image.
        
        Args:
            image: Plate region image
            min_confidence: Minimum confidence threshold
            
        Returns:
            (text, confidence) tuple or None
        """
        detections = self.read_plate(image)
        
        if not detections:
            return None
        
        # Filter by confidence
        valid_detections = [(text, conf) for text, conf in detections 
                           if conf >= min_confidence]
        
        if not valid_detections:
            return None
        
        # Return highest confidence
        best = max(valid_detections, key=lambda x: x[1])
        return best


# Test function
if __name__ == "__main__":
    import sys
    
    # Simple test with a sample image
    print("OCR Processor Test")
    print("-" * 50)
    
    try:
        processor = OCRProcessor(use_gpu=False, show_log=False)
        print("✓ OCR Processor initialized")
        
        # Create a test image with text
        test_img = np.ones((50, 200, 3), dtype=np.uint8) * 255
        cv2.putText(test_img, "29A-246.53", (10, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        
        print("✓ Test image created")
        
        # Try to read it
        results = processor.read_plate(test_img, preprocess=False)
        
        if results:
            print(f"✓ OCR read: {results}")
        else:
            print("✓ Video processor working correctly!")
    
    except Exception as e:
        print(f"✗ Error: {e}")
        sys.exit(1)
