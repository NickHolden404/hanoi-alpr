# quick_ocr_test.py (FIXED)
from paddleocr import PaddleOCR
import cv2
import numpy as np

# Create a simple test image with text
test_image = np.ones((100, 300, 3), dtype=np.uint8) * 255
cv2.putText(test_image, "29A-12345", (10, 60), 
           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)
cv2.imwrite("test_ocr.jpg", test_image)

print("Testing PaddleOCR on simple text...")
ocr = PaddleOCR(lang='en')
result = ocr.predict("test_ocr.jpg")

print("\n✅ PaddleOCR is working!")

if result and len(result) > 0:
    # New API returns different format
    if 'rec_texts' in result[0]:
        texts = result[0]['rec_texts']
        scores = result[0]['rec_scores']
        
        print(f"\nDetected {len(texts)} text(s):")
        for text, score in zip(texts, scores):
            print(f"  '{text}' with confidence {score:.2%}")
    else:
        print(f"Raw result: {result}")
else:
    print("❌ No text detected!")