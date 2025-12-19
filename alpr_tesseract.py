# Hanoi ALPR with Tesseract - Much Faster!
# Install: brew install tesseract && pip install pytesseract

import cv2
import numpy as np
from ultralytics import YOLO
import pytesseract
import re
from pathlib import Path
from typing import List, Dict, Tuple
import time

class TesseractALPR:
    """Fast ALPR using Tesseract OCR."""
    
    def __init__(self):
        print("🚀 Initializing Tesseract ALPR System...")
        print("📦 Loading YOLOv8 model...")
        self.vehicle_model = YOLO('yolov8n.pt')
        
        # Test Tesseract
        try:
            pytesseract.get_tesseract_version()
            print("✅ Tesseract OCR ready!")
        except Exception as e:
            print("❌ Tesseract not found. Install with: brew install tesseract")
            raise e
        
        print("✅ System initialized!\n")
    
    def detect_vehicles(self, frame: np.ndarray) -> List[Dict]:
        """Detect vehicles."""
        results = self.vehicle_model(frame, verbose=False, conf=0.35)[0]
        
        detections = []
        for box in results.boxes:
            class_id = int(box.cls[0])
            if class_id in [2, 3, 7]:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                w, h = x2 - x1, y2 - y1
                
                # Skip small vehicles
                if w < 60 or h < 60:
                    continue
                
                detections.append({
                    'bbox': (int(x1), int(y1), int(x2), int(y2)),
                    'confidence': float(box.conf[0]),
                    'class': 'motorcycle' if class_id == 3 else 'car'
                })
        
        return detections
    
    def preprocess_for_ocr(self, image: np.ndarray) -> np.ndarray:
        """Aggressive preprocessing for plate recognition."""
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Upscale significantly
        h, w = gray.shape
        if h < 150:
            scale = 150 / h
            gray = cv2.resize(gray, None, fx=scale, fy=scale, 
                            interpolation=cv2.INTER_CUBIC)
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(gray, h=10)
        
        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(denoised)
        
        # Sharpen
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)
        
        # Binary threshold - try both light and dark plates
        _, binary1 = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _, binary2 = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        return binary1, binary2
    
    def validate_plate(self, text: str) -> Tuple[bool, str]:
        """Validate if text looks like a license plate."""
        text = text.upper().strip()
        
        # Remove garbage characters
        text = ''.join(c for c in text if c.isalnum() or c in ['-', '.', ' '])
        text = ' '.join(text.split())
        
        # Must have both letters and numbers
        has_numbers = any(c.isdigit() for c in text)
        has_letters = any(c.isalpha() for c in text)
        
        if len(text) >= 5 and has_numbers and has_letters:
            # Fix common OCR errors
            text = text.replace('O', '0').replace('I', '1').replace('S', '5')
            text = text.replace('Z', '2').replace('B', '8')
            return True, text
        
        return False, text
    
    def ocr_region(self, region: np.ndarray) -> Dict:
        """Run Tesseract OCR on region."""
        try:
            # Try multiple preprocessed versions
            binary1, binary2 = self.preprocess_for_ocr(region)
            
            best_result = {'text': '', 'confidence': 0, 'valid': False}
            
            for img in [binary1, binary2]:
                # Tesseract config for license plates
                config = '--psm 7 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-.'
                
                # Get text and confidence
                data = pytesseract.image_to_data(img, config=config, output_type=pytesseract.Output.DICT)
                
                # Combine text with good confidence
                texts = []
                confidences = []
                
                for i, conf in enumerate(data['conf']):
                    if conf > 0:
                        text = data['text'][i].strip()
                        if text:
                            texts.append(text)
                            confidences.append(conf)
                
                if texts:
                    combined = ''.join(texts)
                    avg_conf = sum(confidences) / len(confidences) / 100.0  # Convert to 0-1
                    
                    is_valid, formatted = self.validate_plate(combined)
                    
                    if is_valid and avg_conf > best_result['confidence']:
                        best_result = {
                            'text': formatted,
                            'confidence': avg_conf,
                            'valid': True
                        }
            
            return best_result
        
        except Exception as e:
            return {'text': '', 'confidence': 0.0, 'valid': False}
    
    def process_frame(self, frame: np.ndarray, frame_num: int) -> List[Dict]:
        """Process single frame."""
        detections = []
        
        vehicles = self.detect_vehicles(frame)
        
        for vehicle in vehicles[:8]:  # Check up to 8 vehicles
            x1, y1, x2, y2 = vehicle['bbox']
            h_vehicle = y2 - y1
            
            # Extract bottom portion where plates usually are
            y_start = y1 + int(h_vehicle * 0.55)
            plate_region = frame[y_start:y2, x1:x2]
            
            if plate_region.shape[0] < 20 or plate_region.shape[1] < 40:
                continue
            
            # Try OCR
            result = self.ocr_region(plate_region)
            
            if result['valid'] and result['confidence'] > 0.4:
                detections.append({
                    'frame': frame_num,
                    'plate': result['text'],
                    'confidence': result['confidence'],
                    'vehicle_type': vehicle['class'],
                    'bbox': vehicle['bbox']
                })
        
        return detections
    
    def process_video(self, video_path: str, output_path: str = None,
                     frame_skip: int = 10, max_frames: int = None) -> Dict:
        """Process video."""
        print(f"🎥 Processing: {Path(video_path).name}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video")
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if max_frames:
            total_frames = min(total_frames, max_frames)
        
        print(f"📊 {total_frames} frames @ {fps} FPS")
        print(f"⚙️  Processing every {frame_skip}th frame\n")
        
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        all_detections = []
        frame_num = 0
        start_time = time.time()
        last_print = 0
        
        while cap.isOpened() and (max_frames is None or frame_num < max_frames):
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_num % frame_skip == 0:
                detections = self.process_frame(frame, frame_num)
                all_detections.extend(detections)
                
                # Progress
                if time.time() - last_print > 3:
                    elapsed = time.time() - start_time
                    progress = (frame_num / total_frames) * 100
                    print(f"⏳ {progress:.0f}% | Frame {frame_num}/{total_frames} | "
                          f"Plates found: {len(all_detections)}")
                    last_print = time.time()
                
                if out:
                    for det in detections:
                        x1, y1, x2, y2 = det['bbox']
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, det['plate'], (x1, y1-10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            if out:
                out.write(frame)
            
            frame_num += 1
        
        processing_time = time.time() - start_time
        
        cap.release()
        if out:
            out.release()
        
        unique = self._deduplicate(all_detections)
        
        results = {
            'total_frames': total_frames,
            'processed_frames': frame_num,
            'all_detections': all_detections,
            'unique_plates': unique,
            'processing_time': processing_time,
            'fps': frame_num / processing_time
        }
        
        print(f"\n✅ Done in {processing_time:.1f}s")
        print(f"🎯 Total detections: {len(all_detections)}")
        print(f"🔢 Unique plates: {len(unique)}")
        
        return results
    
    def _deduplicate(self, detections: List[Dict]) -> List[Dict]:
        """Remove duplicates."""
        if not detections:
            return []
        
        unique = []
        seen = set()
        
        for det in sorted(detections, key=lambda x: x['confidence'], reverse=True):
            if det['plate'] not in seen:
                seen.add(det['plate'])
                unique.append(det)
        
        return unique


if __name__ == "__main__":
    alpr = TesseractALPR()
    
    video_path = "data/test_videos/Driving around Hanoi _ Part 5 _ Hanoi _ Vietnam 🇻🇳.mp4"
    
    if Path(video_path).exists():
        results = alpr.process_video(
            video_path,
            output_path="output/videos/tesseract_output.mp4",
            frame_skip=10,
            max_frames=1500  # Test first 1500 frames (~50 seconds)
        )
        
        if results['unique_plates']:
            print("\n" + "="*60)
            print("DETECTED PLATES:")
            print("="*60)
            for i, det in enumerate(results['unique_plates'], 1):
                print(f"{i:2d}. {det['plate']:15s} | {det['vehicle_type']:10s} | "
                      f"Conf: {det['confidence']:.0%} | Frame: {det['frame']}")
            
            # Save to CSV
            import pandas as pd
            df = pd.DataFrame(results['unique_plates'])
            df.to_csv('output/results/tesseract_results.csv', index=False)
            print(f"\n💾 Saved to output/results/tesseract_results.csv")
        else:
            print("\n⚠️  No plates detected")
            print("\n💡 This dashcam footage might have plates that are:")
            print("   - Too small/distant")
            print("   - Too blurry from motion")
            print("   - At difficult angles")
            print("\n📸 Recommendation: Record closer footage (parking lot style)")
    else:
        print(f"Video not found: {video_path}")