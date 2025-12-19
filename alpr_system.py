# Improved Hanoi ALPR - Uses YOLO to detect plates directly
# Install: pip install ultralytics opencv-python paddleocr paddlepaddle numpy pillow

import cv2
import numpy as np
from ultralytics import YOLO
from paddleocr import PaddleOCR
import re
from pathlib import Path
from typing import List, Dict, Tuple
import time

class HanoiALPRImproved:
    """
    Improved ALPR that detects license plates directly instead of 
    extracting from vehicle regions.
    """
    
    def __init__(self, use_custom_plate_model: bool = False):
        """
        Initialize the improved ALPR system.
        
        Args:
            use_custom_plate_model: Use fine-tuned plate detector if available
        """
        print("🚀 Initializing Improved Hanoi ALPR System...")
        
        # For now, we'll use vehicle detection + better region extraction
        # Later you can train a custom plate detector
        print("📦 Loading YOLOv8 model...")
        self.vehicle_model = YOLO('yolov8n.pt')
        
        print("📝 Loading PaddleOCR model...")
        self.reader = PaddleOCR(lang='en')
        
        print("✅ System initialized successfully!\n")
    
    def detect_vehicles(self, frame: np.ndarray) -> List[Dict]:
        """Detect vehicles with lower confidence threshold."""
        vehicle_classes = [2, 3, 7]
        results = self.vehicle_model(frame, verbose=False, conf=0.3)[0]
        
        detections = []
        for box in results.boxes:
            class_id = int(box.cls[0])
            if class_id in vehicle_classes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                
                detections.append({
                    'bbox': (int(x1), int(y1), int(x2), int(y2)),
                    'confidence': conf,
                    'class': 'motorcycle' if class_id == 3 else 'car'
                })
        
        return detections
    
    def find_plate_regions(self, vehicle_crop: np.ndarray, vehicle_type: str) -> List[np.ndarray]:
        """
        Find potential plate regions using multiple strategies.
        Returns list of candidate regions.
        """
        h, w = vehicle_crop.shape[:2]
        regions = []
        
        # Strategy 1: Bottom portion (traditional)
        if vehicle_type == 'car':
            regions.append(vehicle_crop[int(h*0.65):, :])
        else:  # motorcycle
            regions.append(vehicle_crop[int(h*0.5):, :])
        
        # Strategy 2: Use edge detection to find rectangular regions
        gray = cv2.cvtColor(vehicle_crop, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Look for rectangular contours with plate-like aspect ratio
        for contour in contours:
            x, y, cw, ch = cv2.boundingRect(contour)
            
            # Plate aspect ratio typically 2:1 to 5:1
            if cw > 0 and ch > 0:
                aspect_ratio = cw / ch
                area = cw * ch
                
                # Filter for plate-like rectangles
                if 2.0 < aspect_ratio < 6.0 and area > 200:
                    # Add some padding
                    y1 = max(0, y - 5)
                    y2 = min(h, y + ch + 5)
                    x1 = max(0, x - 5)
                    x2 = min(w, x + cw + 5)
                    
                    region = vehicle_crop[y1:y2, x1:x2]
                    if region.shape[0] > 15 and region.shape[1] > 30:
                        regions.append(region)
        
        return regions
    
    def preprocess_for_ocr(self, image: np.ndarray) -> np.ndarray:
        """Aggressive preprocessing for small plates."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Upscale significantly
        h, w = gray.shape
        target_height = 300
        
        if h < target_height:
            scale = target_height / h
            gray = cv2.resize(gray, None, fx=scale, fy=scale, 
                            interpolation=cv2.INTER_CUBIC)
        
        # Contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # Bilateral filter (edge-preserving smoothing)
        filtered = cv2.bilateralFilter(enhanced, 9, 75, 75)
        
        # Adaptive thresholding
        binary = cv2.adaptiveThreshold(
            filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        return binary
    
    def validate_plate_text(self, text: str) -> Tuple[bool, str]:
        """Lenient validation - accepts reasonable plate-like text."""
        text = text.upper().strip()
        # Remove special chars except dash and dot
        text = ''.join(c for c in text if c.isalnum() or c in ['-', '.', ' '])
        text = ' '.join(text.split())
        
        # Basic requirements
        has_numbers = any(c.isdigit() for c in text)
        has_letters = any(c.isalpha() for c in text)
        
        if len(text) >= 5 and has_numbers and has_letters:
            return True, text
        
        # Try OCR error fixes
        fixed = text.replace('O', '0').replace('I', '1').replace('S', '5')
        fixed = fixed.replace('Z', '2').replace('B', '8')
        
        has_numbers = any(c.isdigit() for c in fixed)
        has_letters = any(c.isalpha() for c in fixed)
        
        if len(fixed) >= 5 and has_numbers and has_letters:
            return True, fixed
        
        return False, text
    
    def recognize_plate(self, vehicle_crop: np.ndarray, vehicle_type: str) -> Dict:
        """
        Try OCR on multiple candidate regions and return best result.
        """
        regions = self.find_plate_regions(vehicle_crop, vehicle_type)
        
        best_result = {'text': '', 'confidence': 0.0, 'valid': False}
        
        for region in regions:
            if region.shape[0] < 15 or region.shape[1] < 30:
                continue
            
            # Try original
            result1 = self._ocr_single_region(region)
            if result1['confidence'] > best_result['confidence'] and result1['valid']:
                best_result = result1
            
            # Try preprocessed
            processed = self.preprocess_for_ocr(region)
            processed_bgr = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
            result2 = self._ocr_single_region(processed_bgr)
            if result2['confidence'] > best_result['confidence'] and result2['valid']:
                best_result = result2
        
        return best_result
    
    def _ocr_single_region(self, region: np.ndarray) -> Dict:
        """Run OCR on a single region using new PaddleOCR API."""
        try:
            # Use predict (new API)
            results = self.reader.predict(region)
            
            if not results or len(results) == 0:
                return {'text': '', 'confidence': 0.0, 'valid': False}
            
            # New API format: results[0] contains 'rec_texts' and 'rec_scores'
            result = results[0]
            
            if 'rec_texts' not in result or not result['rec_texts']:
                return {'text': '', 'confidence': 0.0, 'valid': False}
            
            texts = result['rec_texts']
            scores = result['rec_scores']
            
            # Combine all detected text
            combined_text = ' '.join(texts)
            avg_conf = sum(scores) / len(scores) if scores else 0.0
            
            is_valid, formatted = self.validate_plate_text(combined_text)
            
            return {
                'text': formatted,
                'confidence': avg_conf,
                'valid': is_valid
            }
        except Exception as e:
            print(f"OCR Error: {e}")
            return {'text': '', 'confidence': 0.0, 'valid': False}
    
    def process_frame(self, frame: np.ndarray, frame_num: int) -> List[Dict]:
        """Process frame with improved plate detection."""
        detections = []
        
        vehicles = self.detect_vehicles(frame)
        
        for vehicle in vehicles:
            x1, y1, x2, y2 = vehicle['bbox']
            vehicle_crop = frame[y1:y2, x1:x2]
            
            if vehicle_crop.shape[0] < 30 or vehicle_crop.shape[1] < 30:
                continue
            
            plate_result = self.recognize_plate(vehicle_crop, vehicle['class'])
            
            # Very lenient threshold
            if plate_result['valid'] and plate_result['confidence'] > 0.15:
                detections.append({
                    'frame': frame_num,
                    'plate': plate_result['text'],
                    'confidence': plate_result['confidence'],
                    'vehicle_type': vehicle['class'],
                    'bbox': vehicle['bbox']
                })
        
        return detections
    
    def process_video(self, video_path: str, output_path: str = None, 
                     frame_skip: int = 5) -> Dict:
        """Process video with improved detection."""
        print(f"🎥 Processing video: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"📊 Video: {total_frames} frames, {fps} FPS, {width}x{height}")
        print(f"⚙️  Processing every {frame_skip} frames")
        
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        all_detections = []
        frame_num = 0
        start_time = time.time()
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_num % frame_skip == 0:
                detections = self.process_frame(frame, frame_num)
                all_detections.extend(detections)
                
                if frame_num % 100 == 0:
                    print(f"⏳ Frame {frame_num}/{total_frames} - Found {len(all_detections)} plates so far")
                
                if out:
                    for det in detections:
                        x1, y1, x2, y2 = det['bbox']
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        label = f"{det['plate']} ({det['confidence']:.2f})"
                        cv2.putText(frame, label, (x1, y1-10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            if out:
                out.write(frame)
            
            frame_num += 1
        
        processing_time = time.time() - start_time
        
        cap.release()
        if out:
            out.release()
        
        unique_plates = self._deduplicate_detections(all_detections)
        
        results = {
            'total_frames': total_frames,
            'processed_frames': frame_num,
            'all_detections': all_detections,
            'unique_plates': unique_plates,
            'processing_time': processing_time,
            'fps': frame_num / processing_time
        }
        
        print(f"\n✅ Processing complete!")
        print(f"⏱️  Time: {processing_time:.2f}s ({results['fps']:.1f} FPS)")
        print(f"🎯 Total detections: {len(all_detections)}")
        print(f"🔢 Unique plates: {len(unique_plates)}")
        
        return results
    
    def _deduplicate_detections(self, detections: List[Dict], 
                               frame_threshold: int = 50) -> List[Dict]:
        """Remove duplicates."""
        if not detections:
            return []
        
        sorted_dets = sorted(detections, key=lambda x: x['frame'])
        unique = []
        current_plate = None
        current_group = []
        
        for det in sorted_dets:
            if current_plate == det['plate']:
                if det['frame'] - current_group[-1]['frame'] <= frame_threshold:
                    current_group.append(det)
                else:
                    best = max(current_group, key=lambda x: x['confidence'])
                    unique.append(best)
                    current_group = [det]
            else:
                if current_group:
                    best = max(current_group, key=lambda x: x['confidence'])
                    unique.append(best)
                current_plate = det['plate']
                current_group = [det]
        
        if current_group:
            best = max(current_group, key=lambda x: x['confidence'])
            unique.append(best)
        
        return unique


if __name__ == "__main__":
    alpr = HanoiALPRImproved()
    
    video_path = "test_video.mp4"
    
    if Path(video_path).exists():
        results = alpr.process_video(
            video_path,
            output_path="output_improved.mp4",
            frame_skip=3
        )
        
        print("\n" + "="*60)
        print("DETECTED UNIQUE PLATES:")
        print("="*60)
        for i, det in enumerate(results['unique_plates'], 1):
            print(f"{i}. {det['plate']} - Frame {det['frame']} - "
                  f"{det['vehicle_type']} - Confidence: {det['confidence']:.2f}")