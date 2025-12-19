# Fast Hanoi ALPR - Optimized for speed
import cv2
import numpy as np
from ultralytics import YOLO
from paddleocr import PaddleOCR
import re
from pathlib import Path
from typing import List, Dict, Tuple
import time

class FastHanoiALPR:
    """Optimized ALPR focused on speed."""
    
    def __init__(self):
        print("🚀 Initializing Fast Hanoi ALPR System...")
        print("📦 Loading YOLOv8 model...")
        self.vehicle_model = YOLO('yolov8n.pt')
        
        print("📝 Loading PaddleOCR model (this takes a moment)...")
        self.reader = PaddleOCR(lang='en')
        print("✅ System initialized!\n")
    
    def detect_vehicles(self, frame: np.ndarray) -> List[Dict]:
        """Fast vehicle detection."""
        results = self.vehicle_model(frame, verbose=False, conf=0.4)[0]
        
        detections = []
        for box in results.boxes:
            class_id = int(box.cls[0])
            if class_id in [2, 3, 7]:  # car, motorcycle, truck
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                w = x2 - x1
                h = y2 - y1
                
                # Skip very small vehicles (plates will be unreadable anyway)
                if w < 80 or h < 80:
                    continue
                
                detections.append({
                    'bbox': (int(x1), int(y1), int(x2), int(y2)),
                    'confidence': float(box.conf[0]),
                    'class': 'motorcycle' if class_id == 3 else 'car'
                })
        
        return detections
    
    def extract_plate_candidates(self, vehicle_crop: np.ndarray) -> List[np.ndarray]:
        """Extract likely plate regions using simple heuristics."""
        h, w = vehicle_crop.shape[:2]
        candidates = []
        
        # Region 1: Bottom 40% (most common location)
        bottom_region = vehicle_crop[int(h*0.6):, :]
        if bottom_region.shape[0] > 20 and bottom_region.shape[1] > 40:
            candidates.append(bottom_region)
        
        # Region 2: Use edge detection to find rectangles
        gray = cv2.cvtColor(vehicle_crop, cv2.COLOR_BGR2GRAY)
        
        # Simple threshold instead of Canny (faster)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours[:10]:  # Only check top 10 contours
            x, y, cw, ch = cv2.boundingRect(contour)
            
            if cw > 0 and ch > 0:
                aspect_ratio = cw / ch
                area = cw * ch
                
                # Plate-like: wide rectangle, reasonable size
                if 2.5 < aspect_ratio < 5.5 and 500 < area < 50000:
                    y1 = max(0, y - 5)
                    y2 = min(h, y + ch + 5)
                    x1 = max(0, x - 5)
                    x2 = min(w, x + cw + 5)
                    
                    region = vehicle_crop[y1:y2, x1:x2]
                    if region.shape[0] > 20 and region.shape[1] > 40:
                        candidates.append(region)
        
        return candidates[:3]  # Max 3 candidates per vehicle
    
    def preprocess_fast(self, image: np.ndarray) -> np.ndarray:
        """Fast preprocessing."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Upscale if small
        h, w = gray.shape
        if h < 100:
            scale = 100 / h
            gray = cv2.resize(gray, None, fx=scale, fy=scale, 
                            interpolation=cv2.INTER_LINEAR)  # LINEAR is faster
        
        # Simple contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        return enhanced
    
    def validate_plate(self, text: str) -> Tuple[bool, str]:
        """Quick validation."""
        text = text.upper().strip()
        text = ''.join(c for c in text if c.isalnum() or c in ['-', '.', ' '])
        
        has_numbers = any(c.isdigit() for c in text)
        has_letters = any(c.isalpha() for c in text)
        
        # Very lenient: just needs some numbers and letters
        if len(text) >= 4 and has_numbers and has_letters:
            return True, text
        
        return False, text
    
    def ocr_region(self, region: np.ndarray) -> Dict:
        """OCR with timeout protection."""
        try:
            # Preprocess
            processed = self.preprocess_fast(region)
            processed_bgr = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
            
            # OCR
            results = self.reader.predict(processed_bgr)
            
            if not results or len(results) == 0:
                return {'text': '', 'confidence': 0.0, 'valid': False}
            
            result = results[0]
            
            if 'rec_texts' not in result or not result['rec_texts']:
                return {'text': '', 'confidence': 0.0, 'valid': False}
            
            texts = result['rec_texts']
            scores = result['rec_scores']
            
            combined_text = ' '.join(texts)
            avg_conf = sum(scores) / len(scores) if scores else 0.0
            
            is_valid, formatted = self.validate_plate(combined_text)
            
            return {
                'text': formatted,
                'confidence': avg_conf,
                'valid': is_valid
            }
        except Exception as e:
            return {'text': '', 'confidence': 0.0, 'valid': False}
    
    def process_frame(self, frame: np.ndarray, frame_num: int) -> List[Dict]:
        """Process single frame."""
        detections = []
        
        vehicles = self.detect_vehicles(frame)
        
        for vehicle in vehicles[:5]:  # Max 5 vehicles per frame for speed
            x1, y1, x2, y2 = vehicle['bbox']
            vehicle_crop = frame[y1:y2, x1:x2]
            
            if vehicle_crop.shape[0] < 40 or vehicle_crop.shape[1] < 40:
                continue
            
            # Get candidates
            candidates = self.extract_plate_candidates(vehicle_crop)
            
            best_result = {'text': '', 'confidence': 0.0, 'valid': False}
            
            # Try each candidate, keep best
            for candidate in candidates:
                result = self.ocr_region(candidate)
                
                if result['valid'] and result['confidence'] > best_result['confidence']:
                    best_result = result
                
                # If we found a good one, stop
                if result['confidence'] > 0.8:
                    break
            
            # Accept if valid and reasonable confidence
            if best_result['valid'] and best_result['confidence'] > 0.3:
                detections.append({
                    'frame': frame_num,
                    'plate': best_result['text'],
                    'confidence': best_result['confidence'],
                    'vehicle_type': vehicle['class'],
                    'bbox': vehicle['bbox']
                })
        
        return detections
    
    def process_video(self, video_path: str, output_path: str = None, 
                     frame_skip: int = 10, max_frames: int = None) -> Dict:
        """Process video with optimizations."""
        print(f"🎥 Processing video: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if max_frames:
            total_frames = min(total_frames, max_frames)
        
        print(f"📊 Video: {total_frames} frames, {fps} FPS")
        print(f"⚙️  Processing every {frame_skip} frames")
        print(f"⏱️  Estimated time: {total_frames / frame_skip * 2:.0f}s\n")
        
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
                
                # Progress every 5 seconds
                if time.time() - last_print > 5:
                    elapsed = time.time() - start_time
                    progress = (frame_num / total_frames) * 100
                    speed = frame_num / elapsed if elapsed > 0 else 0
                    print(f"⏳ {progress:.1f}% | Frame {frame_num}/{total_frames} | "
                          f"{speed:.1f} FPS | Plates: {len(all_detections)}")
                    last_print = time.time()
                
                if out:
                    for det in detections:
                        x1, y1, x2, y2 = det['bbox']
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        label = f"{det['plate']}"
                        cv2.putText(frame, label, (x1, y1-10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            if out:
                out.write(frame)
            
            frame_num += 1
        
        processing_time = time.time() - start_time
        
        cap.release()
        if out:
            out.release()
        
        unique_plates = self._deduplicate(all_detections)
        
        results = {
            'total_frames': total_frames,
            'processed_frames': frame_num,
            'all_detections': all_detections,
            'unique_plates': unique_plates,
            'processing_time': processing_time,
            'fps': frame_num / processing_time
        }
        
        print(f"\n✅ Complete!")
        print(f"⏱️  Time: {processing_time:.2f}s ({results['fps']:.1f} FPS)")
        print(f"🎯 Detections: {len(all_detections)}")
        print(f"🔢 Unique plates: {len(unique_plates)}")
        
        return results
    
    def _deduplicate(self, detections: List[Dict]) -> List[Dict]:
        """Remove duplicates."""
        if not detections:
            return []
        
        sorted_dets = sorted(detections, key=lambda x: x['frame'])
        unique = []
        seen = set()
        
        for det in sorted_dets:
            key = det['plate']
            if key not in seen:
                seen.add(key)
                unique.append(det)
        
        return unique


if __name__ == "__main__":
    alpr = FastHanoiALPR()
    
    # Test on first 1000 frames only for speed
    video_path = "data/test_videos/Driving around Hanoi _ Part 5 _ Hanoi _ Vietnam 🇻🇳.mp4"
    
    if Path(video_path).exists():
        results = alpr.process_video(
            video_path,
            output_path="output/videos/fast_test.mp4",
            frame_skip=15,  # Every 15th frame
            max_frames=1000  # Just first 1000 frames
        )
        
        print("\n" + "="*60)
        print("DETECTED PLATES:")
        print("="*60)
        for i, det in enumerate(results['unique_plates'], 1):
            print(f"{i}. {det['plate']} - Frame {det['frame']} - "
                  f"{det['vehicle_type']} - Conf: {det['confidence']:.2%}")