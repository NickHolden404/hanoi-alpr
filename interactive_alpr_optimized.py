# Optimized Interactive ALPR - Better settings for more detections
import cv2
import numpy as np
from ultralytics import YOLO
from paddleocr import PaddleOCR
from pathlib import Path
from typing import List, Dict
import time

class OptimizedVietnameseALPR:
    """Optimized ALPR with better detection settings."""
    
    def __init__(self):
        print("🚀 Initializing Optimized Vietnamese ALPR System...")
        self.vehicle_model = YOLO('yolov8n.pt')
        self.reader = PaddleOCR(lang='en')
        
        # Vietnamese plate patterns (old and new formats)
        self.plate_patterns = [
            r'^\d{2}[A-Z]{1,2}\d?[-\s]?\d{3}[\.\s]?\d{2}$',  # New: 30E-223.17
            r'^\d{2}[A-Z]{1,2}[-\s]?\d{4}$',                  # Old: 30E-9296
        ]
        
        print("✅ Ready!\n")
    
    def detect_vehicles(self, frame: np.ndarray) -> List[Dict]:
        """Detect vehicles - more lenient for better coverage."""
        results = self.vehicle_model(frame, verbose=False, conf=0.30)[0]  # Lowered from 0.35
        
        detections = []
        for box in results.boxes:
            class_id = int(box.cls[0])
            if class_id in [2, 3, 7]:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                w, h = x2 - x1, y2 - y1
                area = w * h
                
                # More lenient - accept slightly smaller vehicles
                if area > 5000:  # Lowered from 6000
                    detections.append({
                        'bbox': (int(x1), int(y1), int(x2), int(y2)),
                        'area': area,
                        'class': 'motorcycle' if class_id == 3 else 'car',
                        'confidence': float(box.conf[0])
                    })
        
        detections.sort(key=lambda x: x['area'], reverse=True)
        return detections
    
    def extract_plate_regions(self, vehicle_crop: np.ndarray, vehicle_type: str) -> List[np.ndarray]:
        """Extract multiple candidate plate regions."""
        h, w = vehicle_crop.shape[:2]
        regions = []
        
        # Region 1: Bottom portion (most common)
        if vehicle_type == 'car':
            regions.append(vehicle_crop[int(h*0.55):, :])
        else:
            regions.append(vehicle_crop[int(h*0.50):, :])
        
        # Region 2: Detect plate-like rectangles
        gray = cv2.cvtColor(vehicle_crop, cv2.COLOR_BGR2GRAY)
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            x, y, cw, ch = cv2.boundingRect(contour)
            
            if cw > 0 and ch > 0:
                aspect = cw / ch
                area = cw * ch
                
                if 2.3 < aspect < 5.0 and 800 < area < 30000:
                    y1 = max(0, y - 8)
                    y2 = min(h, y + ch + 8)
                    x1 = max(0, x - 8)
                    x2 = min(w, x + cw + 8)
                    
                    region = vehicle_crop[y1:y2, x1:x2]
                    if region.shape[0] > 25 and region.shape[1] > 60:
                        regions.append(region)
        
        return regions[:4]
    
    def preprocess(self, image: np.ndarray) -> List[np.ndarray]:
        """Preprocess - return multiple versions."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Upscale
        h, w = gray.shape
        if h < 120:
            scale = 120 / h
            gray = cv2.resize(gray, None, fx=scale, fy=scale, 
                            interpolation=cv2.INTER_CUBIC)
        
        versions = []
        
        # Version 1: Denoised + enhanced
        denoised = cv2.fastNlMeansDenoising(gray, h=10)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(denoised)
        versions.append(enhanced)
        
        # Version 2: Sharpened
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)
        versions.append(sharpened)
        
        # Version 3: Binary threshold
        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        versions.append(binary)
        
        return versions
    
    def format_vietnamese_plate(self, text: str) -> tuple:
        """Format text to match Vietnamese plate format (both old and new)."""
        text = text.upper().strip()
        cleaned = ''.join(c for c in text if c.isalnum())
        
        # OCR error corrections
        cleaned = cleaned.replace('O', '0').replace('I', '1').replace('S', '5')
        cleaned = cleaned.replace('Z', '2').replace('B', '8').replace('G', '6')
        
        has_nums = any(c.isdigit() for c in cleaned)
        has_lets = any(c.isalpha() for c in cleaned)
        
        if not (has_nums and has_lets) or len(cleaned) < 6:
            return False, text
        
        digits = ''.join(c for c in cleaned if c.isdigit())
        letters = ''.join(c for c in cleaned if c.isalpha())
        
        if len(digits) < 6 or len(letters) < 1:
            return False, text
        
        province = digits[:2]
        series = letters[:2] if len(letters) >= 2 else letters
        remaining = digits[2:]
        
        # Old format: 4 digits
        if len(remaining) == 4:
            formatted = f"{province}{series}-{remaining}"
            return True, formatted
        
        # New format: 5+ digits
        elif len(remaining) >= 5:
            number = remaining[:3]
            suffix = remaining[3:5]
            formatted = f"{province}{series}-{number}.{suffix}"
            return True, formatted
        
        return False, cleaned
    
    def validate_plate(self, text: str) -> tuple:
        """Validate and format plate."""
        is_valid, formatted = self.format_vietnamese_plate(text)
        
        if not is_valid:
            return False, text
        
        if len(formatted) < 8 or len(formatted) > 15:
            return False, text
        
        # Filter brand names
        garbage = ['SUZUK', 'YAMAHA', 'HONDA', 'VELZ', 'TOYOTA']
        if any(g in formatted.upper() for g in garbage):
            return False, text
        
        return True, formatted
    
    def ocr_region(self, region: np.ndarray) -> Dict:
        """OCR with multiple preprocessing attempts."""
        best_result = {'text': '', 'confidence': 0.0, 'valid': False}
        
        try:
            versions = self.preprocess(region)
            
            for version in versions:
                version_bgr = cv2.cvtColor(version, cv2.COLOR_GRAY2BGR)
                
                results = self.reader.predict(version_bgr)
                
                if not results or len(results) == 0:
                    continue
                
                result = results[0]
                
                if 'rec_texts' not in result or not result['rec_texts']:
                    continue
                
                texts = result['rec_texts']
                scores = result['rec_scores']
                
                combined = ''.join(texts)
                avg_conf = sum(scores) / len(scores) if scores else 0.0
                
                is_valid, formatted = self.validate_plate(combined)
                
                if is_valid and avg_conf > best_result['confidence']:
                    best_result = {
                        'text': formatted,
                        'confidence': avg_conf,
                        'valid': True
                    }
        
        except Exception as e:
            pass
        
        return best_result
    
    def process_frame(self, frame: np.ndarray, frame_num: int) -> List[Dict]:
        """Process frame with optimized settings."""
        detections = []
        
        vehicles = self.detect_vehicles(frame)
        
        # Process more vehicles per frame
        for i, vehicle in enumerate(vehicles[:8]):  # Increased from 6 to 8
            x1, y1, x2, y2 = vehicle['bbox']
            vehicle_crop = frame[y1:y2, x1:x2]
            
            if vehicle_crop.shape[0] < 50 or vehicle_crop.shape[1] < 50:
                continue
            
            regions = self.extract_plate_regions(vehicle_crop, vehicle['class'])
            
            best_result = {'text': '', 'confidence': 0.0, 'valid': False}
            
            for region in regions:
                if region.shape[0] < 25 or region.shape[1] < 50:
                    continue
                
                result = self.ocr_region(region)
                
                if result['valid'] and result['confidence'] > best_result['confidence']:
                    best_result = result
                
                if result['confidence'] > 0.85:
                    break
            
            # More lenient threshold for cars (they have larger, clearer plates)
            min_confidence = 0.35 if vehicle['class'] == 'car' else 0.40
            
            if best_result['valid'] and best_result['confidence'] > min_confidence:
                detections.append({
                    'frame': frame_num,
                    'plate': best_result['text'],
                    'confidence': best_result['confidence'],
                    'vehicle_type': vehicle['class'],
                    'bbox': vehicle['bbox']
                })
        
        return detections
    
    def process_video(self, video_path: str, frame_skip: int = 5, 
                     max_frames: int = 3000) -> Dict:
        """
        Process video with optimized settings.
        
        Args:
            frame_skip: Process every Nth frame (default 5, lower = more plates)
            max_frames: Maximum frames to process (default 3000)
        """
        print(f"\n🎥 Processing: {Path(video_path).name}")
        
        cap = cv2.VideoCapture(video_path)
        total_frames = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), max_frames)
        
        print(f"📊 Processing {total_frames} frames (every {frame_skip}th)")
        print(f"⏱️  Estimated time: {total_frames / frame_skip * 2:.0f}s ({total_frames / frame_skip * 2 / 60:.0f} min)\n")
        
        all_detections = []
        frame_num = 0
        start_time = time.time()
        
        while frame_num < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_num % frame_skip == 0:
                detections = self.process_frame(frame, frame_num)
                all_detections.extend(detections)
                
                if frame_num % 200 == 0:
                    elapsed = time.time() - start_time
                    print(f"⏳ Frame {frame_num}/{total_frames} | "
                          f"Time: {elapsed:.0f}s | Plates: {len(all_detections)}")
            
            frame_num += 1
        
        cap.release()
        
        processing_time = time.time() - start_time
        
        unique = self._smart_deduplicate(all_detections)
        
        results = {
            'total_frames': total_frames,
            'all_detections': all_detections,
            'unique_plates': unique,
            'processing_time': processing_time
        }
        
        print(f"\n✅ Done in {processing_time:.0f}s")
        print(f"🎯 Total detections: {len(all_detections)}")
        print(f"🔢 Unique plates: {len(unique)}")
        
        return results
    
    def _smart_deduplicate(self, detections: List[Dict]) -> List[Dict]:
        """Smart deduplication using similarity."""
        if not detections:
            return []
        
        from difflib import SequenceMatcher
        
        def similar(a, b):
            return SequenceMatcher(None, a, b).ratio()
        
        sorted_dets = sorted(detections, key=lambda x: x['confidence'], reverse=True)
        
        unique = []
        seen = set()
        
        for det in sorted_dets:
            plate = det['plate']
            
            is_duplicate = False
            for seen_plate in seen:
                if similar(plate, seen_plate) > 0.75:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique.append(det)
                seen.add(plate)
        
        return unique


def list_videos():
    """List all available videos."""
    video_dir = Path("data/test_videos")
    videos = sorted(video_dir.glob("*.mp4"))
    return videos

def get_video_quality_badge(video_name: str) -> str:
    """Return quality badge based on previous results."""
    # Based on your testing results
    excellent = ['DRIVING HA NOI 4K', 'Monday in Hanoi', 'Part 5', '6mins in Hanoi', 
                'Hanoi,Vietnam dashcam  overtaking', 'Traffic in Hanoi']
    good = ['Truong Sa', 'Chaotic Traffic', 'Hanoi Driving Record']
    
    for keyword in excellent:
        if keyword.lower() in video_name.lower():
            return "⭐⭐⭐⭐⭐"
    
    for keyword in good:
        if keyword.lower() in video_name.lower():
            return "⭐⭐⭐"
    
    return "⭐⭐"

def interactive_menu():
    """Interactive menu with optimized processing."""
    alpr = None
    
    while True:
        print("\n" + "=" * 70)
        print("🎬 OPTIMIZED INTERACTIVE ALPR TESTING")
        print("=" * 70)
        print("\n💡 Optimized settings: frame_skip=5, max_frames=3000")
        print("   (Process more frames for better coverage)\n")
        
        videos = list_videos()
        
        if not videos:
            print("❌ No videos found in data/test_videos/")
            return
        
        print("Available videos:\n")
        for i, video in enumerate(videos, 1):
            size_mb = video.stat().st_size / (1024 * 1024)
            quality = get_video_quality_badge(video.name)
            print(f"  {i:2d}. {quality} {video.name[:50]:50s} ({size_mb:.1f} MB)")
        
        print("\n  0. Exit")
        print("\n💡 Tip: Choose ⭐⭐⭐⭐⭐ videos for best results!")
        
        choice = input("\nChoose video number to test: ").strip()
        
        if choice == '0':
            print("\n👋 Goodbye!")
            break
        
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(videos):
                video_path = videos[idx]
                
                # Ask for custom settings
                print("\n⚙️  Processing Settings:")
                print("   Default: frame_skip=5, max_frames=3000 (recommended)")
                custom = input("   Use custom settings? (y/N): ").strip().lower()
                
                frame_skip = 5
                max_frames = 3000
                
                if custom == 'y':
                    try:
                        frame_skip = int(input("   Frame skip (1-10, lower=more plates): ") or "5")
                        max_frames = int(input("   Max frames (1500-5000): ") or "3000")
                    except ValueError:
                        print("   Invalid input, using defaults")
                
                # Initialize ALPR if needed
                if alpr is None:
                    print()
                    alpr = OptimizedVietnameseALPR()
                
                # Process video
                results = alpr.process_video(
                    str(video_path),
                    frame_skip=frame_skip,
                    max_frames=max_frames
                )
                
                # Show results
                if results['unique_plates']:
                    print("\n" + "=" * 70)
                    print("✅ DETECTED PLATES:")
                    print("=" * 70)
                    print(f"{'#':<4} {'Plate':<18} {'Type':<12} {'Conf':<8} {'Frame':<8}")
                    print("-" * 70)
                    
                    for i, det in enumerate(results['unique_plates'], 1):
                        print(f"{i:<4} {det['plate']:<18} {det['vehicle_type']:<12} "
                              f"{det['confidence']:.0%}{'':>3} {det['frame']:<8}")
                    
                    # Save
                    import pandas as pd
                    df = pd.DataFrame(results['unique_plates'])
                    csv_name = f"optimized_results_{video_path.stem}.csv"
                    df.to_csv(f'output/results/{csv_name}', index=False)
                    print(f"\n💾 Saved to output/results/{csv_name}")
                    
                    # Show improvement if previous results exist
                    old_csv = Path(f"output/results/results_{video_path.stem}.csv")
                    if old_csv.exists():
                        old_df = pd.read_csv(old_csv)
                        improvement = len(df) - len(old_df)
                        if improvement > 0:
                            print(f"📈 Improvement: +{improvement} plates from previous run!")
                        elif improvement == 0:
                            print(f"📊 Same as previous run ({len(df)} plates)")
                else:
                    print("\n⚠️  No plates detected in this video")
                
                input("\nPress Enter to continue...")
            
            else:
                print("❌ Invalid choice!")
        
        except ValueError:
            print("❌ Please enter a valid number!")
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    interactive_menu()