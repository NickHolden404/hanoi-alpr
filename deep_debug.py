# deep_debug.py
import cv2
from alpr_system import HanoiALPRImproved
from pathlib import Path

def deep_debug():
    """Ultra-detailed debugging to see what's going wrong."""
    
    # Use the best video
    video_path = "data/test_videos/Driving around Hanoi _ Part 5 _ Hanoi _ Vietnam 🇻🇳.mp4"
    
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Get frame from middle
    cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("❌ Could not read frame")
        return
    
    print("🔍 DEEP DEBUG - Step by Step\n")
    
    alpr = HanoiALPRImproved()
    
    # Step 1: Detect vehicles
    print("=" * 60)
    print("STEP 1: Vehicle Detection")
    print("=" * 60)
    vehicles = alpr.detect_vehicles(frame)
    print(f"Found {len(vehicles)} vehicles\n")
    
    # Test first 3 vehicles in detail
    for i, vehicle in enumerate(vehicles[:3], 1):
        print(f"\n{'='*60}")
        print(f"VEHICLE {i}: {vehicle['class']} (confidence: {vehicle['confidence']:.2%})")
        print(f"{'='*60}")
        
        x1, y1, x2, y2 = vehicle['bbox']
        vehicle_crop = frame[y1:y2, x1:x2]
        
        print(f"Vehicle size: {vehicle_crop.shape}")
        
        # Step 2: Find plate regions
        print(f"\nSTEP 2: Finding plate regions...")
        regions = alpr.find_plate_regions(vehicle_crop, vehicle['class'])
        print(f"Found {len(regions)} candidate regions")
        
        # Step 3: Try OCR on each region
        print(f"\nSTEP 3: Testing OCR on each region...")
        
        for j, region in enumerate(regions[:5], 1):  # Test first 5 regions
            print(f"\n  Region {j}: size {region.shape}")
            
            if region.shape[0] < 15 or region.shape[1] < 30:
                print(f"  ❌ Too small, skipping")
                continue
            
            # Save the region
            cv2.imwrite(f"debug_v{i}_region{j}_original.jpg", region)
            
            # Try OCR on original
            print(f"  Testing original image...")
            result1 = alpr._ocr_single_region(region)
            print(f"    Text: '{result1['text']}' | Conf: {result1['confidence']:.2%} | Valid: {result1['valid']}")
            
            # Try preprocessed
            processed = alpr.preprocess_for_ocr(region)
            cv2.imwrite(f"debug_v{i}_region{j}_processed.jpg", processed)
            
            processed_bgr = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
            print(f"  Testing preprocessed image...")
            result2 = alpr._ocr_single_region(processed_bgr)
            print(f"    Text: '{result2['text']}' | Conf: {result2['confidence']:.2%} | Valid: {result2['valid']}")
        
        # Draw numbered vehicle on frame
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(frame, f"#{i}", (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
    
    cv2.imwrite("debug_numbered_vehicles.jpg", frame)
    
    print(f"\n{'='*60}")
    print("✅ Debug complete!")
    print("=" * 60)
    print("\nCheck these files:")
    print("  - debug_numbered_vehicles.jpg (see which vehicles were tested)")
    print("  - debug_v{N}_region{N}_*.jpg (extracted regions)")
    print("\nLook at the extracted regions - do they contain license plates?")

if __name__ == "__main__":
    deep_debug()