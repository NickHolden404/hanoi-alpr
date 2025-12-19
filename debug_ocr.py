# debug_ocr.py
import cv2
import numpy as np
from alpr_system import HanoiALPR

# Load video
video_path = "data/test_videos/Post-lockdown traffic - Hanoi driving.mp4"
cap = cv2.VideoCapture(video_path)

# Get a frame from middle of video
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)
ret, frame = cap.read()
cap.release()

if ret:
    alpr = HanoiALPR()
    
    # Get vehicles
    vehicles = alpr.detect_vehicles(frame)
    print(f"Found {len(vehicles)} vehicles\n")
    
    # Test plate extraction and OCR for each vehicle
    for i, vehicle in enumerate(vehicles[:5], 1):  # Test first 5 vehicles
        print(f"\n{'='*60}")
        print(f"Vehicle {i}: {vehicle['class']} (conf: {vehicle['confidence']:.2%})")
        print(f"{'='*60}")
        
        # Extract plate region
        plate_region = alpr.extract_plate_region(frame, vehicle['bbox'])
        print(f"Plate region size: {plate_region.shape}")
        
        # Save plate region
        cv2.imwrite(f"debug_plate_{i}_original.jpg", plate_region)
        
        # Preprocess
        processed = alpr.preprocess_for_ocr(plate_region)
        cv2.imwrite(f"debug_plate_{i}_processed.jpg", processed)
        
        # Try OCR
        if plate_region.shape[0] >= 20 and plate_region.shape[1] >= 20:
            plate_result = alpr.recognize_plate(plate_region)
            print(f"OCR Text: '{plate_result['text']}'")
            print(f"Confidence: {plate_result['confidence']:.2%}")
            print(f"Valid format: {plate_result['valid']}")
            
            # Draw vehicle bbox and number
            x1, y1, x2, y2 = vehicle['bbox']
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"#{i}", (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            print("Plate region too small!")
    
    cv2.imwrite("debug_vehicles_numbered.jpg", frame)
    print(f"\n{'='*60}")
    print("✅ Check these files:")
    print("  - debug_vehicles_numbered.jpg (shows which vehicle is which)")
    print("  - debug_plate_X_original.jpg (extracted plate regions)")
    print("  - debug_plate_X_processed.jpg (after preprocessing)")
    print(f"{'='*60}")