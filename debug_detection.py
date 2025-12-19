# debug_detection.py
import cv2
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
    # Initialize system
    alpr = HanoiALPR()
    
    # Test vehicle detection only
    print("🔍 Testing vehicle detection...")
    vehicles = alpr.detect_vehicles(frame)
    print(f"✅ Found {len(vehicles)} vehicles")
    
    for i, v in enumerate(vehicles, 1):
        print(f"  {i}. {v['class']:10s} - Confidence: {v['confidence']:.2%}")
        print(f"     BBox: {v['bbox']}")
    
    # Test full pipeline
    print("\n🔍 Testing full pipeline...")
    detections = alpr.process_frame(frame, 0)
    print(f"✅ Found {len(detections)} plates")
    
    for det in detections:
        print(f"  Plate: {det['plate']} - Confidence: {det['confidence']:.2%}")
    
    # Save annotated frame for inspection
    for vehicle in vehicles:
        x1, y1, x2, y2 = vehicle['bbox']
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
    
    cv2.imwrite("debug_frame.jpg", frame)
    print("\n💾 Saved debug_frame.jpg - check this to see if vehicles are detected")
else:
    print("❌ Could not read frame")