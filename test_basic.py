#!/usr/bin/env python3
"""
Quick test script for Hanoi ALPR system
Usage: python test_basic.py
"""

from alpr_system import HanoiALPRImproved
from pathlib import Path
import sys

def main():
    print("=" * 60)
    print("🚗 Hanoi ALPR System - Quick Test")
    print("=" * 60)
    
    # Find test video
    video_dir = Path("data/test_videos")
    videos = list(video_dir.glob("*.mp4")) + list(video_dir.glob("*.mov")) + list(video_dir.glob("*.avi"))
    
    if not videos:
        print("\n❌ No test videos found!")
        print("\nPlease add a video to data/test_videos/")
        print("You can:")
        print("  1. Record traffic with your phone")
        print("  2. Download from YouTube (Hanoi traffic dashcam)")
        print("  3. Use any parking lot footage")
        sys.exit(1)
    
    video_path = str(videos[0])
    print(f"\n📹 Found test video: {Path(video_path).name}")
    
    # Initialize ALPR
    print("\n🚀 Initializing ALPR system...")
    alpr = HanoiALPRImproved()
    
    # Process video
    print(f"\n⏳ Processing video...")
    results = alpr.process_video(
        video_path,
        output_path="output/videos/annotated_output.mp4",
        frame_skip=3
    )
    
    # Display results
    print("\n" + "=" * 60)
    print("✅ PROCESSING COMPLETE")
    print("=" * 60)
    print(f"Time: {results['processing_time']:.2f}s")
    print(f"Speed: {results['fps']:.1f} FPS")
    print(f"Total detections: {len(results['all_detections'])}")
    print(f"Unique plates: {len(results['unique_plates'])}")
    
    if results['unique_plates']:
        print("\n" + "=" * 60)
        print("DETECTED PLATES:")
        print("=" * 60)
        for i, det in enumerate(results['unique_plates'], 1):
            print(f"{i:2d}. {det['plate']:15s} | {det['vehicle_type']:10s} | "
                  f"Confidence: {det['confidence']:.2%} | Frame: {det['frame']}")
        
        # Save to CSV
        import pandas as pd
        df = pd.DataFrame(results['unique_plates'])
        df.to_csv('output/results/detected_plates.csv', index=False)
        print(f"\n💾 Results saved to output/results/detected_plates.csv")
    else:
        print("\n⚠️  No plates detected")
        print("This could mean:")
        print("  - Video quality is low")
        print("  - No vehicles with visible plates")
        print("  - Try adjusting frame_skip or confidence thresholds")
    
    print("\n✅ Check output/videos/annotated_output.mp4 for annotated video")

if __name__ == "__main__":
    main()
