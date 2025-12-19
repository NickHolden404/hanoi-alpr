# test_multiple_videos.py
from alpr_system import HanoiALPRImproved
from pathlib import Path
import sys

def list_videos():
    """List all videos in test directory."""
    video_dir = Path("data/test_videos")
    video_extensions = ['.mp4', '.mov', '.avi', '.mkv']
    
    videos = []
    for ext in video_extensions:
        videos.extend(video_dir.glob(f"*{ext}"))
    
    return sorted(videos)

def main():
    videos = list_videos()
    
    if not videos:
        print("❌ No videos found in data/test_videos/")
        sys.exit(1)
    
    print("=" * 60)
    print("🎥 Available Videos:")
    print("=" * 60)
    for i, video in enumerate(videos, 1):
        size_mb = video.stat().st_size / (1024 * 1024)
        print(f"{i}. {video.name} ({size_mb:.1f} MB)")
    
    # Let user choose
    print("\n" + "=" * 60)
    choice = input("Enter video number to test (or press Enter for all): ").strip()
    
    if choice == "":
        selected_videos = videos
    else:
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(videos):
                selected_videos = [videos[idx]]
            else:
                print("Invalid choice!")
                sys.exit(1)
        except ValueError:
            print("Invalid input!")
            sys.exit(1)
    
    # Initialize ALPR once
    print("\n🚀 Initializing ALPR system...")
    alpr = HanoiALPRImproved()
    
    # Process selected videos
    all_results = {}
    
    for video_path in selected_videos:
        print("\n" + "=" * 60)
        print(f"Processing: {video_path.name}")
        print("=" * 60)
        
        output_name = f"output_{video_path.stem}_annotated.mp4"
        output_path = f"output/videos/{output_name}"
        
        try:
            results = alpr.process_video(
                str(video_path),
                output_path=output_path,
                frame_skip=5  # Process every 5th frame for speed
            )
            
            all_results[video_path.name] = results
            
            # Display results
            print(f"\n{'='*60}")
            print(f"RESULTS FOR: {video_path.name}")
            print(f"{'='*60}")
            print(f"Processing time: {results['processing_time']:.2f}s")
            print(f"Processing speed: {results['fps']:.1f} FPS")
            print(f"Total detections: {len(results['all_detections'])}")
            print(f"Unique plates: {len(results['unique_plates'])}")
            
            if results['unique_plates']:
                print(f"\n{'='*60}")
                print("DETECTED PLATES:")
                print(f"{'='*60}")
                for i, det in enumerate(results['unique_plates'], 1):
                    print(f"{i:2d}. {det['plate']:15s} | {det['vehicle_type']:10s} | "
                          f"Confidence: {det['confidence']:.2%} | Frame: {det['frame']}")
                
                # Save to CSV
                import pandas as pd
                df = pd.DataFrame(results['unique_plates'])
                csv_name = f"results_{video_path.stem}.csv"
                df.to_csv(f'output/results/{csv_name}', index=False)
                print(f"\n💾 Results saved to output/results/{csv_name}")
            else:
                print("\n⚠️  No plates detected in this video")
        
        except Exception as e:
            print(f"\n❌ Error processing {video_path.name}: {e}")
    
    # Summary
    if len(all_results) > 1:
        print("\n" + "=" * 60)
        print("📊 SUMMARY - ALL VIDEOS")
        print("=" * 60)
        for video_name, results in all_results.items():
            print(f"{video_name:40s} | Plates: {len(results['unique_plates']):3d} | "
                  f"Time: {results['processing_time']:6.1f}s")

if __name__ == "__main__":
    main()