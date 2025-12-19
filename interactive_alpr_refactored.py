#!/usr/bin/env python3
"""
Interactive ALPR testing script using refactored modules.
"""

import sys
import os
from pathlib import Path
import time

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src import ALPRSystem, load_config, VideoProcessor


def print_header():
    """Print application header."""
    print("\n" + "=" * 70)
    print("🎬 HANOI ALPR - INTERACTIVE TESTING")
    print("=" * 70)


def list_videos(video_dir: str) -> list:
    """List available videos in directory."""
    from src.video_processor import find_videos
    
    videos = find_videos(video_dir)
    
    if not videos:
        print(f"\n⚠️  No videos found in: {video_dir}")
        return []
    
    print(f"\nAvailable videos ({len(videos)} found):\n")
    
    for i, video_path in enumerate(videos, 1):
        size_mb = Path(video_path).stat().st_size / (1024 * 1024)
        name = Path(video_path).name
        # Truncate long names
        if len(name) > 50:
            name = name[:47] + "..."
        print(f"   {i:2d}. {name:<50} ({size_mb:>6.1f} MB)")
    
    print("\n    0. Exit")
    
    return videos


def progress_callback(current: int, total: int, detections: int):
    """Print progress during video processing."""
    percent = (current / total * 100) if total > 0 else 0
    print(f"\r⏳ Processing: {current}/{total} frames ({percent:.1f}%) | Detections: {detections}", 
          end='', flush=True)


def print_results(alpr: ALPRSystem):
    """Print detection results."""
    detections = alpr.get_unique_detections()
    summary = alpr.get_summary()
    
    print("\n\n" + "=" * 70)
    print("✅ DETECTION RESULTS")
    print("=" * 70)
    
    print(f"\n📊 Summary:")
    print(f"   Total detections: {summary['total_detections']}")
    print(f"   Unique plates: {summary['unique_plates']}")
    print(f"   Average confidence: {summary['avg_confidence']:.1%}")
    
    if detections:
        print(f"\n🏆 Best detection: {summary['best_detection']} ({summary['best_confidence']:.1%})")
        
        print(f"\n📋 All unique plates:\n")
        print(f"{'#':<5} {'Plate':<15} {'Type':<12} {'Conf':<8} {'Frame':<8}")
        print("-" * 70)
        
        for i, det in enumerate(detections, 1):
            print(f"{i:<5} {det.plate:<15} {det.vehicle_type:<12} "
                  f"{det.confidence:>6.1%}  {det.frame:<8}")
    else:
        print("\n⚠️  No valid plates detected")


def main():
    """Main interactive loop."""
    # Load configuration
    try:
        config = load_config()
    except FileNotFoundError:
        print("❌ config.yaml not found!")
        print("Please create config.yaml in the project root directory")
        return 1
    
    # Get video directory from config
    video_dir = config.get('video.video_dir', './videos')
    output_dir = config.get('output.output_dir', './output/results')
    
    # Initialize ALPR system
    print("\n🚀 Initializing ALPR System...")
    alpr = ALPRSystem(config)
    print("✅ System ready!")
    
    while True:
        print_header()
        
        # List videos
        videos = list_videos(video_dir)
        
        if not videos:
            break
        
        # Get user choice
        try:
            choice = input("\n👉 Choose video number (0 to exit): ").strip()
            
            if choice == '0':
                print("\n👋 Goodbye!")
                break
            
            video_idx = int(choice) - 1
            
            if video_idx < 0 or video_idx >= len(videos):
                print("❌ Invalid choice!")
                input("\nPress Enter to continue...")
                continue
            
            video_path = videos[video_idx]
            video_name = Path(video_path).name
            
        except (ValueError, KeyboardInterrupt):
            print("\n\n👋 Goodbye!")
            break
        
        # Process video
        print(f"\n🎥 Processing: {video_name}")
        print(f"📊 Settings: frame_skip={alpr.frame_skip}, max_frames={alpr.max_frames}")
        
        start_time = time.time()
        
        try:
            alpr.process_video(video_path, progress_callback)
            
            elapsed = time.time() - start_time
            print(f"\n\n✅ Done in {elapsed:.0f}s ({elapsed/60:.1f} min)")
            
            # Print results
            print_results(alpr)
            
            # Save results
            output_filename = f"results_{Path(video_name).stem}.csv"
            output_path = Path(output_dir) / output_filename
            alpr.save_results(str(output_path))
            
            print(f"\n💾 Results saved to: {output_path}")
            
        except Exception as e:
            print(f"\n\n❌ Error processing video: {e}")
            import traceback
            traceback.print_exc()
        
        input("\n\nPress Enter to continue...")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
