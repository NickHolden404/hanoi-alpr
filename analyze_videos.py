# analyze_videos.py
import cv2
from pathlib import Path
from alpr_system import HanoiALPRImproved

def analyze_video(video_path):
    """Quick analysis of a video to see if it's suitable for ALPR."""
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        return None
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps if fps > 0 else 0
    
    # Sample 5 frames
    alpr = HanoiALPRImproved()
    frame_samples = [total_frames // 6, total_frames // 3, total_frames // 2, 
                    2 * total_frames // 3, 5 * total_frames // 6]
    
    total_vehicles = 0
    vehicle_sizes = []
    
    for frame_idx in frame_samples:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if ret:
            vehicles = alpr.detect_vehicles(frame)
            total_vehicles += len(vehicles)
            
            # Check vehicle sizes
            for v in vehicles:
                x1, y1, x2, y2 = v['bbox']
                w = x2 - x1
                h = y2 - y1
                vehicle_sizes.append((w, h, w * h))
    
    cap.release()
    
    avg_vehicles = total_vehicles / len(frame_samples)
    
    # Calculate suitability score
    score = 0
    reason = []
    
    # Resolution
    if width >= 1280 and height >= 720:
        score += 30
        reason.append("✅ Good resolution")
    else:
        score += 10
        reason.append("⚠️  Low resolution")
    
    # Vehicle count
    if avg_vehicles >= 3:
        score += 30
        reason.append(f"✅ Good vehicle density ({avg_vehicles:.1f} avg)")
    elif avg_vehicles >= 1:
        score += 15
        reason.append(f"⚠️  Moderate vehicles ({avg_vehicles:.1f} avg)")
    else:
        reason.append("❌ Few vehicles detected")
    
    # Vehicle size (larger is better for OCR)
    if vehicle_sizes:
        avg_area = sum(s[2] for s in vehicle_sizes) / len(vehicle_sizes)
        if avg_area > 20000:
            score += 40
            reason.append(f"✅ Large vehicles (good for OCR)")
        elif avg_area > 10000:
            score += 20
            reason.append(f"⚠️  Medium vehicles")
        else:
            score += 5
            reason.append(f"❌ Small/distant vehicles")
    
    return {
        'resolution': f"{width}x{height}",
        'fps': fps,
        'duration': duration,
        'total_frames': total_frames,
        'avg_vehicles': avg_vehicles,
        'score': score,
        'reasons': reason
    }

def main():
    video_dir = Path("data/test_videos")
    video_extensions = ['.mp4', '.mov', '.avi', '.mkv']
    
    videos = []
    for ext in video_extensions:
        videos.extend(video_dir.glob(f"*{ext}"))
    
    if not videos:
        print("❌ No videos found")
        return
    
    print("=" * 80)
    print("🎥 VIDEO ANALYSIS - Finding Best Videos for ALPR")
    print("=" * 80)
    print("\nAnalyzing videos... (this may take a minute)\n")
    
    results = []
    
    for video in sorted(videos):
        print(f"📹 Analyzing: {video.name}...")
        analysis = analyze_video(video)
        
        if analysis:
            results.append((video, analysis))
    
    # Sort by score
    results.sort(key=lambda x: x[1]['score'], reverse=True)
    
    print("\n" + "=" * 80)
    print("📊 RESULTS (Best to Worst)")
    print("=" * 80)
    
    for i, (video, analysis) in enumerate(results, 1):
        score = analysis['score']
        
        if score >= 70:
            rating = "🟢 EXCELLENT"
        elif score >= 50:
            rating = "🟡 GOOD"
        elif score >= 30:
            rating = "🟠 FAIR"
        else:
            rating = "🔴 POOR"
        
        print(f"\n{i}. {video.name}")
        print(f"   Rating: {rating} (Score: {score}/100)")
        print(f"   Resolution: {analysis['resolution']} | Duration: {analysis['duration']:.1f}s")
        print(f"   Avg vehicles per frame: {analysis['avg_vehicles']:.1f}")
        
        for reason in analysis['reasons']:
            print(f"   {reason}")
    
    print("\n" + "=" * 80)
    print("💡 RECOMMENDATION")
    print("=" * 80)
    
    if results:
        best_video, best_analysis = results[0]
        
        if best_analysis['score'] >= 70:
            print(f"✅ Use '{best_video.name}' - it should work well!")
        elif best_analysis['score'] >= 50:
            print(f"⚠️  '{best_video.name}' might work with adjusted parameters")
        else:
            print(f"❌ None of your videos are ideal for ALPR")
            print("\n📸 For best results, record new footage with:")
            print("   • Close range (parking lot, not dashcam)")
            print("   • Stationary camera")
            print("   • Good lighting")
            print("   • Clear view of license plates")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()