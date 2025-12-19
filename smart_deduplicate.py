# Smart deduplication - merge similar plates
import pandas as pd
from pathlib import Path
from difflib import SequenceMatcher
import re

def normalize_plate(text):
    """Normalize plate text for comparison."""
    text = text.upper().strip()
    # Remove spaces and extra characters
    text = ''.join(c for c in text if c.isalnum() or c in ['-', '.'])
    return text

def similarity_score(a, b):
    """Calculate similarity between two plates."""
    a_norm = normalize_plate(a)
    b_norm = normalize_plate(b)
    
    # Exact match after normalization
    if a_norm == b_norm:
        return 1.0
    
    # Sequence similarity
    seq_sim = SequenceMatcher(None, a_norm, b_norm).ratio()
    
    # Bonus for matching core patterns (numbers match)
    a_nums = ''.join(c for c in a_norm if c.isdigit())
    b_nums = ''.join(c for c in b_norm if c.isdigit())
    
    if a_nums and b_nums:
        num_sim = SequenceMatcher(None, a_nums, b_nums).ratio()
        # Weight sequence and number similarity
        return 0.6 * seq_sim + 0.4 * num_sim
    
    return seq_sim

def is_valid_plate(text):
    """Filter out obviously invalid plates."""
    text = normalize_plate(text)
    
    # Too short or too long
    if len(text) < 5 or len(text) > 15:
        return False
    
    # Must have both letters and numbers
    has_letters = any(c.isalpha() for c in text)
    has_numbers = any(c.isdigit() for c in text)
    
    if not (has_letters and has_numbers):
        return False
    
    # Filter out common garbage patterns
    garbage_patterns = [
        r'SUZUK',  # Brand names
        r'VELZ',
        r'[A-Z]{6,}',  # Too many consecutive letters
        r'\d{6,}',  # Too many consecutive numbers
    ]
    
    for pattern in garbage_patterns:
        if re.search(pattern, text):
            return False
    
    return True

def deduplicate_plates(csv_path, similarity_threshold=0.75):
    """
    Smart deduplication of detected plates.
    
    Args:
        csv_path: Path to CSV file with detections
        similarity_threshold: Threshold for considering plates as duplicates (0-1)
    
    Returns:
        DataFrame with deduplicated results
    """
    print(f"\n🔍 Processing: {Path(csv_path).name}")
    print("=" * 70)
    
    df = pd.read_csv(csv_path)
    
    print(f"Original detections: {len(df)}")
    
    # Step 1: Filter invalid plates
    df['is_valid'] = df['plate'].apply(is_valid_plate)
    valid_df = df[df['is_valid']].copy()
    
    print(f"After filtering invalid: {len(valid_df)} (removed {len(df) - len(valid_df)})")
    
    if len(valid_df) == 0:
        print("⚠️  No valid plates remaining!")
        return pd.DataFrame()
    
    # Step 2: Sort by confidence (highest first)
    valid_df = valid_df.sort_values('confidence', ascending=False)
    
    # Step 3: Group similar plates
    groups = []
    seen_indices = set()
    
    for i, row in valid_df.iterrows():
        if i in seen_indices:
            continue
        
        # Start new group with this plate
        group = [row]
        seen_indices.add(i)
        
        # Find similar plates
        for j, other_row in valid_df.iterrows():
            if j not in seen_indices:
                sim = similarity_score(row['plate'], other_row['plate'])
                
                if sim >= similarity_threshold:
                    group.append(other_row)
                    seen_indices.add(j)
        
        # Keep the one with highest confidence (first in sorted list)
        groups.append(group[0])
    
    # Create deduplicated dataframe
    dedup_df = pd.DataFrame(groups)
    dedup_df = dedup_df.sort_values('confidence', ascending=False).reset_index(drop=True)
    
    print(f"After smart deduplication: {len(dedup_df)} (merged {len(valid_df) - len(dedup_df)} duplicates)")
    
    # Show results
    print(f"\n📋 DEDUPLICATED PLATES:")
    print("-" * 70)
    print(f"{'#':<4} {'Plate':<20} {'Type':<12} {'Confidence':<12} {'Frame':<8}")
    print("-" * 70)
    
    for idx, row in dedup_df.iterrows():
        print(f"{idx+1:<4} {row['plate']:<20} {row['vehicle_type']:<12} "
              f"{row['confidence']:.1%}{'':>5} {row['frame']:<8}")
    
    return dedup_df

def process_all_results():
    """Process all result CSVs and create cleaned versions."""
    results_dir = Path("output/results")
    csv_files = list(results_dir.glob("results_*.csv"))
    
    if not csv_files:
        print("❌ No result CSV files found in output/results/")
        return
    
    print("=" * 70)
    print("🧹 SMART DEDUPLICATION - ALL VIDEOS")
    print("=" * 70)
    
    summary = []
    
    for csv_file in sorted(csv_files):
        if csv_file.name == 'video_comparison.csv':
            continue
        
        dedup_df = deduplicate_plates(csv_file, similarity_threshold=0.75)
        
        if len(dedup_df) > 0:
            # Save cleaned version
            clean_path = results_dir / f"clean_{csv_file.name}"
            dedup_df.to_csv(clean_path, index=False)
            print(f"💾 Saved cleaned version: {clean_path.name}\n")
            
            summary.append({
                'video': csv_file.stem.replace('results_', ''),
                'original': len(pd.read_csv(csv_file)),
                'cleaned': len(dedup_df),
                'improvement': f"{(1 - len(dedup_df)/len(pd.read_csv(csv_file)))*100:.0f}%"
            })
    
    # Summary report
    if summary:
        print("\n" + "=" * 70)
        print("📊 DEDUPLICATION SUMMARY")
        print("=" * 70)
        
        summary_df = pd.DataFrame(summary)
        summary_df = summary_df.sort_values('cleaned', ascending=False)
        
        print(f"\n{'Video':<40} {'Original':<10} {'Cleaned':<10} {'Reduction':<10}")
        print("-" * 70)
        for _, row in summary_df.iterrows():
            video_short = row['video'][:37] + "..." if len(row['video']) > 40 else row['video']
            print(f"{video_short:<40} {row['original']:<10} {row['cleaned']:<10} {row['improvement']:<10}")
        
        print("\n" + "=" * 70)
        print(f"Total plates (original): {summary_df['original'].sum()}")
        print(f"Total plates (cleaned): {summary_df['cleaned'].sum()}")
        print(f"Overall reduction: {(1 - summary_df['cleaned'].sum()/summary_df['original'].sum())*100:.0f}%")
        print("=" * 70)

def deduplicate_single(csv_path):
    """Deduplicate a single CSV file."""
    if not Path(csv_path).exists():
        print(f"❌ File not found: {csv_path}")
        return
    
    dedup_df = deduplicate_plates(csv_path, similarity_threshold=0.75)
    
    if len(dedup_df) > 0:
        # Save
        output_path = Path(csv_path).parent / f"clean_{Path(csv_path).name}"
        dedup_df.to_csv(output_path, index=False)
        print(f"\n💾 Saved to: {output_path}")
        
        # Compare
        original_count = len(pd.read_csv(csv_path))
        improvement = (1 - len(dedup_df) / original_count) * 100
        
        print(f"\n✅ Reduced from {original_count} to {len(dedup_df)} plates ({improvement:.0f}% reduction)")
    
    return dedup_df

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Process specific file
        csv_path = sys.argv[1]
        deduplicate_single(csv_path)
    else:
        # Process all files
        process_all_results()
    
    print("\n✅ Deduplication complete!")