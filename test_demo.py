"""
Test script to process the sample video and generate demo outputs.
"""
import json
from pathlib import Path
from src.video_processor import VideoProcessor
import config

def main():
    # Input video
    video_path = str(config.OUTPUTS_DIR / "sample_test_video.mp4")
    
    if not Path(video_path).exists():
        print(f"Error: Sample video not found at {video_path}")
        print("Please run create_sample_video.py first")
        return
    
    print("=" * 60)
    print("Bird Counting & Weight Estimation - Demo Test")
    print("=" * 60)
    
    # Create processor
    processor = VideoProcessor()
    
    # Process video
    print(f"\nProcessing video: {video_path}")
    result = processor.process_video(
        video_path=video_path,
        output_path=str(config.OUTPUTS_DIR / "demo_annotated.mp4"),
        fps_sample=5,
        conf_thresh=0.3,  # Lower threshold for synthetic objects
        iou_thresh=0.45
    )
    
    # Save demo response JSON
    demo_json_path = config.OUTPUTS_DIR / "demo_response.json"
    with demo_json_path.open("w") as f:
        json.dump(result, f, indent=2)
    
    print("\n" + "=" * 60)
    print("Demo outputs generated successfully!")
    print("=" * 60)
    print(f"\nOutputs:")
    print(f"  - Annotated video: {result['artifacts']['annotated_video']}")
    print(f"  - Counts CSV: {result['artifacts']['counts_csv']}")
    print(f"  - Demo JSON: {demo_json_path}")
    
    print(f"\nSummary Statistics:")
    print(f"  - Total frames processed: {result['summary_statistics']['total_frames_processed']}")
    print(f"  - Unique birds tracked: {result['summary_statistics']['unique_birds_tracked']}")
    print(f"  - Max simultaneous birds: {result['summary_statistics']['max_simultaneous_birds']}")
    print(f"  - Avg birds per frame: {result['summary_statistics']['avg_birds_per_frame']:.2f}")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()
