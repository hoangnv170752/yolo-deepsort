import cv2
import os
import argparse


def extract_frames(video_path, output_dir='frames', interval=1, max_frames=None):
    """
    Extract frames from a video file.
    
    Args:
        video_path: Path to the video file
        output_dir: Directory to save extracted frames
        interval: Extract every nth frame (1 = all frames, 10 = every 10th frame)
        max_frames: Maximum number of frames to extract (None = all frames)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video Properties:")
    print(f"  - Total frames: {total_frames}")
    print(f"  - FPS: {fps:.2f}")
    print(f"  - Resolution: {width}x{height}")
    print(f"\nExtracting frames (interval={interval})...")
    
    frame_count = 0
    saved_count = 0
    
    while cap.isOpened():
        success, frame = cap.read()
        
        if not success:
            break
        
        # Save frame if it matches the interval
        if frame_count % interval == 0:
            frame_filename = os.path.join(output_dir, f'frame_{frame_count:06d}.jpg')
            cv2.imwrite(frame_filename, frame)
            saved_count += 1
            
            if saved_count % 10 == 0:
                print(f"Extracted {saved_count} frames...")
            
            # Check if we've reached max_frames
            if max_frames and saved_count >= max_frames:
                break
        
        frame_count += 1
    
    cap.release()
    
    print(f"\nCompleted!")
    print(f"Total frames extracted: {saved_count}")
    print(f"Frames saved to: {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description='Extract frames from action.avi')
    parser.add_argument('--video', type=str, default='action.avi',
                        help='Path to video file (default: action.avi)')
    parser.add_argument('--output', type=str, default='frames',
                        help='Output directory for frames (default: frames/)')
    parser.add_argument('--interval', type=int, default=1,
                        help='Extract every nth frame (default: 1 = all frames)')
    parser.add_argument('--max-frames', type=int, default=None,
                        help='Maximum number of frames to extract (default: None = all)')
    
    args = parser.parse_args()
    
    extract_frames(
        video_path=args.video,
        output_dir=args.output,
        interval=args.interval,
        max_frames=args.max_frames
    )


if __name__ == "__main__":
    main()
