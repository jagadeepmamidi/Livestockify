"""
Script to create a sample test video with moving objects for demonstration.
This generates a synthetic video to test the bird counting system.
"""
import cv2
import numpy as np
from pathlib import Path
import config


def create_sample_video(
    output_path: str = "test_video.mp4",
    duration: int = 10,
    fps: int = 30,
    width: int = 1280,
    height: int = 720,
    num_birds: int = 8
):
    """
    Create a sample video with moving circular objects simulating birds.
    
    Args:
        output_path: Path to save the video
        duration: Duration in seconds
        fps: Frames per second
        width: Video width
        height: Video height
        num_birds: Number of simulated birds
    """
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Initialize bird positions and velocities
    birds = []
    for i in range(num_birds):
        bird = {
            'x': np.random.randint(50, width - 50),
            'y': np.random.randint(50, height - 50),
            'vx': np.random.randint(-3, 4),
            'vy': np.random.randint(-3, 4),
            'radius': np.random.randint(15, 35),
            'color': (
                np.random.randint(100, 200),
                np.random.randint(100, 200),
                np.random.randint(100, 200)
            )
        }
        birds.append(bird)
    
    total_frames = duration * fps
    
    print(f"Creating sample video: {output_path}")
    print(f"Duration: {duration}s, FPS: {fps}, Resolution: {width}x{height}")
    print(f"Number of birds: {num_birds}")
    
    for frame_idx in range(total_frames):
        # Create background (greenish for farm environment)
        frame = np.ones((height, width, 3), dtype=np.uint8) * 180
        frame[:, :, 1] = 200  # More green
        frame[:, :, 2] = 150  # Less blue
        
        # Add some texture
        noise = np.random.randint(-20, 20, (height, width, 3), dtype=np.int16)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # Update and draw birds
        for bird in birds:
            # Update position
            bird['x'] += bird['vx']
            bird['y'] += bird['vy']
            
            # Bounce off walls
            if bird['x'] <= bird['radius'] or bird['x'] >= width - bird['radius']:
                bird['vx'] *= -1
            if bird['y'] <= bird['radius'] or bird['y'] >= height - bird['radius']:
                bird['vy'] *= -1
            
            # Keep within bounds
            bird['x'] = max(bird['radius'], min(width - bird['radius'], bird['x']))
            bird['y'] = max(bird['radius'], min(height - bird['radius'], bird['y']))
            
            # Draw bird (ellipse to simulate bird shape)
            cv2.ellipse(
                frame,
                (int(bird['x']), int(bird['y'])),
                (bird['radius'], int(bird['radius'] * 0.7)),
                0,
                0,
                360,
                bird['color'],
                -1
            )
            
            # Add some detail (head)
            head_x = int(bird['x'] + bird['radius'] * 0.5)
            head_y = int(bird['y'])
            cv2.circle(frame, (head_x, head_y), int(bird['radius'] * 0.3), bird['color'], -1)
        
        # Add timestamp
        timestamp = f"Time: {frame_idx / fps:.2f}s"
        cv2.putText(
            frame,
            timestamp,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            2
        )
        
        out.write(frame)
        
        if frame_idx % fps == 0:
            print(f"Progress: {frame_idx / total_frames * 100:.1f}%")
    
    out.release()
    print(f"Sample video created successfully: {output_path}")


if __name__ == "__main__":
    # Create outputs directory if it doesn't exist
    config.OUTPUTS_DIR.mkdir(exist_ok=True)
    
    # Create sample video
    output_path = str(config.OUTPUTS_DIR / "sample_test_video.mp4")
    create_sample_video(
        output_path=output_path,
        duration=15,
        fps=30,
        num_birds=10
    )
