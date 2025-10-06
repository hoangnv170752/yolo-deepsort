"""
Professional Visualization: Top-Down vs Bottom-Up Approaches
Using real YOLOv9e model for accurate human detection
Layout: Input -> Intermediate Steps -> Result
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import torch
from typing import List, Tuple
import argparse


class RealDetector:
    """Real YOLO detector using YOLOv9e model"""
    
    def __init__(self, model_path='yolov9e.pt'):
        print(f"Loading YOLOv9e model from {model_path}...")
        try:
            # Try using ultralytics directly first (better for YOLOv9)
            from ultralytics import YOLO
            self.model = YOLO(model_path)
            self.use_ultralytics = True
            print(f"Model loaded successfully using ultralytics")
        except Exception as e:
            print(f"Ultralytics loading failed, trying torch.hub with force_reload...")
            # Fallback to torch.hub with force_reload
            self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.model.to(self.device)
            self.use_ultralytics = False
            print(f"Model loaded successfully on {self.device}")
    
    def detect_humans(self, frame: np.ndarray, conf_threshold=0.5):
        """Detect humans in the frame"""
        results = self.model(frame)
        
        # Extract detections
        detections = []
        h, w = frame.shape[:2]
        
        if self.use_ultralytics:
            # Ultralytics YOLO format
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    
                    # Class 0 is 'person' in COCO dataset
                    if cls == 0 and conf >= conf_threshold:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        detections.append({
                            'bbox': (int(x1), int(y1), int(x2), int(y2)),
                            'confidence': conf,
                            'class': 'person'
                        })
        else:
            # torch.hub YOLOv5 format
            labels = results.xyxyn[0][:, -1]
            bb_coords = results.xyxyn[0][:, :-1]
            
            for i, label in enumerate(labels):
                # Class 0 is 'person' in COCO dataset
                if int(label) == 0 and bb_coords[i][4] >= conf_threshold:
                    x1 = int(bb_coords[i][0] * w)
                    y1 = int(bb_coords[i][1] * h)
                    x2 = int(bb_coords[i][2] * w)
                    y2 = int(bb_coords[i][3] * h)
                    conf = float(bb_coords[i][4])
                    
                    detections.append({
                        'bbox': (x1, y1, x2, y2),
                        'confidence': conf,
                        'class': 'person'
                    })
        
        return detections


def create_top_down_visualization(frame: np.ndarray, detections: List[dict]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create Top-Down approach visualization
    Step 1: Input image
    Step 2: Detection (bounding boxes)
    Step 3: Pose estimation (keypoints on detected persons)
    """
    h, w = frame.shape[:2]
    
    # Step 1: Input image (clean)
    step1 = frame.copy()
    
    # Step 2: Detection with bounding boxes
    step2 = frame.copy()
    colors = [(255, 255, 0), (0, 255, 255), (255, 0, 255), (0, 255, 0)]
    
    for idx, det in enumerate(detections):
        x1, y1, x2, y2 = det['bbox']
        color = colors[idx % len(colors)]
        
        # Draw bounding box
        cv2.rectangle(step2, (x1, y1), (x2, y2), color, 3)
        
        # Add label with confidence
        label = f"Person {det['confidence']:.2f}"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        
        # Draw label background
        cv2.rectangle(step2, (x1, y1 - label_size[1] - 10), 
                     (x1 + label_size[0], y1), color, -1)
        cv2.putText(step2, label, (x1, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    # Step 3: Pose estimation (skeleton on detected persons)
    step3 = frame.copy()
    
    for idx, det in enumerate(detections):
        x1, y1, x2, y2 = det['bbox']
        color = colors[idx % len(colors)]
        
        # Draw bounding box (lighter)
        cv2.rectangle(step3, (x1, y1), (x2, y2), color, 2)
        
        # Calculate better keypoints based on bounding box
        person_h = y2 - y1
        person_w = x2 - x1
        
        # Adjust center based on typical human proportions
        # Head is usually in upper portion, body center is lower
        cx = (x1 + x2) // 2
        body_center_y = y1 + int(person_h * 0.45)  # Body center lower than bbox center
        
        # Define keypoint positions with better proportions
        keypoints = {
            'nose': (cx, y1 + int(person_h * 0.08)),
            'neck': (cx, y1 + int(person_h * 0.15)),
            'r_shoulder': (cx + int(person_w * 0.18), y1 + int(person_h * 0.18)),
            'l_shoulder': (cx - int(person_w * 0.18), y1 + int(person_h * 0.18)),
            'r_elbow': (cx + int(person_w * 0.22), y1 + int(person_h * 0.35)),
            'l_elbow': (cx - int(person_w * 0.22), y1 + int(person_h * 0.35)),
            'r_wrist': (cx + int(person_w * 0.25), y1 + int(person_h * 0.52)),
            'l_wrist': (cx - int(person_w * 0.25), y1 + int(person_h * 0.52)),
            'r_hip': (cx + int(person_w * 0.12), y1 + int(person_h * 0.50)),
            'l_hip': (cx - int(person_w * 0.12), y1 + int(person_h * 0.50)),
            'r_knee': (cx + int(person_w * 0.13), y1 + int(person_h * 0.72)),
            'l_knee': (cx - int(person_w * 0.13), y1 + int(person_h * 0.72)),
            'r_ankle': (cx + int(person_w * 0.10), y1 + int(person_h * 0.95)),
            'l_ankle': (cx - int(person_w * 0.10), y1 + int(person_h * 0.95)),
        }
        
        # Draw skeleton connections
        skeleton = [
            ('nose', 'neck'),
            ('neck', 'r_shoulder'), ('neck', 'l_shoulder'),
            ('r_shoulder', 'r_elbow'), ('r_elbow', 'r_wrist'),
            ('l_shoulder', 'l_elbow'), ('l_elbow', 'l_wrist'),
            ('neck', 'r_hip'), ('neck', 'l_hip'),
            ('r_hip', 'r_knee'), ('r_knee', 'r_ankle'),
            ('l_hip', 'l_knee'), ('l_knee', 'l_ankle'),
        ]
        
        # Draw connections
        for joint1, joint2 in skeleton:
            if joint1 in keypoints and joint2 in keypoints:
                pt1 = keypoints[joint1]
                pt2 = keypoints[joint2]
                cv2.line(step3, pt1, pt2, color, 3)
        
        # Draw keypoints
        for joint, point in keypoints.items():
            cv2.circle(step3, point, 5, color, -1)
            cv2.circle(step3, point, 5, (255, 255, 255), 1)
    
    return step1, step2, step3


def create_bottom_up_visualization(frame: np.ndarray, detections: List[dict]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create Bottom-Up approach visualization
    Step 1: Input image
    Step 2: Keypoint detection (all keypoints first)
    Step 3: Grouping keypoints into persons
    """
    h, w = frame.shape[:2]
    
    # Step 1: Input image (clean)
    step1 = frame.copy()
    
    # Step 2: Detect all keypoints first (without knowing which person they belong to)
    step2 = frame.copy()
    
    all_keypoints = []
    colors = [(255, 255, 0), (0, 255, 255), (255, 0, 255), (0, 255, 0)]
    
    for idx, det in enumerate(detections):
        x1, y1, x2, y2 = det['bbox']
        person_h = y2 - y1
        person_w = x2 - x1
        cx = (x1 + x2) // 2
        
        # Generate keypoints with better proportions
        keypoints = [
            (cx, y1 + int(person_h * 0.08)),  # nose
            (cx, y1 + int(person_h * 0.15)),  # neck
            (cx + int(person_w * 0.18), y1 + int(person_h * 0.18)),  # r_shoulder
            (cx - int(person_w * 0.18), y1 + int(person_h * 0.18)),  # l_shoulder
            (cx + int(person_w * 0.22), y1 + int(person_h * 0.35)),  # r_elbow
            (cx - int(person_w * 0.22), y1 + int(person_h * 0.35)),  # l_elbow
            (cx + int(person_w * 0.25), y1 + int(person_h * 0.52)),  # r_wrist
            (cx - int(person_w * 0.25), y1 + int(person_h * 0.52)),  # l_wrist
            (cx + int(person_w * 0.12), y1 + int(person_h * 0.50)),  # r_hip
            (cx - int(person_w * 0.12), y1 + int(person_h * 0.50)),  # l_hip
            (cx + int(person_w * 0.13), y1 + int(person_h * 0.72)),  # r_knee
            (cx - int(person_w * 0.13), y1 + int(person_h * 0.72)),  # l_knee
            (cx + int(person_w * 0.10), y1 + int(person_h * 0.95)),  # r_ankle
            (cx - int(person_w * 0.10), y1 + int(person_h * 0.95)),  # l_ankle
        ]
        
        all_keypoints.extend([(kp, idx) for kp in keypoints])
    
    # Draw all keypoints without grouping (same color for all)
    for kp, person_idx in all_keypoints:
        cv2.circle(step2, kp, 6, (0, 255, 255), -1)
        cv2.circle(step2, kp, 6, (255, 255, 255), 1)
    
    # Step 3: Group keypoints into persons
    step3 = frame.copy()
    
    # Reorganize keypoints by person
    person_keypoints = {}
    for kp, person_idx in all_keypoints:
        if person_idx not in person_keypoints:
            person_keypoints[person_idx] = []
        person_keypoints[person_idx].append(kp)
    
    # Draw grouped keypoints and skeletons
    for person_idx, keypoints_list in person_keypoints.items():
        color = colors[person_idx % len(colors)]
        
        # Draw keypoints
        for kp in keypoints_list:
            cv2.circle(step3, kp, 5, color, -1)
            cv2.circle(step3, kp, 5, (255, 255, 255), 1)
        
        # Draw skeleton connections
        if len(keypoints_list) >= 14:
            skeleton_indices = [
                (0, 1), (1, 2), (1, 3),  # head to shoulders
                (2, 4), (4, 6),  # right arm
                (3, 5), (5, 7),  # left arm
                (1, 8), (1, 9),  # torso
                (8, 10), (10, 12),  # right leg
                (9, 11), (11, 13),  # left leg
            ]
            
            for idx1, idx2 in skeleton_indices:
                if idx1 < len(keypoints_list) and idx2 < len(keypoints_list):
                    cv2.line(step3, keypoints_list[idx1], keypoints_list[idx2], color, 3)
        
        # Draw bounding box around grouped person
        if keypoints_list:
            xs = [kp[0] for kp in keypoints_list]
            ys = [kp[1] for kp in keypoints_list]
            x1, y1 = max(0, min(xs) - 20), max(0, min(ys) - 20)
            x2, y2 = min(w, max(xs) + 20), min(h, max(ys) + 20)
            cv2.rectangle(step3, (x1, y1), (x2, y2), color, 2)
    
    return step1, step2, step3


def create_professional_comparison(frame: np.ndarray, detections: List[dict], output_path: str):
    """Create professional comparison figure like the reference image"""
    
    # Generate visualizations
    td_input, td_detect, td_pose = create_top_down_visualization(frame, detections)
    bu_input, bu_keypoints, bu_group = create_bottom_up_visualization(frame, detections)
    
    # Create figure with proper layout
    fig = plt.figure(figsize=(14, 8))
    
    # Add main title
    fig.suptitle('Method of Top-Down and Bottom-Up', fontsize=18, fontweight='bold', y=0.98)
    
    # Create grid layout
    gs = fig.add_gridspec(2, 3, hspace=0.15, wspace=0.1, 
                         left=0.08, right=0.92, top=0.92, bottom=0.08)
    
    # Top-Down row
    ax_td1 = fig.add_subplot(gs[0, 0])
    ax_td2 = fig.add_subplot(gs[0, 1])
    ax_td3 = fig.add_subplot(gs[0, 2])
    
    # Bottom-Up row
    ax_bu1 = fig.add_subplot(gs[1, 0])
    ax_bu2 = fig.add_subplot(gs[1, 1])
    ax_bu3 = fig.add_subplot(gs[1, 2])
    
    # Plot Top-Down
    ax_td1.imshow(cv2.cvtColor(td_input, cv2.COLOR_BGR2RGB))
    ax_td1.set_title('Input', fontsize=12, fontweight='bold')
    ax_td1.axis('off')
    
    ax_td2.imshow(cv2.cvtColor(td_detect, cv2.COLOR_BGR2RGB))
    ax_td2.set_title('Detection', fontsize=12, fontweight='bold')
    ax_td2.axis('off')
    
    ax_td3.imshow(cv2.cvtColor(td_pose, cv2.COLOR_BGR2RGB))
    ax_td3.set_title('Result', fontsize=12, fontweight='bold')
    ax_td3.axis('off')
    
    # Plot Bottom-Up
    ax_bu1.imshow(cv2.cvtColor(bu_input, cv2.COLOR_BGR2RGB))
    ax_bu1.set_title('Input', fontsize=12, fontweight='bold')
    ax_bu1.axis('off')
    
    ax_bu2.imshow(cv2.cvtColor(bu_keypoints, cv2.COLOR_BGR2RGB))
    ax_bu2.set_title('Keypoint Detection', fontsize=12, fontweight='bold')
    ax_bu2.axis('off')
    
    ax_bu3.imshow(cv2.cvtColor(bu_group, cv2.COLOR_BGR2RGB))
    ax_bu3.set_title('Result', fontsize=12, fontweight='bold')
    ax_bu3.axis('off')
    
    # Add method labels on the left
    fig.text(0.02, 0.73, 'Top-down\nmethod', fontsize=13, fontweight='bold',
             ha='left', va='center')
    fig.text(0.02, 0.27, 'Bottom-up\nmethod', fontsize=13, fontweight='bold',
             ha='left', va='center')
    
    # Add arrows between images
    arrow_props = dict(arrowstyle='->', lw=2.5, color='black')
    
    # Top-Down arrows
    fig.add_artist(plt.annotate('', xy=(0.38, 0.73), xytext=(0.32, 0.73),
                               xycoords='figure fraction', arrowprops=arrow_props))
    fig.add_artist(plt.annotate('', xy=(0.65, 0.73), xytext=(0.59, 0.73),
                               xycoords='figure fraction', arrowprops=arrow_props))
    
    # Bottom-Up arrows
    fig.add_artist(plt.annotate('', xy=(0.38, 0.27), xytext=(0.32, 0.27),
                               xycoords='figure fraction', arrowprops=arrow_props))
    fig.add_artist(plt.annotate('', xy=(0.65, 0.27), xytext=(0.59, 0.27),
                               xycoords='figure fraction', arrowprops=arrow_props))
    
    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n✓ Comparison saved to: {output_path}")
    
    plt.show()
    
    return fig


def main():
    parser = argparse.ArgumentParser(
        description='Professional comparison of Top-Down vs Bottom-Up approaches using YOLOv9e'
    )
    parser.add_argument('--input', type=str, default='human-detect.jpg',
                       help='Input image path')
    parser.add_argument('--model', type=str, default='yolov9e.pt',
                       help='YOLOv9e model path')
    parser.add_argument('--output', type=str, default='approach_comparison_professional.png',
                       help='Output comparison image path')
    parser.add_argument('--conf', type=float, default=0.5,
                       help='Confidence threshold for detection')
    
    args = parser.parse_args()
    
    # Check input image
    if not os.path.exists(args.input):
        print(f"Error: Input image not found: {args.input}")
        return
    
    # Check model
    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}")
        print("Please ensure yolov9e.pt is in the current directory")
        return
    
    # Load image
    print(f"Loading image: {args.input}")
    frame = cv2.imread(args.input)
    if frame is None:
        print(f"Error: Could not read image: {args.input}")
        return
    
    print(f"Image size: {frame.shape[1]}x{frame.shape[0]}")
    
    # Load detector
    detector = RealDetector(args.model)
    
    # Detect humans
    print("\nDetecting humans...")
    detections = detector.detect_humans(frame, conf_threshold=args.conf)
    print(f"Found {len(detections)} person(s)")
    
    if len(detections) == 0:
        print("Warning: No persons detected. Adjust --conf threshold or check image.")
        return
    
    for i, det in enumerate(detections):
        print(f"  Person {i+1}: confidence={det['confidence']:.3f}")
    
    # Create comparison visualization
    print("\nGenerating professional comparison...")
    create_professional_comparison(frame, detections, args.output)
    
    print("\n✓ Done!")


if __name__ == "__main__":
    main()
