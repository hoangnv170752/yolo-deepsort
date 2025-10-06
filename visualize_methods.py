"""
Professional Visualization: Top-Down vs Bottom-Up Approaches
Using native YOLO detection (YOLOv7/YOLOv9 style) for accurate results
Based on official YOLOv7 detect.py structure
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from typing import List, Tuple, Dict
import argparse
from pathlib import Path


def load_yolo_model(weights_path: str, device: str = ''):
    """Load YOLO model using native PyTorch"""
    # Select device
    device = torch.device('cuda' if torch.cuda.is_available() and device != 'cpu' else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {weights_path}...")
    model = torch.load(weights_path, map_location=device)['model'].float()
    model.to(device).eval()
    
    print("Model loaded successfully!")
    return model, device


def detect_persons(model, device, img_path: str, conf_thres: float = 0.5, img_size: int = 640):
    """Detect persons in image using YOLO model"""
    # Read image
    img0 = cv2.imread(img_path)
    assert img0 is not None, f'Image Not Found {img_path}'
    
    # Padded resize
    img = letterbox(img0, img_size, stride=32)[0]
    
    # Convert
    img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    img = np.ascontiguousarray(img)
    
    img = torch.from_numpy(img).to(device)
    img = img.float()
    img /= 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    
    # Inference
    with torch.no_grad():
        pred = model(img)[0]
    
    # NMS
    pred = non_max_suppression(pred, conf_thres, 0.45, classes=[0], agnostic=False)
    
    # Process detections
    detections = []
    for i, det in enumerate(pred):
        if len(det):
            # Rescale boxes from img_size to img0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
            
            # Extract person detections (class 0)
            for *xyxy, conf, cls in reversed(det):
                if int(cls) == 0:  # person class
                    x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                    detections.append({
                        'bbox': (x1, y1, x2, y2),
                        'confidence': float(conf),
                        'class': 'person'
                    })
    
    return img0, detections


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    """Resize and pad image while meeting stride-multiple constraints"""
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)


def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, max_det=300):
    """Runs Non-Maximum Suppression (NMS) on inference results"""
    nc = prediction.shape[2] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Settings
    max_wh = 4096  # (pixels) minimum and maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label = nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        x = x[xc[xi]]  # confidence

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torch.ops.torchvision.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]

        output[xi] = x[i]

    return output


def xywh2xyxy(x):
    """Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2]"""
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    """Rescale coords (xyxy) from img1_shape to img0_shape"""
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords


def clip_coords(boxes, img_shape):
    """Clip bounding xyxy bounding boxes to image shape (height, width)"""
    boxes[:, 0].clamp_(0, img_shape[1])  # x1
    boxes[:, 1].clamp_(0, img_shape[0])  # y1
    boxes[:, 2].clamp_(0, img_shape[1])  # x2
    boxes[:, 3].clamp_(0, img_shape[0])  # y2


def generate_skeleton_keypoints(x1: int, y1: int, x2: int, y2: int) -> Dict:
    """Generate realistic skeleton keypoints based on bounding box"""
    person_h = y2 - y1
    person_w = x2 - x1
    cx = (x1 + x2) // 2
    
    # Human body proportions (more accurate)
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
    
    return keypoints


def draw_skeleton(img: np.ndarray, keypoints: Dict, color: Tuple, thickness: int = 3):
    """Draw skeleton on image"""
    # Define skeleton connections
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
            cv2.line(img, pt1, pt2, color, thickness)
    
    # Draw keypoints
    for joint, point in keypoints.items():
        cv2.circle(img, point, 5, color, -1)
        cv2.circle(img, point, 5, (255, 255, 255), 1)


def create_top_down_steps(frame: np.ndarray, detections: List[dict]) -> Tuple:
    """Create Top-Down visualization steps"""
    colors = [(255, 255, 0), (0, 255, 255), (255, 0, 255), (0, 255, 0)]
    
    # Step 1: Input
    step1 = frame.copy()
    
    # Step 2: Detection
    step2 = frame.copy()
    for idx, det in enumerate(detections):
        x1, y1, x2, y2 = det['bbox']
        color = colors[idx % len(colors)]
        cv2.rectangle(step2, (x1, y1), (x2, y2), color, 3)
        label = f"Person {det['confidence']:.2f}"
        cv2.putText(step2, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    # Step 3: Pose Estimation
    step3 = frame.copy()
    for idx, det in enumerate(detections):
        x1, y1, x2, y2 = det['bbox']
        color = colors[idx % len(colors)]
        cv2.rectangle(step3, (x1, y1), (x2, y2), color, 2)
        keypoints = generate_skeleton_keypoints(x1, y1, x2, y2)
        draw_skeleton(step3, keypoints, color)
    
    return step1, step2, step3


def create_bottom_up_steps(frame: np.ndarray, detections: List[dict]) -> Tuple:
    """Create Bottom-Up visualization steps"""
    colors = [(255, 255, 0), (0, 255, 255), (255, 0, 255), (0, 255, 0)]
    h, w = frame.shape[:2]
    
    # Step 1: Input
    step1 = frame.copy()
    
    # Step 2: All keypoints
    step2 = frame.copy()
    all_keypoints = []
    for idx, det in enumerate(detections):
        x1, y1, x2, y2 = det['bbox']
        keypoints = generate_skeleton_keypoints(x1, y1, x2, y2)
        all_keypoints.extend([(kp, idx) for kp in keypoints.values()])
    
    for kp, _ in all_keypoints:
        cv2.circle(step2, kp, 6, (0, 255, 255), -1)
        cv2.circle(step2, kp, 6, (255, 255, 255), 1)
    
    # Step 3: Grouped
    step3 = frame.copy()
    for idx, det in enumerate(detections):
        x1, y1, x2, y2 = det['bbox']
        color = colors[idx % len(colors)]
        keypoints = generate_skeleton_keypoints(x1, y1, x2, y2)
        draw_skeleton(step3, keypoints, color)
        cv2.rectangle(step3, (x1, y1), (x2, y2), color, 2)
    
    return step1, step2, step3


def create_comparison_figure(frame: np.ndarray, detections: List[dict], output_path: str):
    """Create professional comparison figure"""
    # Generate steps
    td1, td2, td3 = create_top_down_steps(frame, detections)
    bu1, bu2, bu3 = create_bottom_up_steps(frame, detections)
    
    # Create figure
    fig = plt.figure(figsize=(14, 8))
    fig.suptitle('Method of Top-Down and Bottom-Up', fontsize=18, fontweight='bold', y=0.98)
    
    gs = fig.add_gridspec(2, 3, hspace=0.15, wspace=0.1, 
                         left=0.08, right=0.92, top=0.92, bottom=0.08)
    
    # Top-Down
    ax_td1 = fig.add_subplot(gs[0, 0])
    ax_td2 = fig.add_subplot(gs[0, 1])
    ax_td3 = fig.add_subplot(gs[0, 2])
    
    ax_td1.imshow(cv2.cvtColor(td1, cv2.COLOR_BGR2RGB))
    ax_td1.set_title('Input', fontsize=12, fontweight='bold')
    ax_td1.axis('off')
    
    ax_td2.imshow(cv2.cvtColor(td2, cv2.COLOR_BGR2RGB))
    ax_td2.set_title('Detection', fontsize=12, fontweight='bold')
    ax_td2.axis('off')
    
    ax_td3.imshow(cv2.cvtColor(td3, cv2.COLOR_BGR2RGB))
    ax_td3.set_title('Result', fontsize=12, fontweight='bold')
    ax_td3.axis('off')
    
    # Bottom-Up
    ax_bu1 = fig.add_subplot(gs[1, 0])
    ax_bu2 = fig.add_subplot(gs[1, 1])
    ax_bu3 = fig.add_subplot(gs[1, 2])
    
    ax_bu1.imshow(cv2.cvtColor(bu1, cv2.COLOR_BGR2RGB))
    ax_bu1.set_title('Input', fontsize=12, fontweight='bold')
    ax_bu1.axis('off')
    
    ax_bu2.imshow(cv2.cvtColor(bu2, cv2.COLOR_BGR2RGB))
    ax_bu2.set_title('Keypoint Detection', fontsize=12, fontweight='bold')
    ax_bu2.axis('off')
    
    ax_bu3.imshow(cv2.cvtColor(bu3, cv2.COLOR_BGR2RGB))
    ax_bu3.set_title('Result', fontsize=12, fontweight='bold')
    ax_bu3.axis('off')
    
    # Labels
    fig.text(0.02, 0.73, 'Top-down\nmethod', fontsize=13, fontweight='bold', ha='left', va='center')
    fig.text(0.02, 0.27, 'Bottom-up\nmethod', fontsize=13, fontweight='bold', ha='left', va='center')
    
    # Arrows
    arrow_props = dict(arrowstyle='->', lw=2.5, color='black')
    fig.add_artist(plt.annotate('', xy=(0.38, 0.73), xytext=(0.32, 0.73), xycoords='figure fraction', arrowprops=arrow_props))
    fig.add_artist(plt.annotate('', xy=(0.65, 0.73), xytext=(0.59, 0.73), xycoords='figure fraction', arrowprops=arrow_props))
    fig.add_artist(plt.annotate('', xy=(0.38, 0.27), xytext=(0.32, 0.27), xycoords='figure fraction', arrowprops=arrow_props))
    fig.add_artist(plt.annotate('', xy=(0.65, 0.27), xytext=(0.59, 0.27), xycoords='figure fraction', arrowprops=arrow_props))
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n✓ Saved to: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='yolov9e.pt', help='model.pt path')
    parser.add_argument('--source', type=str, default='human-detect.jpg', help='source image')
    parser.add_argument('--img-size', type=int, default=640, help='inference size')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='confidence threshold')
    parser.add_argument('--device', default='', help='cuda device or cpu')
    parser.add_argument('--output', type=str, default='method_comparison.png', help='output path')
    
    opt = parser.parse_args()
    
    # Check files
    if not os.path.exists(opt.weights):
        print(f"Error: Model not found: {opt.weights}")
        return
    
    if not os.path.exists(opt.source):
        print(f"Error: Image not found: {opt.source}")
        return
    
    print(f"\n{'='*60}")
    print(f"Source: {opt.source}")
    print(f"Weights: {opt.weights}")
    print(f"Confidence: {opt.conf_thres}")
    print(f"{'='*60}\n")
    
    # Load model
    model, device = load_yolo_model(opt.weights, opt.device)
    
    # Detect
    print("Detecting persons...")
    frame, detections = detect_persons(model, device, opt.source, opt.conf_thres, opt.img_size)
    
    print(f"Found {len(detections)} person(s)")
    for i, det in enumerate(detections):
        print(f"  Person {i+1}: confidence={det['confidence']:.3f}, bbox={det['bbox']}")
    
    if len(detections) == 0:
        print("No persons detected. Try lowering --conf-thres")
        return
    
    # Create visualization
    print("\nGenerating comparison...")
    create_comparison_figure(frame, detections, opt.output)
    print("\n✓ Done!")


if __name__ == '__main__':
    main()
