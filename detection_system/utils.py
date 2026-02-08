import os
import json
import logging
from pathlib import Path
import cv2
import numpy as np
from datetime import datetime
import base64

logger = logging.getLogger(__name__)

def ensure_dir(directory):
    """Create directory if it doesn't exist"""
    Path(directory).mkdir(parents=True, exist_ok=True)
    return directory

def load_json_config(config_path):
    """Load JSON configuration file"""
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"Config file not found: {config_path}")
        return {}
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in {config_path}: {e}")
        return {}

def save_json_config(config_path, config_data):
    """Save configuration to JSON file"""
    try:
        ensure_dir(os.path.dirname(config_path))
        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=4)
        logger.info(f"Configuration saved to {config_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving config: {e}")
        return False

def encode_image_to_base64(image):
    """Convert image (numpy array) to base64 string"""
    try:
        if isinstance(image, np.ndarray):
            _, buffer = cv2.imencode('.jpg', image)
            return base64.b64encode(buffer).decode('utf-8')
        return None
    except Exception as e:
        logger.error(f"Error encoding image: {e}")
        return None

def decode_base64_to_image(base64_string):
    """Convert base64 string to image (numpy array)"""
    try:
        img_data = base64.b64decode(base64_string)
        nparr = np.frombuffer(img_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return image
    except Exception as e:
        logger.error(f"Error decoding image: {e}")
        return None

def save_image(image, directory, filename=None):
    """Save image to directory with timestamp filename"""
    try:
        ensure_dir(directory)
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"detection_{timestamp}.jpg"
        
        filepath = os.path.join(directory, filename)
        cv2.imwrite(filepath, image)
        logger.info(f"Image saved to {filepath}")
        return filepath
    except Exception as e:
        logger.error(f"Error saving image: {e}")
        return None

def resize_image(image, width=None, height=None, maintain_aspect=True):
    """Resize image to specified dimensions"""
    try:
        h, w = image.shape[:2]
        
        if width is None and height is None:
            return image
        
        if maintain_aspect:
            if width is not None:
                ratio = width / w
                new_size = (width, int(h * ratio))
            else:
                ratio = height / h
                new_size = (int(w * ratio), height)
        else:
            new_size = (width or w, height or h)
        
        resized = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
        return resized
    except Exception as e:
        logger.error(f"Error resizing image: {e}")
        return image

def validate_image(image):
    """Validate if image is valid numpy array"""
    if image is None:
        return False
    if not isinstance(image, np.ndarray):
        return False
    if image.size == 0:
        return False
    if len(image.shape) < 2:
        return False
    return True

def get_timestamp():
    """Get current timestamp as string"""
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

def get_timestamp_filename():
    """Get timestamp formatted for filename"""
    return datetime.now().strftime('%Y%m%d_%H%M%S')

def calculate_iou(box1, box2):
    """Calculate Intersection over Union of two bounding boxes"""
    try:
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        # Calculate intersection area
        x_inter_min = max(x1_min, x2_min)
        y_inter_min = max(y1_min, y2_min)
        x_inter_max = min(x1_max, x2_max)
        y_inter_max = min(y1_max, y2_max)
        
        if x_inter_max < x_inter_min or y_inter_max < y_inter_min:
            return 0.0
        
        inter_area = (x_inter_max - x_inter_min) * (y_inter_max - y_inter_min)
        
        # Calculate union area
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - inter_area
        
        if union_area == 0:
            return 0.0
        
        iou = inter_area / union_area
        return iou
    except Exception as e:
        logger.error(f"Error calculating IoU: {e}")
        return 0.0

def draw_text_with_background(image, text, position, font_scale=0.6, 
                              thickness=2, text_color=(255, 255, 255), 
                              bg_color=(0, 0, 0), padding=5):
    """Draw text with background rectangle"""
    try:
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Get text size
        (text_width, text_height), baseline = cv2.getTextSize(
            text, font, font_scale, thickness
        )
        
        x, y = position
        
        # Draw background rectangle
        cv2.rectangle(
            image,
            (x - padding, y - text_height - padding),
            (x + text_width + padding, y + baseline + padding),
            bg_color,
            -1
        )
        
        # Draw text
        cv2.putText(
            image,
            text,
            (x, y),
            font,
            font_scale,
            text_color,
            thickness
        )
        
        return image
    except Exception as e:
        logger.error(f"Error drawing text: {e}")
        return image

def create_detection_summary(detections):
    """Create summary statistics from detections"""
    summary = {
        'total': len(detections),
        'by_type': {},
        'high_confidence': 0,
        'alerts_required': 0
    }
    
    for det in detections:
        det_type = det.get('type', 'unknown')
        summary['by_type'][det_type] = summary['by_type'].get(det_type, 0) + 1
        
        if det.get('confidence', 0) > 0.7:
            summary['high_confidence'] += 1
        
        if det.get('alert_required', False):
            summary['alerts_required'] += 1
    
    return summary

def format_confidence(confidence):
    """Format confidence score as percentage string"""
    return f"{confidence * 100:.1f}%"

def get_file_size(filepath):
    """Get file size in human readable format"""
    try:
        size_bytes = os.path.getsize(filepath)
        
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        
        return f"{size_bytes:.1f} TB"
    except Exception as e:
        logger.error(f"Error getting file size: {e}")
        return "Unknown"

def cleanup_old_files(directory, max_age_days=7):
    """Delete files older than max_age_days"""
    try:
        current_time = datetime.now()
        count = 0
        
        for filename in os.listdir(directory):
            filepath = os.path.join(directory, filename)
            
            if os.path.isfile(filepath):
                file_modified = datetime.fromtimestamp(os.path.getmtime(filepath))
                age_days = (current_time - file_modified).days
                
                if age_days > max_age_days:
                    os.remove(filepath)
                    count += 1
        
        logger.info(f"Cleaned up {count} old files from {directory}")
        return count
    except Exception as e:
        logger.error(f"Error cleaning up files: {e}")
        return 0

def is_video_file(filename):
    """Check if file is a video file"""
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm']
    return any(filename.lower().endswith(ext) for ext in video_extensions)

def is_image_file(filename):
    """Check if file is an image file"""
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff']
    return any(filename.lower().endswith(ext) for ext in image_extensions)

def get_video_info(video_path):
    """Get video file information"""
    try:
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            return None
        
        info = {
            'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'duration_seconds': 0
        }
        
        if info['fps'] > 0:
            info['duration_seconds'] = info['frame_count'] / info['fps']
        
        cap.release()
        return info
    except Exception as e:
        logger.error(f"Error getting video info: {e}")
        return None

def merge_overlapping_detections(detections, iou_threshold=0.5):
    """Merge overlapping detections with same type"""
    if not detections:
        return []
    
    merged = []
    used = [False] * len(detections)
    
    for i, det1 in enumerate(detections):
        if used[i]:
            continue
        
        current = det1.copy()
        used[i] = True
        
        for j, det2 in enumerate(detections[i+1:], start=i+1):
            if used[j]:
                continue
            
            if det1['type'] == det2['type']:
                iou = calculate_iou(det1['bbox'], det2['bbox'])
                
                if iou > iou_threshold:
                    # Merge: keep higher confidence
                    if det2['confidence'] > current['confidence']:
                        current = det2.copy()
                    used[j] = True
        
        merged.append(current)
    
    return merged

if __name__ == "__main__":
    # Test utilities
    print("Testing utility functions...")
    
    # Test timestamp
    print(f"Current timestamp: {get_timestamp()}")
    print(f"Filename timestamp: {get_timestamp_filename()}")
    
    # Test IoU
    box1 = [0, 0, 100, 100]
    box2 = [50, 50, 150, 150]
    iou = calculate_iou(box1, box2)
    print(f"IoU between boxes: {iou:.2f}")
    
    print("Utilities test complete!")