import cv2
import numpy as np
from ultralytics import YOLO
import torch
from datetime import datetime, timedelta
import json
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MultiObjectDetector:
    def __init__(self, config_path='config/config.json'):
        """Initialize detector with minimal requirements"""
        self.config = self.load_config(config_path)
        self.device = 'cpu'  # Minimal requirement - CPU only
        self.models = {}
        self.last_alert_time = {}
        self.detection_count = {'fire': 0, 'fall': 0, 'vehicle': 0}
        
        # Alert cooldown (seconds) to prevent spam
        self.alert_cooldown = self.config.get('alert_cooldown', 30)
        
        logger.info(f"Initializing detector on {self.device}")
        self.load_models()
        
    def load_config(self, config_path):
        """Load configuration"""
        try:
            if Path(config_path).exists():
                with open(config_path, 'r') as f:
                    return json.load(f)
            else:
                # Default minimal config
                return {
                    'fire_threshold': 0.4,
                    'fall_threshold': 0.5,
                    'vehicle_threshold': 0.5,
                    'alert_cooldown': 30,
                    'model_size': 'n', 
                    'enable_alerts': True
                }
        except Exception as e:
            logger.error(f"Config load error: {e}")
            return {}
    
    def load_models(self):
        """Load lightweight YOLO models - minimal requirements"""
        try:
            # Use YOLOv8 nano (smallest) for minimal requirements
            model_size = self.config.get('model_size', 'n')
            
            logger.info("Loading YOLOv8-nano models (CPU optimized)...")
            
            # Single model for all detections (even more minimal)
            self.model = YOLO(f'yolov8{model_size}.pt')
            
            logger.info("✓ Models loaded successfully!")
            
        except Exception as e:
            logger.error(f"Model loading error: {e}")
            raise
    
    def detect_objects(self, frame, detection_types=['all']):
        """
        Detect objects in frame
        Args:
            frame: numpy array (image)
            detection_types: list of types to detect ['fire', 'fall', 'vehicle', 'all']
        Returns:
            detections: list of detection dictionaries
        """
        detections = []
        
        if frame is None or frame.size == 0:
            return detections
        
        try:
            # Run inference with minimal settings
            results = self.model(
                frame,
                conf=0.3,
                verbose=False,
                device=self.device
            )
            
            for result in results:
                boxes = result.boxes
                
                for box in boxes:
                    class_id = int(box.cls[0])
                    class_name = result.names[class_id]
                    confidence = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    
                    # Classify detection type
                    det_type = self.classify_detection(class_name, box, frame)
                    
                    if det_type and ('all' in detection_types or det_type in detection_types):
                        detection = {
                            'type': det_type,
                            'class': class_name,
                            'confidence': confidence,
                            'bbox': [x1, y1, x2, y2],
                            'timestamp': datetime.now().isoformat(),
                            'alert_required': self.should_alert(det_type, confidence)
                        }
                        
                        detections.append(detection)
                        self.detection_count[det_type] = self.detection_count.get(det_type, 0) + 1
            
        except Exception as e:
            logger.error(f"Detection error: {e}")
        
        return detections
    
    def classify_detection(self, class_name, box, frame):
        """Classify what type of detection this is"""
        
        # Vehicle detection
        vehicle_classes = ['car', 'truck', 'bus', 'motorcycle', 'bicycle']
        if class_name in vehicle_classes:
            return 'vehicle'
        
        # Person detection - check for fall
        if class_name == 'person':
            if self.is_person_fallen(box, frame):
                return 'fall'
        
        # Fire detection (using color analysis)
        if self.is_fire_detected(box, frame):
            return 'fire'
        
        return None
    
    def is_person_fallen(self, box, frame):
        """Check if person is in fallen position"""
        try:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            
            # Calculate aspect ratio
            width = x2 - x1
            height = y2 - y1
            
            if height == 0:
                return False
            
            aspect_ratio = width / height
            
            # If width > height significantly, likely fallen
            # Normal standing person: aspect_ratio < 1
            # Fallen person: aspect_ratio > 1.5
            if aspect_ratio > 1.3:
                return True
            
            # Check if bounding box is in lower half of frame
            frame_height = frame.shape[0]
            if y2 > frame_height * 0.6:  # In lower 40% of frame
                if aspect_ratio > 1.0:
                    return True
            
        except Exception as e:
            logger.error(f"Fall detection error: {e}")
        
        return False
    
    def is_fire_detected(self, box, frame):
        """Detect fire using color analysis (red-orange-yellow hues)"""
        try:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            
            # Extract region
            roi = frame[y1:y2, x1:x2]
            
            if roi.size == 0:
                return False
            
            # Convert to HSV for color analysis
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            
            # Fire color ranges (red-orange-yellow)
            # Red range 1
            lower_red1 = np.array([0, 120, 70])
            upper_red1 = np.array([10, 255, 255])
            
            # Red range 2
            lower_red2 = np.array([170, 120, 70])
            upper_red2 = np.array([180, 255, 255])
            
            # Orange-Yellow range
            lower_orange = np.array([10, 120, 70])
            upper_orange = np.array([30, 255, 255])
            
            # Create masks
            mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
            mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
            mask3 = cv2.inRange(hsv, lower_orange, upper_orange)
            
            # Combine masks
            fire_mask = mask1 + mask2 + mask3
            
            # Calculate percentage of fire-colored pixels
            fire_pixels = np.sum(fire_mask > 0)
            total_pixels = roi.shape[0] * roi.shape[1]
            
            if total_pixels == 0:
                return False
            
            fire_percentage = (fire_pixels / total_pixels) * 100
            
            # If more than 30% is fire-colored, likely fire
            if fire_percentage > 30:
                return True
            
        except Exception as e:
            logger.error(f"Fire color detection error: {e}")
        
        return False
    
    def should_alert(self, det_type, confidence):
        """Check if alert should be sent (with cooldown)"""
        now = datetime.now()
        
        # Check if we've alerted recently
        if det_type in self.last_alert_time:
            time_since_last = (now - self.last_alert_time[det_type]).total_seconds()
            if time_since_last < self.alert_cooldown:
                return False
        
        # Check confidence thresholds
        thresholds = {
            'fire': self.config.get('fire_threshold', 0.4),
            'fall': self.config.get('fall_threshold', 0.5),
            'vehicle': self.config.get('vehicle_threshold', 0.5)
        }
        
        if confidence >= thresholds.get(det_type, 0.5):
            self.last_alert_time[det_type] = now
            return True
        
        return False
    
    def draw_detections(self, frame, detections):
        """Draw bounding boxes and labels on frame"""
        output = frame.copy()
        
        # Colors for different types
        colors = {
            'fire': (0, 0, 255),      # Red
            'fall': (0, 165, 255),    # Orange
            'vehicle': (0, 255, 0)    # Green
        }
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            color = colors.get(det['type'], (255, 255, 255))
            
            # Draw rectangle
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
            
            # Prepare label
            label = f"{det['type'].upper()}: {det['confidence']:.2f}"
            if det['alert_required']:
                label += " [ALERT]"
            
            # Calculate label size
            (label_width, label_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
            )
            
            # Draw label background
            cv2.rectangle(
                output,
                (x1, y1 - label_height - 10),
                (x1 + label_width, y1),
                color,
                -1
            )
            
            # Draw label text
            cv2.putText(
                output,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2
            )
        
        return output
    
    def get_statistics(self):
        """Get detection statistics"""
        return {
            'total_detections': sum(self.detection_count.values()),
            'fire_detections': self.detection_count.get('fire', 0),
            'fall_detections': self.detection_count.get('fall', 0),
            'vehicle_detections': self.detection_count.get('vehicle', 0),
            'last_alert_times': {
                k: v.isoformat() if isinstance(v, datetime) else v 
                for k, v in self.last_alert_time.items()
            }
        }

if __name__ == "__main__":
    # Test the detector
    detector = MultiObjectDetector()
    print("Detector initialized successfully!")
    print(f"Device: {detector.device}")
    print("Ready to detect: Fire, Falls, Vehicles")