"""
Flask Web Application for Multi-Object Detection
Mobile-responsive, lightweight, AWS-ready
"""

from flask import Flask, render_template, Response, jsonify, request, send_file
import cv2
import numpy as np
from datetime import datetime
import json
import threading
import os
from pathlib import Path
import base64
import io
from PIL import Image
import sys

# Add detection_system to path
sys.path.append(str(Path(__file__).parent))

from detection_system.detector import MultiObjectDetector
from detection_system.alerts import AlertManager

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-change-this'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Initialize detector and alert manager
detector = MultiObjectDetector()
alert_manager = AlertManager()

# Global variables
camera = None
detection_enabled = True
current_detections = []
frame_lock = threading.Lock()
latest_frame = None

class VideoCamera:
    """Handle video capture from different sources"""
    def __init__(self, source=0):
        self.source = source
        self.video = cv2.VideoCapture(source)
        self.video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Lower resolution for performance
        self.video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
    def __del__(self):
        self.video.release()
    
    def get_frame(self):
        """Get current frame from camera"""
        success, frame = self.video.read()
        if not success:
            return None
        return frame
    
    def release(self):
        """Release camera resource"""
        if self.video:
            self.video.release()

def generate_frames():
    """Generate frames for video streaming"""
    global latest_frame, current_detections
    
    while True:
        if camera is None:
            break
        
        frame = camera.get_frame()
        if frame is None:
            continue
        
        # Perform detection if enabled
        if detection_enabled:
            detections = detector.detect_objects(frame, detection_types=['all'])
            
            # Update current detections
            with frame_lock:
                current_detections = detections
                
                # Send alerts for required detections
                for det in detections:
                    if det['alert_required']:
                        alert_manager.send_alert(
                            detection_type=det['type'],
                            confidence=det['confidence'],
                            frame=frame,
                            details={'class': det['class'], 'bbox': det['bbox']}
                        )
            
            # Draw detections
            frame = detector.draw_detections(frame, detections)
        
        # Store latest frame
        with frame_lock:
            latest_frame = frame.copy()
        
        # Encode frame
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if not ret:
            continue
        
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/start_camera', methods=['POST'])
def start_camera():
    """Start camera capture"""
    global camera
    
    try:
        data = request.json or {}
        source = data.get('source', 0)  # 0 for webcam, or RTSP URL
        
        if camera:
            camera.release()
        
        camera = VideoCamera(source)
        return jsonify({'success': True, 'message': 'Camera started'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/stop_camera', methods=['POST'])
def stop_camera():
    """Stop camera capture"""
    global camera
    
    try:
        if camera:
            camera.release()
            camera = None
        return jsonify({'success': True, 'message': 'Camera stopped'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/toggle_detection', methods=['POST'])
def toggle_detection():
    """Toggle detection on/off"""
    global detection_enabled
    
    detection_enabled = not detection_enabled
    return jsonify({
        'success': True,
        'detection_enabled': detection_enabled
    })

@app.route('/api/detections')
def get_detections():
    """Get current detections"""
    with frame_lock:
        return jsonify({
            'detections': current_detections,
            'timestamp': datetime.now().isoformat()
        })

@app.route('/api/statistics')
def get_statistics():
    """Get detection statistics"""
    stats = detector.get_statistics()
    return jsonify(stats)

@app.route('/api/upload_image', methods=['POST'])
def upload_image():
    """Upload and process single image"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Read image
        image_bytes = file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({'error': 'Invalid image'}), 400
        
        # Perform detection
        detections = detector.detect_objects(frame, detection_types=['all'])
        
        # Draw detections
        result_frame = detector.draw_detections(frame, detections)
        
        # Encode result
        _, buffer = cv2.imencode('.jpg', result_frame)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Send alerts if required
        for det in detections:
            if det['alert_required']:
                alert_manager.send_alert(
                    detection_type=det['type'],
                    confidence=det['confidence'],
                    frame=result_frame,
                    details={'class': det['class'], 'bbox': det['bbox']}
                )
        
        return jsonify({
            'success': True,
            'detections': detections,
            'result_image': img_base64
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/alert_history')
def get_alert_history():
    """Get alert history"""
    history = alert_manager.get_alert_history(limit=20)
    return jsonify({'alerts': history})

@app.route('/api/config', methods=['GET', 'POST'])
def config():
    """Get or update configuration"""
    if request.method == 'GET':
        return jsonify({
            'detector_config': detector.config,
            'alert_config': alert_manager.config
        })
    else:
        try:
            data = request.json
            # Update configs (implement as needed)
            return jsonify({'success': True, 'message': 'Config updated'})
        except Exception as e:
            return jsonify({'error': str(e)}), 500

@app.route('/api/health')
def health():
    """Health check endpoint for AWS"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'detector_ready': True,
        'camera_active': camera is not None
    })

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(e):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs('logs', exist_ok=True)
    os.makedirs('static/uploads', exist_ok=True)
    
    # Run app
    # For development
    print("="*50)
    print("Multi-Object Detection System")
    print("="*50)
    print("Starting Flask server...")
    print("Access at: http://localhost:5000")
    print("Or from phone: http://YOUR-IP:5000")
    print("="*50)
    
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
    
    # For production, use gunicorn:
    # gunicorn -w 4 -b 0.0.0.0:5000 app:app