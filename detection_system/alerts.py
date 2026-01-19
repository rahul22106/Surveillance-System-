"""
Alert System - Multi-channel notifications
Supports: Email, SMS, Webhooks, Push Notifications
"""

import smtplib
import requests
import json
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from datetime import datetime
import os
from pathlib import Path
import base64
import cv2

logger = logging.getLogger(__name__)

class AlertManager:
    def __init__(self, config_path='config/alert_config.json'):
        """Initialize alert manager with configuration"""
        self.config = self.load_config(config_path)
        self.alert_history = []
        
    def load_config(self, config_path):
        """Load alert configuration"""
        try:
            if Path(config_path).exists():
                with open(config_path, 'r') as f:
                    return json.load(f)
            else:
                # Default configuration
                return {
                    'email': {
                        'enabled': False,
                        'smtp_server': 'smtp.gmail.com',
                        'smtp_port': 587,
                        'sender_email': '',
                        'sender_password': '',
                        'recipient_emails': []
                    },
                    'sms': {
                        'enabled': False,
                        'service': 'twilio',
                        'account_sid': '',
                        'auth_token': '',
                        'from_number': '',
                        'to_numbers': []
                    },
                    'webhook': {
                        'enabled': False,
                        'urls': []
                    },
                    'telegram': {
                        'enabled': False,
                        'bot_token': '',
                        'chat_ids': []
                    },
                    'sns': {
                        'enabled': False,
                        'topic_arn': '',
                        'region': 'us-east-1'
                    }
                }
        except Exception as e:
            logger.error(f"Config load error: {e}")
            return {}
    
    def send_alert(self, detection_type, confidence, frame=None, details=None):
        """
        Send alert through all enabled channels
        Args:
            detection_type: 'fire', 'fall', or 'vehicle'
            confidence: detection confidence score
            frame: image frame (numpy array)
            details: additional detection details
        """
        timestamp = datetime.now()
        
        # Create alert message
        alert_data = {
            'type': detection_type,
            'confidence': confidence,
            'timestamp': timestamp.isoformat(),
            'details': details or {}
        }
        
        # Add to history
        self.alert_history.append(alert_data)
        
        # Send through enabled channels
        results = {}
        
        if self.config.get('email', {}).get('enabled'):
            results['email'] = self.send_email_alert(alert_data, frame)
        
        if self.config.get('sms', {}).get('enabled'):
            results['sms'] = self.send_sms_alert(alert_data)
        
        if self.config.get('webhook', {}).get('enabled'):
            results['webhook'] = self.send_webhook_alert(alert_data, frame)
        
        if self.config.get('telegram', {}).get('enabled'):
            results['telegram'] = self.send_telegram_alert(alert_data, frame)
        
        if self.config.get('sns', {}).get('enabled'):
            results['sns'] = self.send_sns_alert(alert_data)
        
        logger.info(f"Alert sent: {detection_type} | Channels: {results}")
        
        return results
    
    def send_email_alert(self, alert_data, frame=None):
        """Send email alert"""
        try:
            config = self.config['email']
            
            msg = MIMEMultipart()
            msg['From'] = config['sender_email']
            msg['To'] = ', '.join(config['recipient_emails'])
            msg['Subject'] = f"🚨 ALERT: {alert_data['type'].upper()} Detected!"
            
            # Email body
            body = f"""
            <html>
            <body>
                <h2 style="color: red;">⚠️ Detection Alert</h2>
                <p><strong>Type:</strong> {alert_data['type'].upper()}</p>
                <p><strong>Confidence:</strong> {alert_data['confidence']:.2%}</p>
                <p><strong>Time:</strong> {alert_data['timestamp']}</p>
                <p><strong>Details:</strong> {json.dumps(alert_data['details'], indent=2)}</p>
                <hr>
                <p style="color: gray;">This is an automated alert from your detection system.</p>
            </body>
            </html>
            """
            
            msg.attach(MIMEText(body, 'html'))
            
            # Attach image if provided
            if frame is not None:
                # Convert frame to JPEG
                _, buffer = cv2.imencode('.jpg', frame)
                img_data = buffer.tobytes()
                
                image = MIMEImage(img_data, name='detection.jpg')
                msg.attach(image)
            
            # Send email
            with smtplib.SMTP(config['smtp_server'], config['smtp_port']) as server:
                server.starttls()
                server.login(config['sender_email'], config['sender_password'])
                server.send_message(msg)
            
            logger.info("Email alert sent successfully")
            return {'success': True, 'message': 'Email sent'}
            
        except Exception as e:
            logger.error(f"Email alert error: {e}")
            return {'success': False, 'error': str(e)}
    
    def send_sms_alert(self, alert_data):
        """Send SMS alert using Twilio"""
        try:
            config = self.config['sms']
            
            # Import Twilio
            from twilio.rest import Client
            
            client = Client(config['account_sid'], config['auth_token'])
            
            message_body = f"""
🚨 ALERT: {alert_data['type'].upper()} detected!
Confidence: {alert_data['confidence']:.0%}
Time: {alert_data['timestamp']}
            """.strip()
            
            # Send to all configured numbers
            for to_number in config['to_numbers']:
                message = client.messages.create(
                    body=message_body,
                    from_=config['from_number'],
                    to=to_number
                )
                logger.info(f"SMS sent to {to_number}: {message.sid}")
            
            return {'success': True, 'message': 'SMS sent'}
            
        except Exception as e:
            logger.error(f"SMS alert error: {e}")
            return {'success': False, 'error': str(e)}
    
    def send_webhook_alert(self, alert_data, frame=None):
        """Send webhook alert (e.g., to Slack, Discord, custom API)"""
        try:
            config = self.config['webhook']
            
            # Prepare payload
            payload = {
                'alert': alert_data,
                'image': None
            }
            
            # Encode image if provided
            if frame is not None:
                _, buffer = cv2.imencode('.jpg', frame)
                img_base64 = base64.b64encode(buffer).decode('utf-8')
                payload['image'] = img_base64
            
            # Send to all configured webhooks
            responses = []
            for url in config['urls']:
                response = requests.post(
                    url,
                    json=payload,
                    timeout=10
                )
                responses.append({
                    'url': url,
                    'status': response.status_code,
                    'response': response.text
                })
                logger.info(f"Webhook sent to {url}: {response.status_code}")
            
            return {'success': True, 'responses': responses}
            
        except Exception as e:
            logger.error(f"Webhook alert error: {e}")
            return {'success': False, 'error': str(e)}
    
    def send_telegram_alert(self, alert_data, frame=None):
        """Send Telegram bot alert"""
        try:
            config = self.config['telegram']
            
            bot_token = config['bot_token']
            chat_ids = config['chat_ids']
            
            message = f"""
🚨 *ALERT DETECTED*

*Type:* {alert_data['type'].upper()}
*Confidence:* {alert_data['confidence']:.0%}
*Time:* {alert_data['timestamp']}

_Automated detection alert_
            """.strip()
            
            # Send to all chat IDs
            for chat_id in chat_ids:
                # Send message
                url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
                requests.post(url, json={
                    'chat_id': chat_id,
                    'text': message,
                    'parse_mode': 'Markdown'
                })
                
                # Send image if provided
                if frame is not None:
                    _, buffer = cv2.imencode('.jpg', frame)
                    
                    url = f"https://api.telegram.org/bot{bot_token}/sendPhoto"
                    files = {'photo': ('detection.jpg', buffer.tobytes(), 'image/jpeg')}
                    data = {'chat_id': chat_id}
                    
                    requests.post(url, files=files, data=data)
                
                logger.info(f"Telegram alert sent to chat {chat_id}")
            
            return {'success': True, 'message': 'Telegram alert sent'}
            
        except Exception as e:
            logger.error(f"Telegram alert error: {e}")
            return {'success': False, 'error': str(e)}
    
    def send_sns_alert(self, alert_data):
        """Send AWS SNS alert"""
        try:
            import boto3
            
            config = self.config['sns']
            
            sns_client = boto3.client('sns', region_name=config['region'])
            
            message = f"""
🚨 ALERT: {alert_data['type'].upper()} Detected!

Confidence: {alert_data['confidence']:.0%}
Time: {alert_data['timestamp']}

Details: {json.dumps(alert_data['details'], indent=2)}
            """.strip()
            
            response = sns_client.publish(
                TopicArn=config['topic_arn'],
                Subject=f"Detection Alert: {alert_data['type'].upper()}",
                Message=message
            )
            
            logger.info(f"SNS alert sent: {response['MessageId']}")
            return {'success': True, 'message_id': response['MessageId']}
            
        except Exception as e:
            logger.error(f"SNS alert error: {e}")
            return {'success': False, 'error': str(e)}
    
    def get_alert_history(self, limit=10):
        """Get recent alert history"""
        return self.alert_history[-limit:]
    
    def clear_history(self):
        """Clear alert history"""
        self.alert_history = []
        logger.info("Alert history cleared")

# Simple webhook handler for testing
def create_test_webhook_server():
    """Create a simple Flask server to receive webhooks (for testing)"""
    from flask import Flask, request
    
    app = Flask(__name__)
    
    @app.route('/webhook', methods=['POST'])
    def webhook():
        data = request.json
        print(f"Webhook received: {json.dumps(data, indent=2)}")
        return {'status': 'received'}, 200
    
    return app

if __name__ == "__main__":
    # Test alert manager
    manager = AlertManager()
    print("Alert Manager initialized")
    print(f"Enabled channels: {[k for k, v in manager.config.items() if v.get('enabled')]}")