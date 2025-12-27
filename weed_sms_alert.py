"""
Mobile SMS Alert System for Farmers - Weed Detection Alerts
Production-ready implementation with comprehensive error handling and logging
"""

import csv
import json
import logging
import os
from datetime import datetime
from typing import Dict, Optional, List
from dataclasses import dataclass
from enum import Enum
import time
import queue
import threading

# SMS Gateway Integration (Multiple providers supported)
try:
    from twilio.rest import Client as TwilioClient
except ImportError:
    TwilioClient = None

try:
    import requests
except ImportError:
    requests = None


# ============================================================================
# Configuration Management
# ============================================================================

class SMSProvider(Enum):
    TWILIO = "twilio"
    FAST2SMS = "fast2sms"
    AWS_SNS = "aws_sns"
    MOCK = "mock"  # For testing


@dataclass
class SMSConfig:
    """SMS Gateway Configuration"""
    provider: SMSProvider
    account_sid: Optional[str] = None
    auth_token: Optional[str] = None
    from_number: Optional[str] = None
    api_key: Optional[str] = None
    max_retries: int = 3
    retry_delay: int = 5


@dataclass
class SystemConfig:
    """System-wide Configuration"""
    csv_file_path: str
    log_file_path: str = "sms_alerts.log"
    sms_log_path: str = "sms_delivery_log.csv"
    enable_aadhaar_masking: bool = True
    batch_processing: bool = True
    batch_size: int = 100


# ============================================================================
# Data Models
# ============================================================================

@dataclass
class FarmerInfo:
    """Farmer information from CSV"""
    farmer_id: str
    name: str
    aadhaar: str
    phone_number: str

    def get_masked_aadhaar(self) -> str:
        """Return masked Aadhaar number (XXXX-XXXX-1234)"""
        if len(self.aadhaar) >= 4:
            return f"XXXX-XXXX-{self.aadhaar[-4:]}"
        return "XXXX-XXXX-XXXX"


@dataclass
class WeedAlert:
    """Weed detection alert from ML model"""
    alert_id: str
    farmer_id: str
    weed_type: str
    severity: str
    location: str
    timestamp: datetime

    @classmethod
    def from_dict(cls, data: Dict) -> 'WeedAlert':
        """Create WeedAlert from dictionary"""
        timestamp = data.get('timestamp')
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        elif not isinstance(timestamp, datetime):
            timestamp = datetime.now()
        
        return cls(
            alert_id=data['alert_id'],
            farmer_id=data['farmer_id'],
            weed_type=data['weed_type'],
            severity=data['severity'],
            location=data['location'],
            timestamp=timestamp
        )


@dataclass
class SMSDeliveryLog:
    """SMS delivery status log"""
    alert_id: str
    farmer_id: str
    phone_number: str
    status: str  # sent, failed, retrying
    timestamp: datetime
    error_message: Optional[str] = None
    retry_count: int = 0


# ============================================================================
# CSV Database Manager
# ============================================================================

class FarmerDatabase:
    """Manages farmer information from CSV file"""
    
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.farmers: Dict[str, FarmerInfo] = {}
        self._load_farmers()
        self._setup_logger()
    
    def _setup_logger(self):
        self.logger = logging.getLogger('FarmerDatabase')
    
    def _load_farmers(self):
        """Load farmer data from CSV file"""
        try:
            with open(self.csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Strip whitespace from headers and values
                    row = {k.strip(): v.strip() for k, v in row.items()}
                    farmer = FarmerInfo(
                        farmer_id=row['farmer_id'],
                        name=row['name'],
                        aadhaar=row['aadhaar'],
                        phone_number=row['phone_number']
                    )
                    self.farmers[farmer.farmer_id] = farmer
            
            self.logger.info(f"Loaded {len(self.farmers)} farmers from database")
        
        except FileNotFoundError:
            self.logger.error(f"CSV file not found: {self.csv_path}")
            raise
        except Exception as e:
            self.logger.error(f"Error loading farmer database: {e}")
            raise
    
    def get_farmer(self, farmer_id: str) -> Optional[FarmerInfo]:
        """Retrieve farmer information by ID"""
        return self.farmers.get(farmer_id)
    
    def reload(self):
        """Reload farmer data from CSV"""
        self.farmers.clear()
        self._load_farmers()


# ============================================================================
# SMS Gateway Integrations
# ============================================================================

class SMSGateway:
    """Abstract base class for SMS gateways"""
    
    def __init__(self, config: SMSConfig):
        self.config = config
        self.logger = logging.getLogger('SMSGateway')
    
    def send_sms(self, phone_number: str, message: str) -> bool:
        """Send SMS - to be implemented by subclasses"""
        raise NotImplementedError


class TwilioGateway(SMSGateway):
    """Twilio SMS Gateway Implementation"""
    
    def __init__(self, config: SMSConfig):
        super().__init__(config)
        if TwilioClient is None:
            raise ImportError("Twilio library not installed. Install: pip install twilio")
        
        self.client = TwilioClient(config.account_sid, config.auth_token)
    
    def send_sms(self, phone_number: str, message: str) -> bool:
        try:
            msg = self.client.messages.create(
                body=message,
                from_=self.config.from_number,
                to=phone_number
            )
            self.logger.info(f"SMS sent via Twilio - SID: {msg.sid}")
            return True
        except Exception as e:
            self.logger.error(f"Twilio SMS failed: {e}")
            return False


class Fast2SMSGateway(SMSGateway):
    """Fast2SMS Gateway Implementation"""
    
    def __init__(self, config: SMSConfig):
        super().__init__(config)
        if requests is None:
            raise ImportError("Requests library not installed. Install: pip install requests")
        
        self.api_url = "https://www.fast2sms.com/dev/bulkV2"
    
    def send_sms(self, phone_number: str, message: str) -> bool:
        try:
            # Remove country code if present
            phone = phone_number.replace('+91', '').replace('-', '').strip()
            
            payload = {
                'authorization': self.config.api_key,
                'message': message,
                'numbers': phone,
                'route': 'q',
                'sender_id': 'FSTSMS'
            }
            
            response = requests.post(self.api_url, data=payload, timeout=10)
            response.raise_for_status()
            
            result = response.json()
            if result.get('return'):
                self.logger.info(f"SMS sent via Fast2SMS to {phone}")
                return True
            else:
                self.logger.error(f"Fast2SMS failed: {result}")
                return False
        
        except Exception as e:
            self.logger.error(f"Fast2SMS error: {e}")
            return False


class MockSMSGateway(SMSGateway):
    """Mock SMS Gateway for Testing"""
    
    def send_sms(self, phone_number: str, message: str) -> bool:
        self.logger.info(f"[MOCK] SMS to {phone_number}: {message[:50]}...")
        return True


def create_sms_gateway(config: SMSConfig) -> SMSGateway:
    """Factory method to create appropriate SMS gateway"""
    if config.provider == SMSProvider.TWILIO:
        return TwilioGateway(config)
    elif config.provider == SMSProvider.FAST2SMS:
        return Fast2SMSGateway(config)
    elif config.provider == SMSProvider.MOCK:
        return MockSMSGateway(config)
    else:
        raise ValueError(f"Unsupported SMS provider: {config.provider}")


# ============================================================================
# SMS Message Generator
# ============================================================================

class SMSMessageGenerator:
    """Generates SMS content from alert data"""
    
    TEMPLATE = """Dear {name},
Weed Alert Detected in your field.
Weed Type: {weed_type}
Severity: {severity}
Location: {location}
Time: {timestamp}
Please take necessary action.
- Smart Agriculture Alert System"""
    
    @staticmethod
    def generate_message(farmer: FarmerInfo, alert: WeedAlert) -> str:
        """Generate SMS message from template"""
        timestamp_str = alert.timestamp.strftime("%d-%b-%Y %I:%M %p")
        
        message = SMSMessageGenerator.TEMPLATE.format(
            name=farmer.name,
            weed_type=alert.weed_type,
            severity=alert.severity,
            location=alert.location,
            timestamp=timestamp_str
        )
        
        return message


# ============================================================================
# Delivery Logger
# ============================================================================

class DeliveryLogger:
    """Logs SMS delivery status"""
    
    def __init__(self, log_path: str):
        self.log_path = log_path
        self._init_log_file()
    
    def _init_log_file(self):
        """Initialize CSV log file with headers"""
        if not os.path.exists(self.log_path):
            with open(self.log_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'alert_id', 'farmer_id', 'phone_number', 'status',
                    'timestamp', 'retry_count', 'error_message'
                ])
    
    def log_delivery(self, log_entry: SMSDeliveryLog):
        """Log SMS delivery attempt"""
        with open(self.log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                log_entry.alert_id,
                log_entry.farmer_id,
                log_entry.phone_number,
                log_entry.status,
                log_entry.timestamp.isoformat(),
                log_entry.retry_count,
                log_entry.error_message or ''
            ])


# ============================================================================
# Alert Processing Engine
# ============================================================================

class AlertProcessor:
    """Main alert processing and SMS dispatch engine"""
    
    def __init__(self, system_config: SystemConfig, sms_config: SMSConfig):
        self.system_config = system_config
        self.sms_config = sms_config
        
        # Initialize components
        self.farmer_db = FarmerDatabase(system_config.csv_file_path)
        self.sms_gateway = create_sms_gateway(sms_config)
        self.delivery_logger = DeliveryLogger(system_config.sms_log_path)
        self.message_generator = SMSMessageGenerator()
        
        # Setup logging
        self._setup_logging()
        
        # Alert queue for batch processing
        self.alert_queue = queue.Queue()
        self.processing_thread = None
        self.running = False
    
    def _setup_logging(self):
        """Configure system logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.system_config.log_file_path),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('AlertProcessor')
    
    def process_alert(self, alert: WeedAlert) -> bool:
        """Process a single weed detection alert"""
        self.logger.info(f"Processing alert {alert.alert_id} for farmer {alert.farmer_id}")
        
        # Retrieve farmer information
        farmer = self.farmer_db.get_farmer(alert.farmer_id)
        if not farmer:
            self.logger.error(f"Farmer {alert.farmer_id} not found in database")
            self._log_failed_delivery(alert, "Farmer not found")
            return False
        
        # Generate SMS message
        message = self.message_generator.generate_message(farmer, alert)
        
        # Send SMS with retry logic
        return self._send_with_retry(farmer, alert, message)
    
    def _send_with_retry(self, farmer: FarmerInfo, alert: WeedAlert, 
                         message: str) -> bool:
        """Send SMS with automatic retry on failure"""
        for attempt in range(self.sms_config.max_retries):
            try:
                success = self.sms_gateway.send_sms(farmer.phone_number, message)
                
                if success:
                    self._log_successful_delivery(farmer, alert)
                    return True
                else:
                    if attempt < self.sms_config.max_retries - 1:
                        self.logger.warning(
                            f"SMS failed (attempt {attempt + 1}), retrying..."
                        )
                        time.sleep(self.sms_config.retry_delay)
                    else:
                        self._log_failed_delivery(alert, "Max retries exceeded", attempt + 1)
            
            except Exception as e:
                self.logger.error(f"Error sending SMS: {e}")
                if attempt < self.sms_config.max_retries - 1:
                    time.sleep(self.sms_config.retry_delay)
                else:
                    self._log_failed_delivery(alert, str(e), attempt + 1)
        
        return False
    
    def _log_successful_delivery(self, farmer: FarmerInfo, alert: WeedAlert):
        """Log successful SMS delivery"""
        log_entry = SMSDeliveryLog(
            alert_id=alert.alert_id,
            farmer_id=farmer.farmer_id,
            phone_number=farmer.phone_number,
            status="sent",
            timestamp=datetime.now()
        )
        self.delivery_logger.log_delivery(log_entry)
        self.logger.info(f"SMS successfully sent to {farmer.name}")
    
    def _log_failed_delivery(self, alert: WeedAlert, error: str, 
                            retry_count: int = 0):
        """Log failed SMS delivery"""
        log_entry = SMSDeliveryLog(
            alert_id=alert.alert_id,
            farmer_id=alert.farmer_id,
            phone_number="unknown",
            status="failed",
            timestamp=datetime.now(),
            error_message=error,
            retry_count=retry_count
        )
        self.delivery_logger.log_delivery(log_entry)
    
    def start_batch_processing(self):
        """Start background thread for batch alert processing"""
        if self.system_config.batch_processing:
            self.running = True
            self.processing_thread = threading.Thread(
                target=self._batch_processor,
                daemon=True
            )
            self.processing_thread.start()
            self.logger.info("Batch processing started")
    
    def _batch_processor(self):
        """Background batch processor"""
        while self.running:
            alerts_batch = []
            
            # Collect batch
            while len(alerts_batch) < self.system_config.batch_size:
                try:
                    alert = self.alert_queue.get(timeout=1)
                    alerts_batch.append(alert)
                except queue.Empty:
                    break
            
            # Process batch
            if alerts_batch:
                for alert in alerts_batch:
                    self.process_alert(alert)
            
            time.sleep(1)
    
    def queue_alert(self, alert: WeedAlert):
        """Add alert to processing queue"""
        self.alert_queue.put(alert)
    
    def stop(self):
        """Stop batch processing"""
        self.running = False
        if self.processing_thread:
            self.processing_thread.join()


# ============================================================================
# Main Application Interface
# ============================================================================

class WeedAlertSMSSystem:
    """Main application interface"""
    
    def __init__(self, system_config: SystemConfig, sms_config: SMSConfig):
        self.processor = AlertProcessor(system_config, sms_config)
        self.logger = logging.getLogger('WeedAlertSMSSystem')
    
    def handle_ml_alert(self, alert_data: Dict):
        """Handle incoming alert from ML model"""
        try:
            alert = WeedAlert.from_dict(alert_data)
            self.logger.info(f"Received alert: {alert.alert_id}")
            
            if self.processor.system_config.batch_processing:
                self.processor.queue_alert(alert)
            else:
                self.processor.process_alert(alert)
        
        except Exception as e:
            self.logger.error(f"Error handling ML alert: {e}")
    
    def start(self):
        """Start the alert system"""
        self.processor.start_batch_processing()
        self.logger.info("Weed Alert SMS System started")
    
    def stop(self):
        """Stop the alert system"""
        self.processor.stop()
        self.logger.info("Weed Alert SMS System stopped")


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    # System configuration
    system_config = SystemConfig(
        csv_file_path="farmers.csv",
        log_file_path="sms_alerts.log",
        sms_log_path="sms_delivery_log.csv",
        batch_processing=True,
        batch_size=50
    )
    
    # SMS Gateway configuration (choose provider)
    sms_config = SMSConfig(
        provider=SMSProvider.MOCK,  # Change to TWILIO or FAST2SMS in production
        # For Twilio:
        # account_sid="your_account_sid",
        # auth_token="your_auth_token",
        # from_number="+1234567890",
        # For Fast2SMS:
        # api_key="your_api_key",
        max_retries=3,
        retry_delay=5
    )
    
    # Initialize system
    alert_system = WeedAlertSMSSystem(system_config, sms_config)
    alert_system.start()
    
    # Example: Simulate ML model generating alerts
    example_alert = {
        'alert_id': 'ALERT-2024-001',
        'farmer_id': 'F001',
        'weed_type': 'Broadleaf Weed',
        'severity': 'High',
        'location': 'Field-A, Plot-3',
        'timestamp': datetime.now().isoformat()
    }
    
    alert_system.handle_ml_alert(example_alert)
    
    # Keep system running
    print("Alert system running. Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        alert_system.stop()
        print("System stopped.")
