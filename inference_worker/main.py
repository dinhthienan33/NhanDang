import os
import json
import time
import logging
from confluent_kafka import Consumer, Producer, KafkaError
import requests
import numpy as np
import cv2

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("inference_worker")

# Kafka configuration
KAFKA_BOOTSTRAP_SERVERS = os.environ.get('KAFKA_BOOTSTRAP_SERVERS', 'kafka:9092')
TRITON_URL = os.environ.get('TRITON_URL', 'triton:8000')

# Kafka topics
INPUT_TOPIC = 'image_processing_requests'
OUTPUT_TOPIC = 'image_processing_results'

def delivery_report(err, msg):
    """Callback for producer to report delivery result."""
    if err is not None:
        logger.error(f'Message delivery failed: {err}')
    else:
        logger.info(f'Message delivered to {msg.topic()} [{msg.partition()}]')

def process_image(image_data, request_id):
    """
    Process the image using Triton Inference Server.
    In a real implementation, this would use the Triton Client API.
    For simplicity, we're using a direct HTTP request.
    """
    try:
        # Here we would convert the image data to the format expected by the model
        # and send it to Triton Inference Server
        
        # For demonstration, we'll use a direct request to our FastAPI backend
        # which handles the model inference
        url = f'http://backend:8000/process'
        files = {'file': ('image.jpg', image_data, 'image/jpeg')}
        
        response = requests.post(url, files=files)
        
        if response.status_code != 200:
            logger.error(f"Failed to process image: {response.text}")
            return None
            
        # Return the processed image data
        return response.content
        
    except Exception as e:
        logger.exception(f"Error processing image: {e}")
        return None

def main():
    """Main function to consume messages and process images."""
    # Create Producer instance
    producer = Producer({'bootstrap.servers': KAFKA_BOOTSTRAP_SERVERS})
    
    # Create Consumer instance
    consumer = Consumer({
        'bootstrap.servers': KAFKA_BOOTSTRAP_SERVERS,
        'group.id': 'inference_worker_group',
        'auto.offset.reset': 'earliest'
    })
    
    # Subscribe to input topic
    consumer.subscribe([INPUT_TOPIC])
    
    try:
        logger.info("Starting inference worker")
        
        while True:
            # Poll for messages
            msg = consumer.poll(timeout=1.0)
            
            if msg is None:
                continue
                
            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    logger.info(f'Reached end of partition {msg.partition()}')
                else:
                    logger.error(f'Error: {msg.error()}')
            else:
                try:
                    # Parse the message value
                    message_data = json.loads(msg.value())
                    request_id = message_data.get('request_id')
                    image_data = message_data.get('image_data')
                    
                    logger.info(f"Processing request {request_id}")
                    
                    # Process the image
                    result = process_image(image_data, request_id)
                    
                    if result:
                        # Prepare response
                        response = {
                            'request_id': request_id,
                            'status': 'success',
                            'result': result
                        }
                        
                        # Send result to output topic
                        producer.produce(
                            OUTPUT_TOPIC, 
                            key=request_id,
                            value=json.dumps(response),
                            callback=delivery_report
                        )
                        producer.flush()
                        
                        logger.info(f"Processed request {request_id}")
                    else:
                        # Send failure message
                        response = {
                            'request_id': request_id,
                            'status': 'error',
                            'error': 'Failed to process image'
                        }
                        
                        producer.produce(
                            OUTPUT_TOPIC, 
                            key=request_id,
                            value=json.dumps(response),
                            callback=delivery_report
                        )
                        producer.flush()
                        
                        logger.error(f"Failed to process request {request_id}")
                        
                except Exception as e:
                    logger.exception(f"Error processing message: {e}")
                    
    except KeyboardInterrupt:
        logger.info("Shutting down inference worker")
    finally:
        # Close down consumer
        consumer.close()

if __name__ == "__main__":
    # Wait for Kafka to be ready
    time.sleep(10)
    main() 