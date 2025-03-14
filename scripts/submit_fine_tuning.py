#!/usr/bin/env python3
"""
Script to submit a fine-tuning job to OpenAI for GPT-4V.
"""

import os
import sys
import json
import argparse
import time
from openai import OpenAI

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Submit a fine-tuning job to OpenAI")
    parser.add_argument(
        "--training-file", 
        type=str, 
        required=True,
        help="Path to JSONL file containing training data"
    )
    parser.add_argument(
        "--validation-file", 
        type=str, 
        help="Path to JSONL file containing validation data (optional)"
    )
    parser.add_argument(
        "--model", 
        type=str, 
        default="gpt-4o",
        help="Base model to fine-tune"
    )
    parser.add_argument(
        "--suffix", 
        type=str, 
        default="surgical-instruments",
        help="Suffix to add to the fine-tuned model name"
    )
    parser.add_argument(
        "--api-key", 
        type=str, 
        help="OpenAI API key (if not set in OPENAI_API_KEY environment variable)"
    )
    parser.add_argument(
        "--n-epochs", 
        type=int, 
        default=3,
        help="Number of epochs to train for"
    )
    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=1,
        help="Batch size for training"
    )
    parser.add_argument(
        "--learning-rate-multiplier", 
        type=float, 
        default=1.0,
        help="Learning rate multiplier"
    )
    return parser.parse_args()

def validate_jsonl(file_path):
    """Validate that the JSONL file is formatted correctly for GPT-4V fine-tuning."""
    print(f"Validating {file_path}...")
    
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        if not lines:
            print(f"Error: {file_path} is empty")
            return False
        
        for i, line in enumerate(lines):
            try:
                data = json.loads(line)
                
                # Check required fields
                if "messages" not in data:
                    print(f"Error: Line {i+1} is missing 'messages' field")
                    return False
                
                messages = data["messages"]
                if not isinstance(messages, list) or len(messages) < 2:
                    print(f"Error: Line {i+1} should have at least 2 messages")
                    return False
                
                # Check for system message
                if messages[0]["role"] != "system":
                    print(f"Warning: Line {i+1} first message is not a system message")
                
                # Check for user message with image
                user_message = None
                for msg in messages:
                    if msg["role"] == "user":
                        user_message = msg
                        break
                
                if not user_message:
                    print(f"Error: Line {i+1} is missing a user message")
                    return False
                
                content = user_message["content"]
                if not isinstance(content, list):
                    print(f"Error: Line {i+1} user message content should be a list with text and image")
                    return False
                
                has_image = False
                for item in content:
                    if item.get("type") == "image_url":
                        has_image = True
                        if "image_url" not in item or "url" not in item["image_url"]:
                            print(f"Error: Line {i+1} has invalid image_url format")
                            return False
                
                if not has_image:
                    print(f"Error: Line {i+1} user message does not contain an image")
                    return False
                
                # Check for assistant message
                assistant_msg = None
                for msg in messages:
                    if msg["role"] == "assistant":
                        assistant_msg = msg
                        break
                
                if not assistant_msg:
                    print(f"Error: Line {i+1} is missing an assistant message")
                    return False
                
            except json.JSONDecodeError:
                print(f"Error: Line {i+1} is not valid JSON")
                return False
            except Exception as e:
                print(f"Error on line {i+1}: {e}")
                return False
        
        print(f"Validation successful for {file_path} ({len(lines)} examples)")
        return True
    except Exception as e:
        print(f"Error validating {file_path}: {e}")
        return False

def upload_file(client, file_path, purpose="fine-tune"):
    """Upload a file to OpenAI."""
    print(f"Uploading {file_path}...")
    
    try:
        with open(file_path, "rb") as file:
            response = client.files.create(
                file=file,
                purpose=purpose
            )
        
        file_id = response.id
        print(f"Uploaded {file_path} as {file_id}")
        return file_id
    except Exception as e:
        print(f"Error uploading {file_path}: {e}")
        return None

def wait_for_file_processing(client, file_id, timeout=600):
    """Wait for a file to be processed by OpenAI."""
    print(f"Waiting for file {file_id} to be processed...")
    
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            file_info = client.files.retrieve(file_id)
            if file_info.status == "processed":
                print(f"File {file_id} processed successfully")
                return True
            elif file_info.status == "error":
                print(f"Error processing file {file_id}: {file_info.error}")
                return False
            
            print(f"File status: {file_info.status}. Waiting...")
            time.sleep(5)
        except Exception as e:
            print(f"Error checking file status: {e}")
            time.sleep(5)
    
    print(f"Timeout waiting for file {file_id} to be processed")
    return False

def create_fine_tuning_job(client, args, training_file_id, validation_file_id=None):
    """Create a fine-tuning job."""
    print("Creating fine-tuning job...")
    
    try:
        job_params = {
            "training_file": training_file_id,
            "model": args.model,
            "suffix": args.suffix,
            "hyperparameters": {
                "n_epochs": args.n_epochs,
                "batch_size": args.batch_size,
                "learning_rate_multiplier": args.learning_rate_multiplier
            }
        }
        
        if validation_file_id:
            job_params["validation_file"] = validation_file_id
        
        response = client.fine_tuning.jobs.create(**job_params)
        
        job_id = response.id
        print(f"Created fine-tuning job {job_id}")
        return job_id
    except Exception as e:
        print(f"Error creating fine-tuning job: {e}")
        return None

def monitor_fine_tuning_job(client, job_id, check_interval=60):
    """Monitor the status of a fine-tuning job."""
    print(f"Monitoring fine-tuning job {job_id}...")
    
    while True:
        try:
            job_info = client.fine_tuning.jobs.retrieve(job_id)
            status = job_info.status
            
            print(f"Job status: {status}")
            
            if status == "succeeded":
                print(f"Fine-tuning completed successfully!")
                print(f"Fine-tuned model: {job_info.fine_tuned_model}")
                return True
            elif status in ["failed", "cancelled"]:
                print(f"Fine-tuning failed with status: {status}")
                if hasattr(job_info, "error") and job_info.error:
                    print(f"Error: {job_info.error}")
                return False
            
            # Print training metrics if available
            if hasattr(job_info, "training_metrics") and job_info.training_metrics:
                metrics = job_info.training_metrics
                print(f"Current metrics:")
                for key, value in metrics.items():
                    print(f"  {key}: {value}")
            
            print(f"Waiting {check_interval} seconds for next status check...")
            time.sleep(check_interval)
        except Exception as e:
            print(f"Error monitoring job: {e}")
            time.sleep(check_interval)

def main():
    args = parse_args()
    
    # Get API key
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OpenAI API key not provided. Set the OPENAI_API_KEY environment variable or use --api-key")
        sys.exit(1)
    
    # Create OpenAI client
    client = OpenAI(api_key=api_key)
    
    # Validate training file
    if not validate_jsonl(args.training_file):
        print("Training file validation failed")
        sys.exit(1)
    
    # Validate validation file if provided
    if args.validation_file and not validate_jsonl(args.validation_file):
        print("Validation file validation failed")
        sys.exit(1)
    
    # Upload training file
    training_file_id = upload_file(client, args.training_file)
    if not training_file_id:
        print("Failed to upload training file")
        sys.exit(1)
    
    # Wait for training file to be processed
    if not wait_for_file_processing(client, training_file_id):
        print("Training file processing failed")
        sys.exit(1)
    
    # Upload validation file if provided
    validation_file_id = None
    if args.validation_file:
        validation_file_id = upload_file(client, args.validation_file)
        if not validation_file_id:
            print("Failed to upload validation file")
            sys.exit(1)
        
        # Wait for validation file to be processed
        if not wait_for_file_processing(client, validation_file_id):
            print("Validation file processing failed")
            sys.exit(1)
    
    # Create fine-tuning job
    job_id = create_fine_tuning_job(client, args, training_file_id, validation_file_id)
    if not job_id:
        print("Failed to create fine-tuning job")
        sys.exit(1)
    
    # Monitor fine-tuning job
    monitor_fine_tuning_job(client, job_id)

if __name__ == "__main__":
    main()
