#!/usr/bin/env python3
"""
Script to prepare training data for OpenAI GPT-4V fine-tuning.
This script:
1. Processes images from the specified directory
2. Uploads them to a server to generate direct links
3. Creates a formatted JSONL file for OpenAI GPT-4V fine-tuning
"""

import os
import json
import argparse
import requests
from pathlib import Path
from PIL import Image
import io
import base64
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import re

# Define categories based on filename patterns
CATEGORIES = {
    "Adson brown forceps": r"Adson brown\s+",
    "Adson smooth forceps": r"Adson smooth forceps\s+",
    "Jacobson clamp curved": r"Extra fine point Jacobson clamp curved\s+",
    "Hemostatic clamp curved": r"Hemastatclamp curved\s+",
    "Jackson right angle clamp": r"Jackson right angle clamp\s+",
    "Mosquito clamp curved": r"MosquitoClampCurved\s+"
}

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Prepare training data for OpenAI GPT-4V fine-tuning")
    parser.add_argument(
        "--photos-dir", 
        type=str, 
        default="./photos",
        help="Path to directory containing photos"
    )
    parser.add_argument(
        "--output-file", 
        type=str, 
        default="./training_data.jsonl",
        help="Output JSONL file for training data"
    )
    parser.add_argument(
        "--upload-endpoint", 
        type=str, 
        required=True,
        help="Endpoint for uploading images"
    )
    parser.add_argument(
        "--max-workers", 
        type=int, 
        default=5,
        help="Maximum number of concurrent uploads"
    )
    parser.add_argument(
        "--resize", 
        type=int, 
        default=1024,
        help="Resize images to this maximum dimension while preserving aspect ratio"
    )
    parser.add_argument(
        "--api-key", 
        type=str, 
        help="API key for the upload service (if required)"
    )
    parser.add_argument(
        "--dry-run", 
        action="store_true",
        help="Process files without uploading"
    )
    return parser.parse_args()

def get_instrument_category(filename):
    """Determine the category of surgical instrument from the filename."""
    for category, pattern in CATEGORIES.items():
        if re.search(pattern, filename):
            return category
    return "Unknown"

def resize_image(image_path, max_size):
    """Resize image while preserving aspect ratio."""
    try:
        with Image.open(image_path) as img:
            # Calculate new dimensions
            width, height = img.size
            if width > height:
                if width > max_size:
                    new_width = max_size
                    new_height = int(height * (max_size / width))
            else:
                if height > max_size:
                    new_height = max_size
                    new_width = int(width * (max_size / height))
                else:
                    return img.copy()
            
            # Resize and return
            return img.resize((new_width, new_height), Image.LANCZOS)
    except Exception as e:
        print(f"Error resizing {image_path}: {e}")
        return None

def upload_image(image_path, args):
    """Upload image to server and return direct link."""
    if args.dry_run:
        return f"http://example.com/images/{os.path.basename(image_path)}"
    
    # Resize image before uploading
    img = resize_image(image_path, args.resize)
    if img is None:
        return None
    
    # Convert to bytes
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    img_byte_arr = img_byte_arr.getvalue()
    
    # Prepare headers
    headers = {}
    if args.api_key:
        headers['Authorization'] = f'Bearer {args.api_key}'
    
    # Upload image
    try:
        response = requests.post(
            args.upload_endpoint,
            files={'file': (os.path.basename(image_path), img_byte_arr, 'image/jpeg')},
            headers=headers
        )
        response.raise_for_status()
        result = response.json()
        
        # This will need to be adjusted based on the actual API response format
        image_url = result.get('url')
        if not image_url:
            # Try common response formats
            image_url = result.get('data', {}).get('url')
            if not image_url:
                image_url = result.get('image_url')
                if not image_url:
                    raise ValueError(f"Could not extract image URL from response: {result}")
        
        return image_url
    except Exception as e:
        print(f"Error uploading {image_path}: {e}")
        return None

def create_training_example(image_path, image_url):
    """Create a training example for GPT-4V fine-tuning."""
    instrument_category = get_instrument_category(os.path.basename(image_path))
    
    # Create the training example in the format expected by OpenAI
    training_example = {
        "messages": [
            {"role": "system", "content": "You are a surgical instrument identification assistant. When shown an image of a surgical instrument, identify the type of instrument accurately."},
            {"role": "user", "content": [
                {"type": "text", "text": "What is this surgical instrument?"},
                {"type": "image_url", "image_url": {"url": image_url}}
            ]},
            {"role": "assistant", "content": f"This is a {instrument_category}. It's used in surgical procedures for grasping, clamping, or manipulating tissues."}
        ]
    }
    
    return training_example

def process_images(args):
    """Process all images in the directory."""
    # Find all image files recursively
    photos_dir = Path(args.photos_dir)
    image_files = []
    for ext in ['.jpg', '.jpeg', '.png']:
        image_files.extend(list(photos_dir.glob(f"**/*{ext}")))
    
    print(f"Found {len(image_files)} images")
    
    # Upload images and create training examples
    training_examples = []
    
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        # Map upload_image function to all image files
        future_to_image = {
            executor.submit(upload_image, str(image_file), args): image_file 
            for image_file in image_files
        }
        
        for future in tqdm(future_to_image, total=len(image_files), desc="Uploading images"):
            image_file = future_to_image[future]
            try:
                image_url = future.result()
                if image_url:
                    example = create_training_example(str(image_file), image_url)
                    training_examples.append(example)
            except Exception as e:
                print(f"Error processing {image_file}: {e}")
    
    print(f"Created {len(training_examples)} training examples")
    
    # Write training examples to JSONL file
    with open(args.output_file, 'w') as f:
        for example in training_examples:
            f.write(json.dumps(example) + '\n')
    
    print(f"Training data written to {args.output_file}")

def main():
    args = parse_args()
    process_images(args)

if __name__ == "__main__":
    main()
