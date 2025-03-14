# Surgical Instrument GPT-4V Fine-Tuning

This repository contains scripts and tools to prepare a dataset of surgical instrument images for fine-tuning OpenAI's GPT-4V model. The goal is to create a specialized model that can accurately identify different types of surgical instruments from images.

## Project Structure

```
training/
├── README.md              # This file
├── photos/                # Directory containing surgical instrument photos
│   └── Philadelphia, March 12, 2025/  # Photos organized by collection
│       ├── Adson brown forceps images
│       ├── Adson smooth forceps images
│       ├── Jacobson clamp images
│       └── ...
└── scripts/               # Scripts for data preparation and model fine-tuning
    ├── prepare_training_data.py  # Script to process images and create training data
    └── submit_fine_tuning.py     # Script to submit fine-tuning job to OpenAI
```

## Prerequisites

Before running the scripts, ensure you have the following dependencies installed:

```bash
pip install openai pillow tqdm requests
```

## Step 1: Prepare Training Data

The first step is to process the surgical instrument images and create a JSONL file suitable for GPT-4V fine-tuning.

```bash
python scripts/prepare_training_data.py \
  --photos-dir ./photos \
  --output-file ./training_data.jsonl \
  --upload-endpoint https://your-image-hosting-service.com/upload \
  --api-key your-api-key-if-required
```

This script will:
1. Process all images in the photos directory
2. Upload them to your specified image hosting service
3. Create a JSONL file with training examples in the format required by OpenAI

### Command Line Options

- `--photos-dir`: Path to directory containing photos (default: ./photos)
- `--output-file`: Output JSONL file for training data (default: ./training_data.jsonl)
- `--upload-endpoint`: Endpoint for uploading images (required)
- `--max-workers`: Maximum number of concurrent uploads (default: 5)
- `--resize`: Resize images to this maximum dimension while preserving aspect ratio (default: 1024)
- `--api-key`: API key for the upload service (if required)
- `--dry-run`: Process files without uploading (for testing)

## Image Hosting Options

For OpenAI GPT-4V fine-tuning, your images need to be accessible via direct URLs. There are several options for hosting your images:

1. **AWS S3**: Create a public bucket and use the AWS CLI or SDK to upload images
2. **Azure Blob Storage**: Similar to S3, but on Microsoft's cloud
3. **Cloudinary**: Specialized service for image hosting with a free tier
4. **ImgBB**: Simple image hosting service with an API
5. **Custom server**: If you have your own server or hosting

The `prepare_training_data.py` script can be configured to work with any service that provides an HTTP upload endpoint.

## Step 2: Submit Fine-Tuning Job

After preparing your training data, submit a fine-tuning job to OpenAI:

```bash
python scripts/submit_fine_tuning.py \
  --training-file ./training_data.jsonl \
  --model gpt-4o \
  --suffix surgical-instruments \
  --api-key your-openai-api-key
```

### Command Line Options

- `--training-file`: Path to JSONL file containing training data (required)
- `--validation-file`: Path to JSONL file containing validation data (optional)
- `--model`: Base model to fine-tune (default: gpt-4o)
- `--suffix`: Suffix to add to the fine-tuned model name (default: surgical-instruments)
- `--api-key`: OpenAI API key (if not set in OPENAI_API_KEY environment variable)
- `--n-epochs`: Number of epochs to train for (default: 3)
- `--batch-size`: Batch size for training (default: 1)
- `--learning-rate-multiplier`: Learning rate multiplier (default: 1.0)

## JSONL Format for GPT-4V Fine-Tuning

The training data JSONL file must follow this format for each example:

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are a surgical instrument identification assistant. When shown an image of a surgical instrument, identify the type of instrument accurately."
    },
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "What is this surgical instrument?"
        },
        {
          "type": "image_url",
          "image_url": {
            "url": "https://example.com/images/surgical-instrument-1.jpg"
          }
        }
      ]
    },
    {
      "role": "assistant",
      "content": "This is an Adson brown forceps. It's used in surgical procedures for grasping, clamping, or manipulating tissues."
    }
  ]
}
```

## Splitting Data for Training and Validation

It's recommended to create separate training and validation sets. You can do this manually by creating two separate JSONL files or by modifying the `prepare_training_data.py` script to automatically split the data.

## Monitoring Fine-Tuning Progress

The `submit_fine_tuning.py` script will monitor the progress of your fine-tuning job. You can also check the status using the OpenAI Dashboard or API.

## Using Your Fine-Tuned Model

Once the fine-tuning is complete, you can use your model with the OpenAI API:

```python
from openai import OpenAI

client = OpenAI(api_key="your-api-key")

response = client.chat.completions.create(
    model="ft:gpt-4o:your-org:surgical-instruments:1.0.0",
    messages=[
        {"role": "system", "content": "You are a surgical instrument identification assistant."},
        {"role": "user", "content": [
            {"type": "text", "text": "What is this surgical instrument?"},
            {"type": "image_url", "image_url": {"url": "https://example.com/your-image.jpg"}}
        ]}
    ]
)

print(response.choices[0].message.content)
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
