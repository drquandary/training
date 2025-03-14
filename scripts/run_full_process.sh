#!/bin/bash
# Script to run the full process of preparing data and submitting a fine-tuning job

# Check for required environment variables
if [ -z "$OPENAI_API_KEY" ]; then
    echo "Error: OPENAI_API_KEY environment variable not set"
    exit 1
fi

if [ -z "$UPLOAD_ENDPOINT" ]; then
    echo "Error: UPLOAD_ENDPOINT environment variable not set"
    exit 1
fi

# Set default values
PHOTOS_DIR="./photos"
OUTPUT_FILE="./training_data.jsonl"
MODEL="gpt-4o"
SUFFIX="surgical-instruments"
MAX_WORKERS=5
RESIZE=1024
N_EPOCHS=3

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --photos-dir)
            PHOTOS_DIR="$2"
            shift 2
            ;;
        --output-file)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --suffix)
            SUFFIX="$2"
            shift 2
            ;;
        --max-workers)
            MAX_WORKERS="$2"
            shift 2
            ;;
        --resize)
            RESIZE="$2"
            shift 2
            ;;
        --n-epochs)
            N_EPOCHS="$2"
            shift 2
            ;;
        --upload-api-key)
            UPLOAD_API_KEY="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Display configuration
echo "Configuration:"
echo "  Photos directory: $PHOTOS_DIR"
echo "  Output file: $OUTPUT_FILE"
echo "  Upload endpoint: $UPLOAD_ENDPOINT"
echo "  Model: $MODEL"
echo "  Suffix: $SUFFIX"
echo "  Max workers: $MAX_WORKERS"
echo "  Resize: $RESIZE"
echo "  Epochs: $N_EPOCHS"
echo ""

# Confirm before proceeding
read -p "Proceed with the above configuration? (y/n) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 1
fi

# Step 1: Prepare training data
echo "Step 1: Preparing training data..."
UPLOAD_API_KEY_PARAM=""
if [ ! -z "$UPLOAD_API_KEY" ]; then
    UPLOAD_API_KEY_PARAM="--api-key $UPLOAD_API_KEY"
fi

python scripts/prepare_training_data.py \
    --photos-dir "$PHOTOS_DIR" \
    --output-file "$OUTPUT_FILE" \
    --upload-endpoint "$UPLOAD_ENDPOINT" \
    --max-workers "$MAX_WORKERS" \
    --resize "$RESIZE" \
    $UPLOAD_API_KEY_PARAM

if [ $? -ne 0 ]; then
    echo "Error preparing training data"
    exit 1
fi

echo "Training data prepared successfully: $OUTPUT_FILE"

# Confirm before submitting to OpenAI
read -p "Submit fine-tuning job to OpenAI? (y/n) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Fine-tuning job submission skipped."
    exit 0
fi

# Step 2: Submit fine-tuning job
echo "Step 2: Submitting fine-tuning job..."
python scripts/submit_fine_tuning.py \
    --training-file "$OUTPUT_FILE" \
    --model "$MODEL" \
    --suffix "$SUFFIX" \
    --n-epochs "$N_EPOCHS"

if [ $? -ne 0 ]; then
    echo "Error submitting fine-tuning job"
    exit 1
fi

echo "Process completed successfully!"
