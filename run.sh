#!/bin/bash

# Debug info: Print the current working directory
echo "Current working directory: $(pwd)"

# Debug info: List the contents of the current directory before copying
echo "Contents of the current directory before copying:"
ls -lah

# Debug info: Check if the destination directory is accessible
if [ ! -w "$(pwd)" ]; then
    echo "Warning: Current directory is not writable. Please check permissions."
else
    echo "Destination directory is writable."
fi

# Debug info: Ensure /staging/wshao33 is accessible
echo "Checking access to /staging/wshao33..."
if [ ! -d /staging/wshao33 ]; then
    echo "Warning: /staging/wshao33 directory does not exist or is not accessible."
else
    if [ ! -r /staging/wshao33 ]; then
        echo "Warning: /staging/wshao33 directory is not readable. Please check permissions."
    else
        echo "/staging/wshao33 is accessible."
        # Debug info: List the contents of the /staging/wshao33 directory
        echo "Contents of /staging/wshao33:"
        ls -lah /staging/wshao33
    fi
fi

# Copy dataset and model files from the mounted directory
#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "Copying dataset and model files from mounted directory..."

if ! cp -v /staging/wshao33/comb_comments.parquet ./; then
    echo "Error: Failed to copy comb_comments.parquet."
    exit 1
fi

if ! cp -v /staging/wshao33/llama_model.tar.gz ./; then
    echo "Error: Failed to copy llama_model.tar.gz."
    exit 1
fi

echo "All files copied successfully!"


# Debug info: List the contents of the current directory after copying
echo "Contents of the current directory after copying:"
ls -lah

# Check if the files were copied
if [ ! -f comb_comments.parquet ]; then
    echo "Warning: comb_comments.parquet is missing after the copy operation."
fi
if [ ! -f llama_model.tar.gz ]; then
    echo "Warning: llama_model.tar.gz is missing after the copy operation."
else
    # Extract the model
    echo "Extracting model..."
    if ! tar -xzf llama_model.tar.gz; then
        echo "Error: Failed to extract llama_model.tar.gz."
    else
        rm llama_model.tar.gz
    fi
fi

# Debug info: List the contents of the current directory after extraction
echo "Contents of the current directory after extracting the model:"
ls -lah

# Run the Python script
echo "Running the Python script..."
if ! python reddit_llama.py; then
    echo "Error: Python script execution failed."
fi


# Define variables
SOURCE_DIR="./best_llama_model"
DEST_DIR="/staging/wshao33"
TAR_FILE="best_llama_model.tar.gz"

# Check if the source directory exists
if [ ! -d "$SOURCE_DIR" ]; then
  echo "Error: Source directory $SOURCE_DIR does not exist."
  exit 1
fi

# Ensure the destination directory exists
if [ ! -d "$DEST_DIR" ]; then
  echo "Destination directory $DEST_DIR does not exist. Creating it..."
  mkdir -p "$DEST_DIR"
fi

# Create a tar.gz archive of the source directory
echo "Creating tar archive..."
tar -czf "$DEST_DIR/$TAR_FILE" -C "$(dirname "$SOURCE_DIR")" "$(basename "$SOURCE_DIR")"

# Check if the tar command was successful
if [ $? -eq 0 ]; then
  echo "Archive created successfully: $DEST_DIR/$TAR_FILE"
else
  echo "Error: Failed to create the archive."
  exit 1
fi




echo "Job completed. Check the logs for any errors or warnings."
