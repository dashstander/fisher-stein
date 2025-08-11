#!/usr/bin/env python3
"""
Script to reorganize S3 files from the old naming convention to the new organized structure.
Moves files from gpt2-wikipedia-layer* to gpt2-wikipedia/layer-6/ with clean filenames.
"""

import boto3
import logging
from botocore.exceptions import ClientError

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Configuration
BUCKET_NAME = "fisher-stein-matrices-01"
OLD_PREFIX = "gpt2-wikipedia-layer"  # Files currently have names like "gpt2-wikipedia-layerfisher_seq_*.npz"
NEW_PREFIX = "gpt2-wikipedia/layer-6/"

def reorganize_s3_files():
    """Reorganize S3 files from old structure to new organized structure."""
    
    s3_client = boto3.client('s3')
    
    try:
        # List all objects in the bucket
        logging.info(f"Scanning bucket {BUCKET_NAME} for files to reorganize...")
        
        paginator = s3_client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=BUCKET_NAME, Prefix=OLD_PREFIX)
        
        files_to_move = []
        
        for page in pages:
            if 'Contents' not in page:
                continue
                
            for obj in page['Contents']:
                key = obj['Key']
                
                # Skip if it's already in the new format
                if key.startswith(NEW_PREFIX):
                    continue
                    
                # Check if this is a file we want to move
                if OLD_PREFIX in key and any(pattern in key for pattern in ['fisher_seq_', 'config.json', 'summary.json']):
                    files_to_move.append(key)
        
        logging.info(f"Found {len(files_to_move)} files to reorganize")
        
        # Move each file
        for old_key in files_to_move:
            # Extract the clean filename
            if 'fisher_seq_' in old_key:
                # Extract fisher_seq_XXXXXX.npz from gpt2-wikipedia-layerfisher_seq_XXXXXX.npz
                filename = old_key.split('fisher_seq_')[1]
                filename = 'fisher_seq_' + filename
            elif 'config.json' in old_key:
                filename = 'config.json'
            elif 'summary.json' in old_key:
                filename = 'summary.json'
            else:
                logging.warning(f"Unexpected file pattern: {old_key}, skipping")
                continue
            
            new_key = NEW_PREFIX + filename
            
            logging.info(f"Moving {old_key} -> {new_key}")
            
            try:
                # Copy to new location
                s3_client.copy_object(
                    Bucket=BUCKET_NAME,
                    CopySource={'Bucket': BUCKET_NAME, 'Key': old_key},
                    Key=new_key
                )
                
                # Delete old location
                s3_client.delete_object(Bucket=BUCKET_NAME, Key=old_key)
                
                logging.info(f"Successfully moved {filename}")
                
            except ClientError as e:
                logging.error(f"Failed to move {old_key}: {e}")
                
    except ClientError as e:
        logging.error(f"Error accessing S3: {e}")
        return False
    
    logging.info("File reorganization complete!")
    
    # Verify the new structure
    logging.info("Verifying new file structure...")
    try:
        response = s3_client.list_objects_v2(Bucket=BUCKET_NAME, Prefix=NEW_PREFIX)
        if 'Contents' in response:
            logging.info(f"Files now in {NEW_PREFIX}:")
            for obj in response['Contents']:
                logging.info(f"  {obj['Key']}")
        else:
            logging.warning(f"No files found in {NEW_PREFIX}")
    except ClientError as e:
        logging.error(f"Error verifying new structure: {e}")
    
    return True


if __name__ == "__main__":
    logging.info("Starting S3 file reorganization...")
    success = reorganize_s3_files()
    
    if success:
        logging.info("Reorganization completed successfully!")
    else:
        logging.error("Reorganization failed!")