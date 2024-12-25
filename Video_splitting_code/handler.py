import boto3
import os
import subprocess
import json
import urllib.parse
from botocore.exceptions import NoCredentialsError, ClientError

# Initialize AWS clients for S3 and Lambda services

region_name = 'us-east-1'

s3_client = boto3.client('s3', region_name=region_name)
lambda_client = boto3.client('lambda', region_name=region_name)

# Define constants for bucket names and temporary directory
INPUT_BUCKET = '1229407428-input'
STAGE_1_BUCKET = '1229407428-stage-1'
TEMP_DIR = '/tmp'

def handler(event, context):
    """
    Main handler function for the Lambda.
    Processes the incoming S3 event, extracts a frame from the video,
    and triggers face recognition.
    """
    try:
        # Extract bucket and key information from the S3 event
        record = event['Records'][0]['s3']
        bucket = record['bucket']['name']
        key = urllib.parse.unquote_plus(record['object']['key'], encoding='utf-8')
        
        # Process the video and upload the extracted frame
        video_path = process_video(bucket, key)
        output_image = upload_processed_image(video_path, key)
        
        # Trigger the face recognition Lambda function
        invoke_face_recognition(output_image)
        
        return {
            'statusCode': 200,
            'body': json.dumps('Successfully processed video.')
        }
    except Exception as e:
        print(f"Error processing video: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps(f'Error processing video: {str(e)}')
        }

def process_video(bucket, key):
    """
    Downloads the video from S3 to the Lambda's temporary storage.
    """
    video_path = os.path.join(TEMP_DIR, key)
    download_from_s3(bucket, key, video_path)
    return video_path

def download_from_s3(bucket, key, download_path):
    """
    Downloads a file from S3 to the specified path.
    """
    try:
        s3_client.download_file(bucket, key, download_path)
    except ClientError as e:
        print(f"Error downloading file from S3: {str(e)}")
        raise

def upload_processed_image(video_path, original_key):
    """
    Extracts a frame from the video and uploads it to S3.
    """
    output_image = extract_frame(video_path)
    upload_to_s3(output_image, STAGE_1_BUCKET)
    return output_image

def extract_frame(video_path):
    """
    Uses ffmpeg to extract the first frame of the video as an image.
    """
    filename = os.path.basename(video_path)
    output_image = os.path.splitext(filename)[0] + ".jpg"
    output_path = os.path.join(TEMP_DIR, output_image)
    
    split_cmd = f'ffmpeg -i {video_path} -vframes 1 {output_path}'
    run_command(split_cmd)
    
    return output_image

def run_command(command):
    """
    Executes a shell command and handles potential errors.
    """
    try:
        subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        print(f"Command failed: {e.cmd}")
        print(f"Return code: {e.returncode}")
        print(f"Output: {e.output}")
        raise

def upload_to_s3(file_name, bucket, object_name=None):
    """
    Uploads a file to the specified S3 bucket.
    """
    if object_name is None:
        object_name = file_name
    
    file_path = os.path.join(TEMP_DIR, file_name)
    try:
        s3_client.upload_file(file_path, bucket, object_name)
    except NoCredentialsError:
        print("AWS credentials not available")
        raise
    except ClientError as e:
        print(f"Error uploading file to S3: {str(e)}")
        raise

def invoke_face_recognition(image_file_name):
    """
    Invokes the face recognition Lambda function asynchronously.
    """
    payload = {
        'bucket_name': STAGE_1_BUCKET,
        'image_file_name': image_file_name
    }
    try:
        lambda_client.invoke(
            FunctionName='face-recognition',
            InvocationType='Event',
            Payload=json.dumps(payload)
        )
    except ClientError as e:
        print(f"Error invoking face recognition Lambda: {str(e)}")
        raise

if __name__ == "__main__":
    # For local testing
    test_event = {
        "Records": [{
            "s3": {
                "bucket": {"name": INPUT_BUCKET},
                "object": {"key": "test_video.mp4"}
            }
        }]
    }
    print(handler(test_event, None))