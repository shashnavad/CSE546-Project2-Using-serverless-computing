__copyright__ = "Copyright 2024, VISA Lab"
__license__   = "MIT"

import os
os.environ['TORCH_HOME'] = '/tmp'
import cv2
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
import boto3

# Initialize the S3 client for AWS operations
region_name = 'us-east-1'

s3_client = boto3.client('s3', region_name=region_name)

# Define S3 bucket names
input_bucket = '1229407428-stage-1'
result_bucket = '1229407428-output'

# Initialize face detection and recognition models
face_detector = MTCNN(image_size=240, margin=0, min_face_size=20)  # MTCNN for face detection
face_recognizer = InceptionResnetV1(pretrained='vggface2').eval()  # InceptionResnetV1 for face embedding

def process_face_recognition(image_file_path):
    # Read the image file
    img = cv2.imread(image_file_path, cv2.IMREAD_COLOR)
    
    # Extract the file key (name without extension)
    file_key = os.path.splitext(os.path.basename(image_file_path))[0].split(".")[0]
    
    # Convert the image to RGB (PIL format)
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    # Detect face in the image
    face, prob = face_detector(img, return_prob=True, save_path=None)
    
    # Load pre-saved face embedding data
    saved_data = torch.load('./data.pt')
    
    if face is not None:
        # Generate embedding for the detected face
        emb = face_recognizer(face.unsqueeze(0)).detach()
        
        # Unpack saved embeddings and names
        embedding_list, name_list = saved_data
        
        # Calculate distances between the detected face and saved embeddings
        dist_list = [torch.dist(emb, emb_db).item() for emb_db in embedding_list]
        
        # Find the index of the minimum distance (best match)
        idx_min = dist_list.index(min(dist_list))
        
        # Save the recognized name to a file
        with open("/tmp/" + file_key + ".txt", 'w+') as f:
            f.write(name_list[idx_min])
        
        return name_list[idx_min]
    else:
        print(f"No face is detected")
    return None

def lambda_handler(event, context):
    # Extract bucket name and image file name from the event
    source_bucket = event['bucket_name']
    source_image = event['image_file_name']
    local_image_path = f"/tmp/{source_image}"
    
    # Download the image from S3 bucket
    s3_client.download_file(Bucket=source_bucket, Key=source_image, Filename=local_image_path)
    print(f"Downloaded {source_image} from bucket {source_bucket} to {local_image_path}")
    
    # Perform face recognition
    recognition_output = process_face_recognition(local_image_path)

    # If a face is recognized, upload the result to S3
    if recognition_output:
        result_file_name = os.path.splitext(source_image)[0] + ".txt"
        result_file_path = '/tmp/' + result_file_name
        s3_client.upload_file(result_file_path, result_bucket, result_file_name)

    # Return the result of the face recognition process
    return {
        'statusCode': 200,
        'body': f"Face recognition completed. Result: {recognition_output}"
    }

# For local testing
if __name__ == "__main__":
    test_event = {
        'bucket_name': input_bucket,
        'image_file_name': 'test_image.jpg'
    }
    print(lambda_handler(test_event, None))