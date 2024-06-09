import json
import boto3
import base64
import os
import io
from PIL import Image
from pinecone import Pinecone

MAX_IMAGE_HEIGHT: int = 2048
MAX_IMAGE_WIDTH: int = 2048
pc = Pinecone(api_key="d4c340a9-8327-49c6-892c-7024b3b44024")
index = pc.Index("gallerystore")
destination_bucket = 'classifiedbucketusingllm'

bedrock_runtime=boto3.client('bedrock-runtime',region_name='us-east-1')
s3=boto3.client('s3')

def image_embedding(bytes_data):
  input_image = base64.b64encode(bytes_data).decode('utf8')
  body = json.dumps(
    {
        "inputImage": input_image
    }
  )
  response = bedrock_runtime.invoke_model(
  body=body,
  modelId="amazon.titan-embed-image-v1",
  accept="application/json",
  contentType="application/json"
  )
  response_body = json.loads(response.get("body").read())
  return response_body.get("embedding")

def lambda_handler(event, context):
    # Print the received event for debugging
    print(f"Received event: {json.dumps(event)}")
    
    # Get the bucket and object key from the S3 event
    bucket = event['Records'][0]['s3']['bucket']['name']
    key = event['Records'][0]['s3']['object']['key']
    
    try:
        # Download the image from S3
        response = s3.get_object(Bucket=bucket, Key=key)
        image_data = response['Body'].read()
        
        # Open the image
        image = Image.open(io.BytesIO(image_data))
        
        # Resize the image to 2048x2048
        resized_image = image.resize((2048, 2048))
        
        embedded_image = image_embedding(resized_image)
        
        # Classify the resized image using the mock function
        image_class = index.query(  namespace="ns1",
                                    vector=embedded_image,
                                    include_metadata=True,
                                    top_k=1
                                )['matches'][0]['metadata']['category']

        print(f"Image classified as: {image_class}")
        
        # Construct new key for the destination
        new_key = f"{image_class}/{os.path.basename(key)}"
        
        # Copy the file to the new location
        s3.copy_object(Bucket=destination_bucket, CopySource={'Bucket': bucket, 'Key': key}, Key=new_key)
        
        # Delete the original file from the source location
        s3.delete_object(Bucket=bucket, Key=key)
        
        print(f"File moved to {new_key} and deleted from {key}")
        
        return {
            'statusCode': 200,
            'body': json.dumps(f"File moved to {new_key} and deleted from {key}")
        }
        
    except Exception as e:
        print(e)
        print(f"Error processing object {key} from bucket {bucket}.")
        return {
            'statusCode': 500,
            'body': json.dumps(f"Error processing object {key} from bucket {bucket}.")
        }
