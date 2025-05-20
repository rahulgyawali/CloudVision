import os
import json
import boto3
import base64
import torch
import numpy as np
from PIL import Image
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger()
logger.setLevel(logging.INFO)
sqs = boto3.client('sqs')
RESP_Q_URL = 'https://sqs.us-east-1.amazonaws.com/038462753394/1229604729-resp-queue'
SUCCESS_MSG = "Face Recognition Successful"
FAILURE_MSG = "Error in Face Recognition"
model_path = os.path.join(os.environ.get('LAMBDA_TASK_ROOT', ''), 'resnetV1.pt')
model_wt_path = os.path.join(os.environ.get('LAMBDA_TASK_ROOT', ''), 'resnetV1_video_weights.pt')
saved_data = torch.load(model_wt_path)
resnet = torch.jit.load(model_path)      

#Face recognition class
class face_recognition:
     async def face_recognition_func(self, face_img_path):
        face_pil = Image.open(face_img_path).convert("RGB")
        key = os.path.splitext(os.path.basename(face_img_path))[0].split(".")[0]
        face_numpy = np.array(face_pil, dtype=np.float32) / 255.0
        face_numpy = np.transpose(face_numpy, (2, 0, 1))
        face_tensor = torch.tensor(face_numpy, dtype=torch.float32)
        if face_tensor is not None:
            emb = resnet(face_tensor.unsqueeze(0)).detach()
            embedding_list = saved_data[0]
            name_list = saved_data[1]
            dist_list = [torch.dist(emb, emb_db).item() for emb_db in embedding_list]
            min_index = dist_list.index(min(dist_list))
            name = name_list[min_index]
            return name
        else:
            logger.info("face not recognized")
            return None

#Face recognition model object
model = face_recognition()

#Preparing body for response queue
def get_json_body(request_id,name):
    json_body = {'request_id': request_id,'result': name if name else "Unknown"}
    response_body = json.dumps(json_body)
    return response_body

#Preparing response handler
def get_handler_response(message):
    message_body = json.dumps({'message':message})
    response = {'statusCode': 200,'body': message_body}
    return response

#Sending message to resposne queuue
def send_message_to_queue(request_id,recognized_name):
    json_body = get_json_body(request_id,recognized_name)
    logger.info(f"Sending messge to resposne queue {json_body}")
    response = sqs.send_message(QueueUrl=RESP_Q_URL,MessageBody=json_body)
    logger.info(f"Response from insertion to response queue: {response}")

#Perform prediction for image
async def prediction(face_base64):
    input_image_path = '/tmp/input.jpg'
    with open(input_image_path, 'wb') as f:
        f.write(base64.b64decode(face_base64))
    #Calling face recogintion function
    output_image_name = await model.face_recognition_func(input_image_path)
    logger.info(f"Output result from model is {output_image_name}")
    return output_image_name

#Parse input record 
def process_input(record):
    message_body_str = json.loads(record['body'])
    logger.info("Message body "+message_body_str)
    message_body = json.loads(message_body_str) 
    request_id = message_body['request_id']
    face_image = message_body['face_image']
    filename = message_body['filename']
    logger.info(f"Request received for Reqesut id: {request_id} filename: {filename}")
    return request_id,face_image

#Face recognizer handler
async def face_recognizer(event):
    for record in event['Records']:
        #Process Input record
        request_id,face_image = process_input(record)
        #Perform Prediction
        name = await prediction(face_image)
        #Send result to response queue
        send_message_to_queue(request_id,name)
    response = get_handler_response(SUCCESS_MSG)
    return response

#Default handler
def handler(event, context):
    try:
        logger.info("Request triggerd")
        response = asyncio.run(face_recognizer(event))
        logger.info(f"Request processed {response}")
        return response
    except Exception as e:
        response = get_handler_response(f"{FAILURE_MSG} {str(e)}")
        logger.info(f"Request failed {response}")
        return response
