import os
import json
import boto3
import base64
import numpy as np
import asyncio
from facenet_pytorch import MTCNN
from PIL import Image
import logging

sqs = boto3.client('sqs')
REQ_Q_URL = 'https://sqs.us-east-1.amazonaws.com/038462753394/1229604729-req-queue'
SUCCESS_MSG = "Face Detection Successful"
FAILURE_MSG = "Error in Face Detection"
logger = logging.getLogger()
logger.setLevel(logging.INFO)    

#face detection class
class face_detection:
    def __init__(self):
        self.mtcnn = MTCNN(image_size=240, margin=0, min_face_size=20)

    async def face_detection_func(self, test_image_path, output_path):
        img = Image.open(test_image_path).convert("RGB")
        img = np.array(img)
        img = Image.fromarray(img)
        key = os.path.splitext(os.path.basename(test_image_path))[0].split(".")[0]
        face, prob = self.mtcnn(img, return_prob=True, save_path=None)
        if face is not None:
            os.makedirs(output_path, exist_ok=True)
            face_img = face - face.min()
            face_img = face_img / face_img.max()
            face_img = (face_img * 255).byte().permute(1, 2, 0).numpy()
            face_pil = Image.fromarray(face_img, mode="RGB")
            face_img_path = os.path.join(output_path, f"{key}_face.jpg")
            face_pil.save(face_img_path)
            return face_img_path
        else:
            logger.info("face not detected")
            return None

#Face detection model object
model = face_detection()

#Sending message to request queue
def send_message_to_queue(request_id,filename,face_image_path):
    with open(face_image_path, 'rb') as f:
        face_data = f.read()
    image_base64 = base64.b64encode(face_data).decode('utf-8')
    json_body = get_json_body(request_id,filename,image_base64)
    message_body = json.dumps(json_body)
    logger.info(f"Reqeust body for Reqeuest queue: {message_body}")
    response = sqs.send_message(QueueUrl=REQ_Q_URL,MessageBody=message_body)
    logger.info(f"Response from insertion to request queue: {response}")

#Preparing body for sending to request queue
def get_json_body(request_id,filename,image_base64):
    json_body = {'request_id': request_id,'face_image': image_base64,'filename': filename}
    response_body = json.dumps(json_body)
    return response_body

#Generating Response 
def get_handler_response(message,request_id):
    message_body = json.dumps({'message':message,'request_id':request_id})
    response = {'statusCode': 200,'body': message_body}
    return response

#Processing input
def process_input(event):
    body = json.loads(event['body'])
    content = body['content']
    request_id = body['request_id']
    filename = os.path.basename(body['filename'])
    logger.info(f"Request received for Reqesut id: {request_id} filename: {filename}")
    return filename,content,request_id

#Face detector handler
async def face_detector(event):
    #Parsing input from POST request
    filename,content,request_id = process_input(event)
    #Preparing input and output image path
    input_image_path = f"/tmp/{filename}"
    with open(input_image_path, 'wb') as f:
        f.write(base64.b64decode(content))
    output_image_path = "/tmp/output"
    os.makedirs(output_image_path, exist_ok=True)
    #Calling face detection model
    result_image_path = await model.face_detection_func(input_image_path, output_image_path)
    logger.info(f"Output result from model is {result_image_path}")
    if result_image_path:
        #Sending message to request queue with result
        send_message_to_queue(request_id,filename,result_image_path)
        response = get_handler_response(SUCCESS_MSG,request_id)
    else:
        response = get_handler_response(FAILURE_MSG,request_id)
    logger.info(f"Final response for {request_id} {response}")
    return response

#Default handler
def handler(event, context):
    try:
        logger.info("Reqeust triggerd")
        response = asyncio.run(face_detector(event))
        logger.info(f"Request processed {response}")
        return response
    except Exception as e:
        response = get_handler_response(f"{FAILURE_MSG} {str(e)}","")
        logger.info(f"Request failed {response}")
        return response
