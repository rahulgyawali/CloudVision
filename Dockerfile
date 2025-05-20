FROM public.ecr.aws/lambda/python:3.8

RUN yum update -y && yum install -y cmake gcc gcc-c++ mesa-libGL && yum clean all

COPY requirements.txt ${LAMBDA_TASK_ROOT}/
RUN pip install -r ${LAMBDA_TASK_ROOT}/requirements.txt --no-cache-dir
RUN pip install torch==1.9.0 torchvision==0.10.0 --index-url https://download.pytorch.org/whl/cpu

RUN mkdir -p /tmp/.cache
ENV TORCH_HOME=/tmp/.cache/torch
ENV XDG_CACHE_HOME=/tmp/.cache/torch

COPY resnetV1_video_weights.pt ${LAMBDA_TASK_ROOT}/
COPY resnetV1.pt ${LAMBDA_TASK_ROOT}/
COPY handler_face_detection.py ${LAMBDA_TASK_ROOT}/
COPY handler_face_recognition.py ${LAMBDA_TASK_ROOT}/

CMD ["default-handler"]