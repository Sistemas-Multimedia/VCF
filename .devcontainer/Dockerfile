FROM ubuntu:20.04

RUN apt update
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN apt install python3-pip -y

RUN pip install opencv-python numpy zlib-state scikit-image scikit-learn