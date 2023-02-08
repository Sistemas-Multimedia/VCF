FROM ubuntu:20.04

ENV VCF noninteractive

RUN apt update
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN apt install python3-pip -y

RUN pip install opencv-python numpy zlib-state scikit-image scikit-learn
RUN apt install git -y
RUN pip install "scalar_quantization@git+https://github.com/vicente-gonzalez-ruiz/scalar_quantization"
RUN pip install "color_transforms@git+https://github.com/vicente-gonzalez-ruiz/color_transforms"
RUN pip install "DWT2D@git+https://github.com/vicente-gonzalez-ruiz/DWT2D"