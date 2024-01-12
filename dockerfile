FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-devel
COPY requirements.txt .
RUN pip install -r requirements.txt
RUN mkdir hal
VOLUME [ "hal" ]

# it should be run: docker run --gpus all -it $id -v .:hal
