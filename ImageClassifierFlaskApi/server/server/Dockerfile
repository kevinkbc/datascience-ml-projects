FROM nvcr.io/nvidia/pytorch:21.05-py3

ENV DEBIAN_FRONTEND=noninteractive

ADD https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth /home/app/server/weights/

COPY . /home/app/server
WORKDIR /home/app/server

CMD python server.py