FROM nvidia/cuda:11.1-runtime-ubuntu20.04
#FROM python:3.8

LABEL author="Gus Hahn-Powell"
LABEL description="Image definition for Arabic NLP"


# Update
RUN apt-get -y update
RUN apt-get -y upgrade
RUN apt-get install -y software-properties-common

# install python 3.8
RUN add-apt-repository -y ppa:deadsnakes/ppa
RUN apt-get -y update
RUN apt-get install -y python3.8
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1
RUN python --version
RUN apt-get install -y python3-pip
RUN update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1
#RUN python -m pip install --upgrade pip setuptools wheel
RUN pip install --upgrade pip setuptools wheel

# Create app directory
WORKDIR /app

# Jupyter deps
RUN pip install -U "ipython>=7.19.0,<8"
RUN pip install -U jupyter==1.0.0
RUN pip install -U jupyter-contrib-nbextensions==0.5.1
RUN jupyter contrib nbextension install --user
# Commonly used test utils
RUN pip install -U pytest==5.3.4

RUN pip install torch==1.7.0+cu110 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install torchvision==0.8.1+cu110 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install torchaudio===0.7.0

# Project-specific deps
# Bundle app source
COPY . .

RUN chmod u+x  scripts/*
RUN mv scripts/* /usr/local/bin/
# RUN chmod u+x /usr/local/bin/test-all
# RUN chmod u+x /usr/local/bin/launch-notebook
RUN rmdir scripts

RUN pip install -e ".[all]"

# Launch jupyter
CMD ["/bin/bash", "/usr/local/bin/launch-notebook"]
