# README
# https://github.com/NVIDIA/nvidia-docker
# https://forum.manjaro.org/t/howto-installing-docker-and-nvidia-runtime-my-experience-and-howto/97017
# https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#install-guide
# https://github.com/Lightning-AI/lightning/tree/master/dockers

ARG PYTHON_VERSION=3.9
ARG PYTORCH_VERSION=1.12
ARG CUDA_VERSION=11.4

FROM pytorchlightning/pytorch_lightning:base-cuda-py${PYTHON_VERSION}-torch${PYTORCH_VERSION}-cuda${CUDA_VERSION}

LABEL maintainer="Lightning-AI <https://github.com/Lightning-AI>"

ARG LIGHTNING_VERSION=""

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

# The code to run when container is started:
ADD arff2pandas arff2pandas
ADD configs configs
ADD data data
ADD dataloaders dataloaders
ADD experiments experiments
ADD models models
ADD niapy_extension niapy_extension
ADD storage storage

COPY evaluate.py /app
COPY rnn_vae_run.py /app

RUN python -c "import torch ; print(torch.__version__)" >> torch_version.info
CMD [ "python" , "rnn_vae_run.py"]
