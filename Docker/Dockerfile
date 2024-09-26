FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

# Env vars for the nvidia-container-runtime.
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES graphics,utility,compute

RUN apt-get update
RUN apt-get install -y unzip sudo git wget
RUN apt-get install -y ffmpeg libsm6 libxext6

RUN apt-get update
RUN apt-get install -y wget && rm -rf /var/lib/apt/lists/*

ARG USERNAME=yuhengq
RUN useradd -ms /bin/bash ${USERNAME} && \
    echo "${USERNAME}:password" | chpasswd

RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && bash Miniconda3-latest-Linux-x86_64.sh -b -p /home/${USERNAME}/.conda \
    && rm -f Miniconda3-latest-Linux-x86_64.sh 

USER root
RUN chown -R ${USERNAME}:${USERNAME} /home/${USERNAME}/.conda

USER ${USERNAME}
ENV PATH="/home/${USERNAME}/.conda/bin:${PATH}"
WORKDIR /home/${USERNAME}
RUN conda init && . ~/.bashrc

# Setup environment
SHELL ["/bin/bash", "-l", "-c"]