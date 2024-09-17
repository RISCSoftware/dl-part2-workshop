FROM huggingface/transformers-pytorch-gpu:latest
# FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update --allow-insecure-repositories && apt-get install -yq \
    htop \
    cifs-utils \
    build-essential \
    # provides nohup command
    coreutils \
    vim \
    nano \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    curl \
    rsync \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libxml2-dev \
    libxmlsec1-dev \
    libffi-dev \
    liblzma-dev \
    git \
    python3-distutils \
    # provides pstree
    psmisc \
    && rm -rf /var/lib/apt/lists/*

# Install additional python packages
RUN python3 -m pip install -U jupyter

# Set timezone
ARG TZ=Europe/Vienna
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Setup directory structure
RUN mkdir /repo
ADD . /repo
# WORKDIR /repo

# ENTRYPOINT ["tail", "-f", "/dev/null"]  # does not work with docker run -d
CMD ["tail", "-f", "/dev/null"]
