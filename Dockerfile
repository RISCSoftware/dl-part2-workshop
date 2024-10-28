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
RUN python3 -m pip install \
    # Required for this workshop
    "jupyter==1.0.0" \
    "notebook==7.2.2" \
    "rise==5.7.1" \
    "jupyterlab_rise==0.42.0" \
    "d2l==1.0.3" \
    # Not required for this workshop, but for the preceding workshop
    "ipympl==0.9.3" \
    "opencv-python==4.9.0.80" \
    # Developer tools (not required for this workshop)
    "black"

# Set timezone
ARG TZ=Europe/Vienna
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Setup directory structure
RUN mkdir /repo
WORKDIR /repo
COPY . /repo

# Add selected files from the first workshop
RUN git clone https://github.com/risc-mi/dl-cv-workshop.git /tmp/dl-cv-workshop && \
    mkdir -p /repo/workshop_2 && \
    cp /tmp/dl-cv-workshop/README.md /repo/workshop_2 && \
    cp /tmp/dl-cv-workshop/CMIYC.ipynb /repo/workshop_2 && \
    cp /tmp/dl-cv-workshop/MNIST-complete.ipynb /repo/workshop_2 && \
    cp /tmp/dl-cv-workshop/MNIST.ipynb /repo/workshop_2 && \
    cp /tmp/dl-cv-workshop/Introduction.pdf /repo/workshop_2 && \
    cp -r /tmp/dl-cv-workshop/pics /repo/workshop_2 && \
    cp -r /tmp/dl-cv-workshop/mnist /repo/workshop_2 && \
    cp -r /tmp/dl-cv-workshop/cmiyc /repo/workshop_2 && \
    rm -rf /tmp/dl-cv-workshop

# ENTRYPOINT ["tail", "-f", "/dev/null"]  # does not work with docker run -d
CMD ["tail", "-f", "/dev/null"]
