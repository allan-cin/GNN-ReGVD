FROM nvidia/cuda:11.6.2-base-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update -qqy && \
  apt-get install -y \
  build-essential \
  cmake \
  curl \
  g++ \
  git \
  libsm6 \
  libxext6 \
  ffmpeg \
  libxrender1 \
  locales \
  pkg-config \
  poppler-utils \
  python3.8 python3.8-dev python3.8-distutils \
  python3-pip \
  software-properties-common \
  unzip \
  wget \
  && \
  update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1 \
  && \
  apt-get clean && \
  apt-get autoremove && \
  rm -rf /var/lib/apt/lists/*

# Set default python version
RUN rm -f /usr/bin/python && ln -s /usr/bin/python3 /usr/bin/python
RUN rm -f /usr/bin/pip && ln -s /usr/bin/pip3 /usr/bin/pip

COPY . /GNN-ReGVD
WORKDIR /GNN-ReGVD

RUN pip install torch==2.0.0+cu117 torchvision==0.15.1+cu117 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu117
RUN pip install -r req.txt
RUN pip install -U requests

ENTRYPOINT ["tail"]
CMD ["-f","/dev/null"]
