#----------------------------------------------------------
# TODO
# - jupyter lab 기본 실행 스크립트 
#   - startup 파일
#----------------------------------------------------------

#----------------------------------------------------------
# From
#----------------------------------------------------------
ARG ARCH=
ARG CUDA=10.1
ARG OS_NAME=ubuntu
ARG UBUNTU_VERSION=18.04

# FROM nvidia/cuda${ARCH:+-$ARCH}:${CUDA}-base-ubuntu${UBUNTU_VERSION} as base 
FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04
# FROM ${OS_NAME}:${UBUNTU_VERSION}

ARG CUDNN=7.6.4.38-1
ARG CUDNN_MAJOR_VERSION=7
ARG LIB_DIR_PREFIX=x86_64

#----------------------------------------------------------
# Ubuntu Docker Image
#----------------------------------------------------------
MAINTAINER "elvin <vcthinkr@gmail.com>"

#----------------------------------------------------------
# Basics						  # --no-install-recommends
#----------------------------------------------------------
RUN cd /etc/apt \
	&& sed -i "s/archive.ubuntu.com/mirror.kakao.com/g" sources.list

RUN set -ex \
	&& apt-get update \
    	&& apt-get install -y \
	build-essential \
	ubuntu-drivers-common \ 
	lsb-release \
	sudo \
	fakeroot \
	software-properties-common \ 
	dh-make \
	devscripts \
	python3 \
	python3-pip \
	net-tools \
	bash \
	curl \
	wget \ 
	vim \
	neovim \
	git \ 
	tree \
	htop \
	ncdu \
	zip \
	unzip \
	apt-utils \
	virtualenv \
	openssh-client \
	openssh-server \
	libcublas10=10.2.1.243-1 \ 
    libcublas-dev=10.2.1.243-1 

#----------------------------------------------------------
# Setup Configs
#----------------------------------------------------------

#----------------------------------------------------------
# Dev: C++
#----------------------------------------------------------
RUN set -ex \
	&& apt-get install -y --no-install-recommends \
	cmake \
	coreutils \
	gcc \
	g++ \
	libtbb2 \ 
	libjpeg-dev \
	graphviz

#----------------------------------------------------------
# Dev: Data science
#----------------------------------------------------------
ADD ./configs/requirements.txt /tmp/requirements.txt
RUN set -ex && pip3 install --upgrade pip && pip install -r /tmp/requirements.txt
# RUN set -ex && jupyter labextension install jupyterlab_vim   ==> Node.js npm required

#----------------------------------------------------------
# Theme & Tools
#----------------------------------------------------------
RUN set -ex && apt-get install -y zsh 

#----------------------------------------------------------
# APT Clean
#----------------------------------------------------------
RUN set -ex \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

#----------------------------------------------------------
# Make User
#----------------------------------------------------------
ENV CONTAINER_USER elvin
ENV USER_UID 1000

RUN useradd -m -s /bin/bash -N -u ${USER_UID} ${CONTAINER_USER} \
	&& echo ${CONTAINER_USER}:${CONTAINER_USER} | chpasswd \
	&& cp /etc/sudoers /etc/sudoers.bak \
	&& echo "${CONTAINER_USER}  ALL=(root) ALL" >> /etc/sudoers

USER ${CONTAINER_USER}
ADD ./configs/init.vim /home/elvin/.configs/nvim/init.vim

#----------------------------------------------------------
# Setup container env 
#----------------------------------------------------------
EXPOSE 22
EXPOSE 8888
# VOLUME ["mount"]

WORKDIR /home/elvin
ENTRYPOINT ["/bin/zsh"]

#----------------------------------------------------------
# Setup shell config
#----------------------------------------------------------
#WORKDIR /tf
#VOLUME ["/tf"]
#COPY bashrc /etc/bash.bashrc
#RUN chmod a+rwx /etc/bash.bashrc

#----------------------------------------------------------
# Run
#----------------------------------------------------------
# CMD jupyter notebook --port=8888 --ip=0.0.0.0

