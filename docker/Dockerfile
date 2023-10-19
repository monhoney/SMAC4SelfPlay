FROM nvcr.io/nvidia/pytorch:21.06-py3
MAINTAINER Moonhoen Lee <techmoong@gmail.com>
RUN apt-get update
RUN apt-get upgrade -y
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get install -y git-core build-essential zip curl unzip bc wget
##RUN apt-get install -y freeglut3-dev libfreetype6-dev python-dev libsdl2-image-dev libsdl2-ttf-dev libsdl2-gfx-dev libsdl2-dev libsdl2-mixer-dev libsdl2-net-dev libsmpeg-dev python-numpy subversion libportmidi-dev ffmpeg libswscale-dev libavformat-dev libavcodec-dev
##RUN apt-get install -y python-box2d python3-pygame
#RUN apt-get install -y python-box2d
#WORKDIR /install
#RUN wget https://github.com/deepmind/mujoco/releases/download/2.3.4/mujoco-2.3.4-linux-x86_64.tar.gz
#RUN tar zxvf mujoco-2.3.4-linux-x86_64.tar.gz
#RUN apt-get install -y swig
#RUN apt-get update
#RUN apt-get upgrade -y
#RUN apt-get install -y libosmesa6-dev libgl1-mesa-glx libglfw3
#RUN apt-get install -y libgl1-mesa-dri libegl1-mesa libgbm1
#RUN apt-get install -y libegl-dev

WORKDIR /root/.ssh
ADD id_rsa /root/.ssh/id_rsa
RUN chmod 600 /root/.ssh/id_rsa
RUN touch /root/.ssh/known_hosts
RUN ssh-keyscan github.com >> /root/.ssh/known_hosts

RUN apt update --fix-missing
RUN apt upgrade -y

RUN apt install -y locales
RUN apt install -y language-pack-ko
RUN apt install vim -y

WORKDIR /work
RUN git clone --recursive git@github.com:monhoney/SMAC2Study.git
WORKDIR /work/SMAC2Study
RUN pip install -r requirements.txt
RUN pip install typing_extensions==4.4.0
WORKDIR /work/SMAC2Study/3rd_library/smac
RUN pip install -e .

RUN echo "export BASE_DIR=/work/SMAC2Study" >> /root/.bashrc

RUN echo "copy netrc"
ADD .netrc /root/.netrc

ENV BASE_DIR=/work/SMAC2Study

ARG DISABLE_CACHE
WORKDIR /work/SMAC2Study
RUN git pull
RUN rm /root/.ssh/id_rsa

ENTRYPOINT ["/bin/bash"]
