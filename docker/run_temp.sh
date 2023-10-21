#!/bin/bash

if [ -z ${USE_DEV_DOCKER+x} ]
then
    echo "USE_DEV_DOCKER is unset. you should execute set_env.sh"
    exit 1
fi

docker container run --gpus $GPU_OPTION -it --rm --name $CONTAINER_NAME \
  --net=host --ipc=host \
  -e SUPPORT_EGL=true \
  -v $SC2_PATH:/root/StarCraftII \
  $IMAGE
