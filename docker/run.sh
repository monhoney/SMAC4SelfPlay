#!/bin/bash

if [ -z ${USE_DEV_DOCKER+x} ]
then
    echo "USE_DEV_DOCKER is unset. you should execute set_env.sh"
    exit 1
fi

docker container exec -it $CONTAINER_NAME bash
