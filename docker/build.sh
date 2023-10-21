#!/bin/bash

if [ -z ${USE_DEV_DOCKER+x} ]
then
    echo "USE_DEV_DOCKER is unset. you should execute set_env.sh"
    exit 1
fi

echo "copy id_rsa" 
cp $HOME/.ssh/id_rsa .

echo "copy .netrc"
cp $HOME/.netrc .

echo "build docker"
docker build $BUILD_OPTION -f Dockerfile -t $IMAGE .

echo "remove id_rsa"
rm id_rsa
rm .netrc
