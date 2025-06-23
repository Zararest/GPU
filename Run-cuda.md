# Cuda container
## Install deps
Cuda 12.9 is not working for gtx 1660ti max-Q.
Cuda 11.6.1 is used instead.

Nvidia divers are required in order to support cuda 12.9:
```
sudo apt install nvidia-driver-575 nvidia-persistenced \
     libnvidia-cfg1-575
```

Cuda 12.9 is valid on June 2025. Tags in docker hub might be deprecated in the future, so check which version is available now:
https://hub.docker.com/r/nvidia/cuda

## Install image
Install cuda dev image:
```bash
docker login
docker pull nvidia/cuda:11.6.1-devel-ubuntu20.04
sudo apt install -y nvidia-container-runtime
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Build container:
```bash
docker build --no-cache --tag cuda-toolkit .
sudo docker run -it -d -v ~/:/root --runtime=nvidia --gpus all --name PBQP-dev --entrypoint /bin/bash cuda-toolkit  # -d to run container in the background
```
Attach to container:
- with VSCode: `ctrl+P`: Attach to running container
- with CLI: `docker attach PBQP-dev`
