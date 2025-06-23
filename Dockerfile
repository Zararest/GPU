FROM nvidia/cuda:12.9.0-devel-ubuntu24.04

RUN set -x
RUN apt-get update
RUN apt-get -y install \
  vim less man jq htop curl wget git sudo ca-certificates \
  python3 unzip openssh-client neofetch pip build-essential pandoc cmake