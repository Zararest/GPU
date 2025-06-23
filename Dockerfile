FROM nvidia/cuda:12.3.2-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

RUN set -x
RUN apt-get update
RUN apt-get -y install build-essential cmake