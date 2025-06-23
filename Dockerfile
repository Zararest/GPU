FROM nvidia/cuda:11.6.1-devel-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

RUN set -x
RUN apt-get update
RUN apt-get -y install build-essential cmake