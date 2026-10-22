# Dockerfile — Automated Reproduction Container for ConfCC
# ConfCC: Conformal Congestion Control (Distribution-Free Delay Bounds)
# Venues: IEEE/ACM Transactions on Networking (ToN) / IEEE INFOCOM

FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    python3 \
    python3-pip \
    python3-dev \
    texlive-latex-base \
    texlive-latex-extra \
    texlive-fonts-recommended \
    texlive-science \
    texlive-bibtex-extra \
    biber \
    ghostscript \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir \
    numpy \
    pandas \
    matplotlib \
    seaborn \
    scipy \
    simpy

WORKDIR /confcc
COPY . /confcc

RUN chmod +x /confcc/reproduce_all.sh

CMD ["/bin/bash", "-c", "./reproduce_all.sh"]
