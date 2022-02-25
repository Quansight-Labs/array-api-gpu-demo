#FROM rocm/dev-ubuntu-20.04:4.3
FROM cupy/cupy-rocm:v10.2.0

# Install base utilities
RUN sudo apt update && \
    sudo apt upgrade -y && \
    sudo apt install -y wget && \
    sudo apt install build-essential -y && \
    sudo apt install git -y && \
    sudo apt install hipfft && \
    sudo apt clean && \
    sudo rm -rf /var/lib/apt/lists/*

#RUN sudo mkdir -p /opt/conda

ENV ROCM_HOME=/opt/rocm-4.3.1
ENV LD_LIBRARY_PATH=$ROCM_HOME/lib
ENV CUPY_INSTALL_USE_HIP=1
ENV HCC_AMDGPU_TARGET=gfx906

# Install miniconda
ENV CONDA_DIR /opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
     sudo /bin/bash ~/miniconda.sh -b -p /opt/conda

# Put conda in path so we can use conda activate
ENV PATH=$CONDA_DIR/bin:$PATH

WORKDIR /amd-demo
