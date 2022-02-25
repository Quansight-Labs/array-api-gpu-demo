FROM cupy/cupy-rocm:v10.2.0
USER root

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

RUN mkdir /amd-demo/packages/ && cd /amd-demo/packages && \
    git clone https://github.com/aktech/scipy.git --branch amd-demo --recursive && \
    git clone https://github.com/aktech/scikit-learn.git --branch amd-demo  --recursive && \
    git clone https://github.com/aktech/scikit-image.git --branch amd-demo  --recursive && \
    git clone https://github.com/aktech/cupy.git --branch amd-demo --recursive

ENV CONDA_ENV_NAME=docker-amd
ENV CUPY_NUM_BUILD_JOBS=55

COPY environment.yml /amd-demo/environment.yml

RUN conda info && \
    conda install mamba -n base -c conda-forge && \
    mamba env create -f /amd-demo/environment.yml && \
    mamba install -c conda-forge sysroot_linux-64=2.17 --yes && \
    conda clean --all && \
    rm -rf /opt/conda/envs/$CONDA_ENV_NAME/pkgs/ && \
    rm -rf /opt/conda/pkgs

SHELL ["conda", "run", "-n", "docker-amd", "/bin/bash", "-c"]
RUN conda info
RUN conda init bash

RUN cd /amd-demo/packages/cupy && python setup.py develop && \
    python -m pip install scipy && \
    cd /amd-demo/packages/scikit-learn && python setup.py develop --no-deps && \
    python -m pip uninstall scipy -y && \
    cd /amd-demo/packages/scipy && python dev.py --build-only && \
    conda clean --all && \
    rm -rf /opt/conda/envs/$CONDA_ENV_NAME/pkgs/ && \
    rm -rf /opt/conda/pkgs

ENV PYTHONPATH=$PYTHONPATH:/amd-demo/packages/scipy/installdir/lib/python3.8/site-packages
RUN cd /amd-demo/packages/scikit-image && python setup.py develop --no-deps && \
    echo "conda activate docker-amd" >> ~/.bashrc
