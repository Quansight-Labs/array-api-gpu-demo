FROM gpuci/miniconda-cuda:11.2-devel-ubuntu20.04
RUN apt-get update -y \
    && apt-get install -y \
    libxml2 \
    libxml2-dev && \
    rm -rf /var/lib/apt/lists/*

ENV GH_BRANCH=array-api-gpu-demo
WORKDIR /demo
RUN mkdir /demo/packages/ && cd /demo/packages && \
    git clone https://github.com/aktech/scipy.git --branch $GH_BRANCH --recursive && \
    git clone https://github.com/aktech/scikit-learn.git --branch $GH_BRANCH  --recursive && \
    git clone https://github.com/aktech/scikit-image.git --branch $GH_BRANCH  --recursive && \
    git clone https://github.com/aktech/cupy.git --branch $GH_BRANCH --recursive

COPY environment.yml /demo/environment.yml
COPY plot_coin_segmentation.ipynb /demo/plot_coin_segmentation.ipynb
COPY segmentation_performance.py /demo/segmentation_performance.py


ENV CONDA_ENV_NAME=demo
ENV CUDA_VERSION=11.2
ENV CUDA_PATH=/usr/local/cuda-$CUDA_VERSION
ENV CUDA_HOME=$CUDA_PATH
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_HOME/lib64
ENV PATH=$PATH:$CUDA_HOME/bin
ENV CUPY_NUM_BUILD_JOBS=55
ENV CUPY_NVCC_GENERATE_CODE=current

RUN conda info && \
    conda install mamba -n base -c conda-forge && \
    mamba env create -f /demo/environment.yml && \
    mamba install -c conda-forge sysroot_linux-64=2.17 --yes && \
    conda clean --all && \
    rm -rf /opt/conda/envs/$CONDA_ENV_NAME/pkgs/ && \
    rm -rf /opt/conda/pkgs

SHELL ["conda", "run", "-n", "demo", "/bin/bash", "-c"]
RUN conda info

ENV CXXFLAGS="$CXXFLAGS -I$CUDA_HOME/include"
ENV CFLAGS="$CFLAGS -I$CUDA_HOME/include"
ENV LDFLAGS="${LDFLAGS} -Wl,-rpath-link,${CUDA_HOME}/lib64 -L${CUDA_HOME}/lib64 -L$CUDA_HOME/lib64/stubs"

RUN cd /demo/packages/cupy && python setup.py develop && \
    python -m pip install scipy && \
    cd /demo/packages/scikit-learn && python setup.py develop --no-deps && \
    python -m pip uninstall scipy -y && \
    cd /demo/packages/scipy && python dev.py --build-only && \
    conda clean --all && \
    rm -rf /opt/conda/envs/$CONDA_ENV_NAME/pkgs/ && \
    rm -rf /opt/conda/pkgs

ENV PYTHONPATH=$PYTHONPATH:/demo/packages/scipy/installdir/lib/python3.8/site-packages
RUN cd /demo/packages/scikit-image && python setup.py develop --no-deps && \
    echo "conda activate demo" >> ~/.bashrc
