FROM gpuci/miniconda-cuda:11.2-devel-ubuntu20.04
RUN apt-get update -y \
    && apt-get install -y \
    libxml2 \
    libxml2-dev && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /amd-demo
RUN mkdir /amd-demo/packages/ && cd /amd-demo/packages && \
	git clone https://github.com/aktech/scipy.git --branch amd-demo --recursive && \
	git clone https://github.com/aktech/scikit-learn.git --branch amd-demo  --recursive && \
	git clone https://github.com/aktech/scikit-image.git --branch amd-demo  --recursive && \
	git clone https://github.com/aktech/cupy.git --branch amd-demo --recursive

COPY environment.yml /amd-demo/environment.yml
COPY plot_coin_segmentation.ipynb /amd-demo/plot_coin_segmentation.ipynb

ENV CONDA_ENV_NAME=docker-amd
ENV CUDA_VERSION=11.2
ENV CUDA_PATH=/usr/local/cuda-$CUDA_VERSION
ENV CUDA_HOME=$CUDA_PATH
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_HOME/lib64
ENV PATH=$PATH:$CUDA_HOME/bin
ENV CUPY_NUM_BUILD_JOBS=55
ENV CUPY_NVCC_GENERATE_CODE=current

RUN conda info && \
    conda install mamba -n base -c conda-forge && \
    mamba env create -f /amd-demo/environment.yml && \
    mamba install -c conda-forge sysroot_linux-64=2.17 --yes && \
    conda clean --all && \
    rm -rf /opt/conda/envs/$CONDA_ENV_NAME/pkgs/ && \
    rm -rf /opt/conda/pkgs

SHELL ["conda", "run", "-n", "docker-amd", "/bin/bash", "-c"]
RUN conda info

ENV CXXFLAGS="$CXXFLAGS -I$CUDA_HOME/include"
ENV CFLAGS="$CFLAGS -I$CUDA_HOME/include"
ENV LDFLAGS="${LDFLAGS} -Wl,-rpath-link,${CUDA_HOME}/lib64 -L${CUDA_HOME}/lib64 -L$CUDA_HOME/lib64/stubs"

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
