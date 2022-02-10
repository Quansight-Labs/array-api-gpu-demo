FROM gpuci/miniconda-cuda:11.2-devel-ubuntu20.04
#FROM nvidia/cuda:10.2-base
RUN apt-get update -y \
    && apt-get install -y \
    libxml2 \
    libxml2-dev

CMD nvidia-smi
RUN ls /usr/local/
WORKDIR /amd-demo
RUN mkdir /amd-demo/packages/ && cd /amd-demo/packages && \
	git clone https://github.com/aktech/scipy.git --branch amd-demo --recursive && \
	git clone https://github.com/aktech/scikit-learn.git --branch amd-demo  --recursive && \
	git clone https://github.com/aktech/scikit-image.git --branch amd-demo  --recursive && \
	git clone https://github.com/aktech/cupy.git --branch amd-demo --recursive

COPY environment.yml /amd-demo/environment.yml
ENV CONDA_ENV_NAME=docker-amd
RUN conda info
RUN conda install mamba -n base -c conda-forge
RUN conda info
RUN mamba env create -f /amd-demo/environment.yml

#ENV PATH /opt/conda/envs/$CONDA_ENV_NAME/bin:$PATH
#RUN echo "source activate $CONDA_ENV_NAME" >> ~/.bashrc
#SHELL ["/bin/bash", "--login", "-c"]
SHELL ["conda", "run", "-n", "docker-amd", "/bin/bash", "-c"]
RUN conda info

ENV CUDA_VERSION=11.2
ENV CUDA_PATH=/usr/local/cuda-$CUDA_VERSION
ENV CUDA_HOME=$CUDA_PATH
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_HOME/lib64
ENV PATH=$PATH:$CUDA_HOME/bin
ENV CUPY_NUM_BUILD_JOBS=55
ENV CUPY_NVCC_GENERATE_CODE=current

RUN conda install -c conda-forge sysroot_linux-64=2.17 --yes

ENV CXXFLAGS="$CXXFLAGS -I$CUDA_HOME/include"
ENV CFLAGS="$CFLAGS -I$CUDA_HOME/include"
ENV LDFLAGS="${LDFLAGS} -Wl,-rpath-link,${CUDA_HOME}/lib64 -L${CUDA_HOME}/lib64 -L$CUDA_HOME/lib64/stubs"

RUN cd /amd-demo/packages/cupy && python setup.py develop
RUN pip install scipy
RUN cd /amd-demo/packages/scikit-learn && python setup.py develop --no-deps
RUN pip uninstall scipy -y
RUN cd /amd-demo/packages/scipy && python dev.py --build-only
ENV PYTHONPATH=$PYTHONPATH:/amd-demo/packages/scipy/installdir/lib/python3.8/site-packages
RUN cd /amd-demo/packages/scikit-image && python setup.py develop --no-deps

RUN echo "conda activate docker-amd" >> ~/.bashrc
