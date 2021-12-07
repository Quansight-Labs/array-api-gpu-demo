set -ex
echo "Set environment variables"
export CUDA_PATH=/usr/local/cuda-11.2.0
export LD_LIBRARY_PATH=$CUDA_PATH/lib64
export CUPY_NVCC_GENERATE_CODE=current
export CUPY_NUM_BUILD_JOBS=40

echo "Install cupy"
cd /home/aktech/cupy
python setup.py develop
cd /home/aktech/amd_demo
echo "Create conda environment"
conda env create -f environment.yml
