# Demo: Segmentation on CPU and GPU

This is a demonstration of the performance benefits of using SciPy, scikit-learn
and scikit-image on GPUs (AMD or NVIDIA) using Array API.

## Running the demo

First login to Docker Registry

```bash
export CR_PAT=YOUR_PERSONAL_ACCESS_TOKEN_FOR_GITHUB
echo $CR_PAT | docker login ghcr.io -u USERNAME --password-stdin

```


### On NVIDIA GPU

```
docker run --gpus all -it -p 8788:8788 ghcr.io/quansight-labs/array-api-gpu-demo-cuda:latest bash
```

and then run jupyterlab inside the container:

```
jupyter lab --ip=0.0.0.0 --port=8788 --allow-root
```

### On AMD GPU

```
docker run -it --device=/dev/kfd --device=/dev/dri --security-opt seccomp=unconfined --group-add video ghcr.io/quansight-labs/array-api-gpu-demo-rocm:latest bash
```

and then run jupyterlab inside the container:

```
jupyter lab --ip=0.0.0.0 --port=8788 --allow-root
```


Now run the `plot_coin_segmentation.ipynb` notebook.


### Running Segmentation Performance script

Get into the docker container for either of the above mentioned GPU platform
and run the following script:

```
python segmentation_performance.py
```


This will run the segmentation for various proportions of the greek coins
image for cupy and numpy array, i.e. on CPU and GPU.

This will also create a plot of the performance comparison between numpy and cupy,
which would look something like:

#### On NVIDIA GPU
![cupy vs numpy](numpy_vs_cupy_nvidia.png)

#### On AMD GPU
![cupy vs numpy](numpy_vs_cupy_amd.png)


### Plotting on common numpy performance scale

To get a better idea of the performance of AMD and NVIDIA GPUs, we plot the graph with
a common numpy performance plot, taking the slowest times from NVIDIA and AMD performance
data. This can be achieved via following:

- Run the segmentation on both platforms:

```
docker run --gpus all -p 8788:8788 -v ${PWD}/artifacts:/demo/artifacts ghcr.io/quansight-labs/array-api-gpu-demo-cuda:latest conda run -n demo python segmentation_performance.py
```


```
docker run -it -p 8788:8788 -v ${PWD}/artifacts:/demo/artifacts --device=/dev/kfd --device=/dev/dri --security-opt seccomp=unconfined --group-add video ghcr.io/quansight-labs/array-api-gpu-demo-rocm:latest bash
```


- Copy the artifacts of the both runs on a common machine. These artifacts can be found in the
`artifacts/` directory.

- Now run the command for plotting with the given data:

```
docker run -v ${PWD}/artifacts:/demo/artifacts ghcr.io/quansight-labs/array-api-gpu-demo-cuda:latest conda run -n demo python segmentation_performance.py --plot
```

- You can find the final plots in the `artifacts/` directory.
