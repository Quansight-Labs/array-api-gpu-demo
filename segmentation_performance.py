# Author: Gael Varoquaux <gael.varoquaux@normalesup.org>, Brian Cheung
# Modified by: Amit Kumar <dtu.amit@gmail.com>
# License: BSD 3 clause
import os
import time

import tqdm

import numpy.array_api as npx
import cupy.array_api as cpx

from cupyx.scipy.ndimage import gaussian_filter as cupy_gaussian_filter
from cupyx.scipy import sparse as cupy_sparse

from scipy.ndimage import gaussian_filter as scipy_gaussian_filter

import matplotlib.pyplot as plt
import skimage
from skimage.data import coins
from skimage.transform import rescale as skimage_rescale

from sklearn.feature_extraction import image

from sklearn.cluster import spectral_clustering
from sklearn.utils.fixes import parse_version

from scipy import sparse as scipy_sparse

# these were introduced in skimage-0.14
if parse_version(skimage.__version__) >= parse_version("0.14"):
    rescale_params = {"anti_aliasing": False}
else:
    rescale_params = {}

beta = 10
eps = 1e-6
LABELLING_METHOD = "discretize"

# Apply spectral clustering (this step goes much faster if you have pyamg
# installed)
N_REGIONS = 25

RESIZE_PROPORTIONS = [0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1]


def create_image_graph(
        coins,
        filter,
        rescale,
        return_as=scipy_sparse.coo_matrix,
        resize_proportion=0.2
):
    # Resize it to 20% of the original size to speed up the processing
    # Applying a Gaussian filter for smoothing prior to down-scaling
    # reduces aliasing artifacts.
    smoothened_coins = filter(coins, sigma=2)
    # Taking the old array for cuCIM as we don't need to change cuCIM to support Array API
    # We'll change skimage to support the Array API rather.
    # smoothened_coins = smoothened_coins._array if hasattr(smoothened_coins, '_array') else smoothened_coins
    rescaled_coins = rescale(smoothened_coins, resize_proportion, mode="reflect", **rescale_params)

    # Convert the image into a graph with the value of the gradient on the
    # edges.
    graph = image.img_to_graph(rescaled_coins, return_as=return_as)
    return rescaled_coins, graph


# Take a decreasing function of the gradient: an exponential
# The smaller beta is, the more independent the segmentation is of the
# actual image. For beta=1, the segmentation is close to a voronoi
def set_graph_data(xp, graph):
    graph.data = xp.exp(-beta * xp.asarray(graph.data) / xp.std(xp.asarray(graph.data))) + eps
    graph.row = xp.asarray(graph.row)
    graph.col = xp.asarray(graph.col)
    return graph


def plot_(rescaled_coins, xp, labels):
    labels = xp.reshape(labels, rescaled_coins.shape)
    plt.figure(figsize=(5, 5))
    if xp.__name__ == 'cupy.array_api':
        # For plotting with Matplotlib we need a NumPy array
        rescaled_coins = rescaled_coins._array.get()
        labels = labels._array.get()

    plt.imshow(rescaled_coins, cmap=plt.cm.gray)
    for l in range(N_REGIONS):
        plt.contour(labels == l, colors=[plt.cm.nipy_spectral(l / float(N_REGIONS))])

    plt.xticks(())
    plt.yticks(())
    return plt


def segmentation(xp, coins, gaussian_filter, return_as, show_plot=False, resize_proportion=0.2):
    print(f"Running segmentation for {xp.__name__.split('.')[0]} for Resize proportion {resize_proportion}")
    rescaled_coins, graph = create_image_graph(
        coins, gaussian_filter, skimage_rescale,
        return_as=return_as, resize_proportion=resize_proportion
    )
    image_shape = rescaled_coins.shape
    graph = set_graph_data(xp, graph)
    t0 = time.time()
    labels = spectral_clustering(
        graph, n_clusters=N_REGIONS,
        assign_labels=LABELLING_METHOD,
        random_state=42
    )
    t1 = time.time()
    time_taken = t1 - t0
    if show_plot:
        plt = plot_(rescaled_coins, xp, labels)
        title = f"Spectral clustering via {xp}: %s, %.2fs" % (LABELLING_METHOD, (time_taken))
        print(title)
        plt.title(title)
        plt.show()

    return time_taken, image_shape


def run_segmentation_performance():
    numpy_times = []
    cupy_times = []
    image_sizes = []
    coins_npx = npx.asarray(coins())
    coins_cpx = cpx.asarray(coins())
    for r_proportion in tqdm.tqdm(RESIZE_PROPORTIONS):
        numpy_time, image_size = segmentation(
            xp=npx,
            coins=coins_npx,
            gaussian_filter=scipy_gaussian_filter,
            return_as=scipy_sparse.coo_matrix,
            resize_proportion=r_proportion
        )
        numpy_times.append(numpy_time)
        image_sizes.append(str(image_size))

        cupy_time, _ = segmentation(
            xp=cpx,
            coins=coins_cpx,
            gaussian_filter=cupy_gaussian_filter,
            return_as=cupy_sparse.coo_matrix,
            resize_proportion=r_proportion
        )
        cupy_times.append(cupy_time)

    plot_performance(cupy_times[1:], numpy_times[1:], image_sizes[1:])


def plot_performance(cupy_times, numpy_times, image_sizes):
    plt.plot(cupy_times, color="green", label="cupy")
    plt.plot(numpy_times, color="blue", label="numpy")

    # x-label
    xi = list(range(len(image_sizes)))
    plt.xticks(xi, image_sizes)

    plt.legend()
    plt.ylabel('Time Taken (sec)')
    plt.xlabel('Image Dimension')
    artifacts_path = 'artifacts'
    if not os.path.exists(artifacts_path):
        os.mkdir(artifacts_path)

    plot_path = os.path.join(artifacts_path, 'numpy_vs_cupy.png')
    plt.savefig(plot_path)


if __name__ == '__main__':
    print("Running segmentation performance")
    t1 = time.time()
    run_segmentation_performance()
    print(f"Total Time Taken: {time.time() - t1} sec")
