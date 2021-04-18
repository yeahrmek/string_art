import argparse

from joblib import Parallel, delayed

import numpy as np

from skimage.color import rgb2gray
from skimage.transform import hough_line
from skimage import io

from sklearn.utils import shuffle
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt


def hough_transform(image, n_angles=360):
    """
    Perform Hough transform.

    Args:
        image: np.ndarray, shape=(h, w, d)

    Returns:
        intensities: np.ndarray, shape=(n_lines, n_angles)
            Integral value for each line

        angles: np.ndarray, shape=(n_angles, )
            Angles of the lines for which integral was computed

        distances: np.ndarray, shape=(n_dist, )
            Distance from the origin to the closes point on a line

        bounds: list
            Angles' and distances' upper and lower bounds
    """
    image = rgb2gray(image)
    tested_angles = np.linspace(-np.pi / 2, np.pi / 2, n_angles, endpoint=False)
    intensities, angles, distances = hough_line(image, theta=tested_angles)

    angle_step = 0.5 * np.diff(angles).mean()
    d_step = 0.5 * np.diff(distances).mean()

    bounds = [np.rad2deg(angles[0] - angle_step),
              np.rad2deg(angles[-1] + angle_step),
              distances[-1] + d_step, distances[0] - d_step]
    return intensities, angles, distances, bounds


def quantize_colors(image, n_colors):
    """
    Quantize image colors into given number of colors

    Args:
        image: np.ndarray, shape=(h, w, d)
            Input image (grayscale)

        n_colors: int
            Number of colors to quantize to

    Returns:
        image_quant: np.ndarray(h, w)
            Quantized image
    """
    if image.ndim == 3:
        image_array = np.reshape(image, (-1, image.shape[-1]))
    else:
        image_array = np.reshape(image, (-1, 1))
    image_array = image_array / 255

    image_array_train = shuffle(image_array, random_state=0)[:2000]
    kmeans = KMeans(n_clusters=n_colors, random_state=0).fit(image_array_train)

    labels = kmeans.predict(image_array)
    centers = kmeans.cluster_centers_
    image_quant = np.reshape(centers[labels], image.shape)

    image_quant /= image_quant.max()
    return image_quant


def draw_threads(image, intensities, angles, distances, n_points=30,
                 threshold=0.2, lw=0.1):
    """

    Args:
        image: np.ndarray, shape=(h, w)
            Quantized image

        intensities: np.ndarray, shape=(n_lines, n_angles)
            Intensities along lines from Hough transform

        angles: np.ndarray, shape=(n_angles, )
            Angles of lines from Hough transform

        distances: np.ndarray, shape=(n_distances, )
            Distances from origin to the closes point on a line

        n_points: int
            Number of points (nails)

        threshold: float
            Threshold to remove lines with low intensity

    Returns:

    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 6))

    axes[0].imshow(image, cmap=plt.cm.gray)

    ax = axes[1]
    ax.set_aspect(1)

    ax.set_ylim((image.shape[0], 0))
    ax.set_xlim((0, image.shape[1]))
    ax.set_axis_off()
    ax.set_title('Detected lines')

    center = np.array((image.shape[0] // 2, image.shape[1] // 2))
    r = np.sqrt(np.mean(center ** 2))
    alpha_list = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    points = np.stack((r * np.sin(alpha_list), r * np.cos(alpha_list)), axis=1) + \
             center.reshape(1, -1)

    def get_closest(array, y, dist_fn):
        best_dist = np.inf
        best_idx = None
        for i, x in enumerate(array):

            if best_dist > dist_fn(x, y):
                best_dist = dist_fn(x, y)
                best_idx = i
        return best_idx

    def get_line_indices(d, theta, point0, point1):
        if point0[0] != point1[0]:
            slope = (point0[1] - point1[1]) / (point0[0] - point1[0])
        else:
            slope = np.tan(np.pi / 2)

        i_angle = get_closest(theta, slope,
                              lambda x, y: abs(np.tan(x + np.pi / 2) - y))
        angle = theta[i_angle]

        i_dist = get_closest(d, point0,
                             lambda x, y: np.linalg.norm(y - x * np.array([np.cos(angle),
                                                                           np.sin(angle)])))

        return i_dist, i_angle

    h_idx = Parallel(-1)(delayed(get_line_indices)(distances, angles, point0, point1)
                         for i, point0 in enumerate(points)
                         for point1 in points[i + 1:])

    alpha_list = np.array([intensities[i, j] for i, j in h_idx])**2
    alpha_list = alpha_list / alpha_list.max()

    for (i, j), alpha in zip(h_idx, alpha_list):
        dist = distances[i]
        angle = angles[j]
        (x0, y0) = dist * np.array([np.cos(angle), np.sin(angle)])

        if alpha > threshold:
            ax.axline((x0, y0), slope=np.tan(angle + np.pi / 2),
                      alpha=alpha,
                      c='k', lw=lw)
    ax.scatter(points[:, 0], points[:, 1], zorder=10, c='r')

    return fig, ax, alpha_list


def main(in_file, out_file=None):
    image = io.imread(in_file)

    image_quant = quantize_colors(1 - rgb2gray(image), n_colors=3)

    image_to_process = MinMaxScaler().fit_transform(image_quant)
    image_to_process[image_to_process == 0.62250887] = 1
    image_to_process = np.round(image_to_process)

    intensities, angles, distances, _ = hough_transform(1 - image_to_process)

    fig, ax, alpha = draw_threads(image_to_process,
                                  intensities, angles, distances,
                                  n_points=100, threshold=0.3)
    if out_file is None:
        return fig

    fig.savefig(out_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_file', type=str, help='Path to input picture')
    parser.add_argument('--out_file', type=str, help='Path to output picture')

    args = parser.parse_args()

    main(args.in_file, args.out_file)

