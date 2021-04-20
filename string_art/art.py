from joblib import Parallel, delayed
import numpy as np

import cv2
import matplotlib.pyplot as plt


def high_contrast(image):
    # Converting image to LAB Color model
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # Splitting the LAB image to different channels
    l, a, b = cv2.split(lab)

    # Applying CLAHE to L-channel
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)

    # Merge the CLAHE enhanced L-channel with the a and b channel
    limg = cv2.merge((cl, a, b))

    # Converting image from LAB Color model to RGB model
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return final


def prepare_image(path, width=None, height=None, invert=True):
    """
    Prepare image

    Args:
        path: str
            Path to the image

        width : int
            Width of the final image in pixels

        height: int
            Height of the final image in pixels
    Returns:

    """

    image = cv2.imread(path)

    # increase contrast
    image = high_contrast(image)

    # convert to grayscale
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if width is None and height is None:
        height, width = image.shape[:2]
    elif width is None:
        width = height / image.size[1] * image.size[0]
    elif height is None:
        height = width / image.size[0] * image.size[1]

    image = cv2.resize(image, (width, height))

    if invert:
        image = 255 - image
    return image


def get_hooks(n_hooks, width, height):
    """
    Generate hooks coordinates for the square image

    Args:
        n_hooks: int
            Number of hooks to generate

        width: int
            Width of the image in pixels

        height: int
            Height of the image in pixels

    Returns:
        list(tuple(int, int))
    """
    n_hooks_list = []
    for i in range(4, 1, -1):
        n_hooks_list.append(n_hooks // i + int((n_hooks % i) > 0))
        n_hooks -= n_hooks_list[-1]
    n_hooks_list.append(n_hooks)

    hooks = [
        np.stack((
            np.zeros(n_hooks_list[0]),
            np.linspace(0, width - 1, n_hooks_list[0], endpoint=False)),
            axis=1
        ),
        np.stack((
            np.linspace(0, height - 1, n_hooks_list[1], endpoint=False),
            np.ones(n_hooks_list[1]) * (width - 1)),
            axis=1
        ),
        np.stack((
            np.ones(n_hooks_list[2]) * (height - 1),
            np.linspace(width - 1, 0, n_hooks_list[2], endpoint=False)),
            axis=1
        ),
        np.stack((
            np.linspace(height - 1, 0, n_hooks_list[3], endpoint=False),
            np.zeros(n_hooks_list[3])),
            axis=1
        )
    ]

    return np.vstack(hooks).astype(int)


def pixel_path(p0, p1):
    """
    Return pixel path from p0 to p1

    Args:
        p0: tuple(float, float)
            Start pixel

        p1: tuple(float, float)
            End pixel

    Returns:
        list(tuple(float, float))
            Ordered list of pixels
    """
    p0 = np.array(p0)
    p1 = np.array(p1)

    max_x = max(p0[0], p1[0])
    max_y = max(p0[1], p1[1])

    path_length = max(np.linalg.norm(p1 - p0), 1)

    path = p0 + (p1 - p0).reshape(1, -1) / path_length * \
           np.arange(path_length + 1).reshape(-1, 1)
    path = np.clip(path, [0, 0], [max_x, max_y])

    path = np.unique(np.round(path), axis=0).astype(int)

    return path


def all_pixel_paths(hooks):
    """
    Build dictionary of all pixel paths between given pixels

    Args:
        hooks: np.ndarray

    Returns:
        dict
    """
    paths = {}
    for i, p0 in enumerate(hooks):
        for j, p1 in enumerate(hooks):
            if i == j:
                continue
            p1 = hooks[j]
            paths[(i, j)] = pixel_path(p0, p1)

    return paths


def loss(image, pixel_path):
    old_pixel_values = image[pixel_path[:, 1], pixel_path[:, 0]]
    error = old_pixel_values.sum()
    return error


def optimize(image, n_lines, hooks, pixel_paths, line_weight=15, line_width=3,
             min_offset=30, show_plots=False, min_loss=-500):
    lines = []

    line_mask = np.zeros_like(image)
    image = image.copy().astype(int)
    n_hooks = len(hooks)

    if show_plots:
        fig, ax = plt.subplots(1, 1)#, figsize=(6, 6))
        ax.scatter(hooks[:, 0], hooks[:, 1], s=5)
        ax.set_xlim([-5, image.shape[1] + 5])
        ax.set_ylim([image.shape[0] + 5, -5])
        ax.set_aspect('equal')

    start_hook = 0
    prev_hooks = [start_hook]
    for i in range(n_lines):

        best_loss = -np.inf
        best_line = None

        # def calc_loss(start_hook, cur_hook):
        #     if cur_hook in prev_hooks:
        #         return -np.inf
        #     path = pixel_paths[(start_hook, cur_hook)]
        #     return loss(image, path)

        # lines = [(start_hook, (start_hook + offset) % n_hooks)
        #          for start_hook in np.random.permutation(n_hooks)[:10]
        #          for offset in range(min_offset, n_hooks - min_offset)]
        #
        # losses = Parallel(1)(delayed(calc_loss)(l[0], l[1]) for l in lines)
        #
        # best_loss, best_line = max(zip(losses, lines), key=lambda x: x[0])

        # if best_loss < 0:
        #     continue

        for start_hook in np.random.permutation(n_hooks)[:10]:
            for offset in range(min_offset, n_hooks - min_offset):

                cur_hook = (start_hook + offset) % n_hooks

                if cur_hook in prev_hooks:
                    continue

                path = pixel_paths[(start_hook, cur_hook)]
                loss_val = loss(image, path)

                if loss_val > best_loss:
                    best_loss = loss_val
                    best_line = (start_hook, cur_hook)

        if best_loss < min_loss:
            continue

        lines.append(best_line)
        line_mask = line_mask * 0
        cv2.line(line_mask, tuple(hooks[best_line[0]]),
                 tuple(hooks[best_line[1]]), line_weight, line_width)
        image = image - line_mask.astype(int)

        prev_hooks.append(best_line[1])
        if len(prev_hooks) > 20:
            prev_hooks = prev_hooks[1:]

        if show_plots:
            p0, p1 = hooks[best_line[0]], hooks[best_line[1]]
            ax.plot([p0[0], p1[0]], [p0[1], p1[1]],
                         c='k', lw=0.5, alpha=1)
            fig.suptitle(f'N lines: {i + 1}\nLoss: {best_loss}')
            fig.canvas.draw()

    return image, lines


def find_lines(image, n_hooks, n_lines, line_weight, line_width, show_progress=True):
    hooks = get_hooks(n_hooks, *image.shape)

    pixel_paths = all_pixel_paths(hooks)

    image_opt, lines = optimize(image, n_lines, hooks, pixel_paths,
                                line_weight=line_weight, line_width=line_width,
                                show_plots=show_progress)
    return lines