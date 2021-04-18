import tqdm

from joblib import Parallel, delayed
import numpy as np

import PIL
import matplotlib.pyplot as plt


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

    image = PIL.Image.open(path)
    if width is None and height is None:
        width, height = image.size
    elif width is None:
        width = image.size[1] / height * image.size[0]
    elif height is None:
        height = image.size[0] / width * image.size[1]

    image = np.array(image.resize((width, height)).convert(mode='L')).reshape(height, width)

    if invert:
        image = 255 - image
    return image#.T[:,::-1]


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

    return np.vstack(hooks)


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
        for j in range(i + 1, len(hooks)):
            p1 = hooks[j]
            paths[(i, j)] = pixel_path(p0, p1)

    return paths


def loss(image, pixel_path, darkness=160, penalty=0, weight_type='mask'):
    old_pixel_values = image[tuple(pixel_path[:, ::-1].T)]
    # new_pixel_values = old_pixel_values - darkness


    # if weight_type == 'values':
    #     new_penalty = new_pixel_values[new_pixel_values > 0].sum() - \
    #                   penalty * new_pixel_values[new_pixel_values < 0].sum()
    #
    #     old_penalty = old_pixel_values[old_pixel_values > 0].sum() - \
    #                   penalty * old_pixel_values[old_pixel_values < 0].sum()
    # elif weight_type == 'mask':
    #     new_penalty = new_pixel_values.sum() - \
    #                   (1 + penalty) * new_pixel_values[new_pixel_values < 0].sum()
    #
    #     old_penalty = old_pixel_values.sum() - \
    #                   (1 + penalty) * old_pixel_values[old_pixel_values < 0].sum()
    #
    #     old_penalty = max(old_penalty, 0)

    line_norm = len(pixel_path)
    error = old_pixel_values.sum()
    error = max(error, 0)
    return error / line_norm
    # return (old_penalty - new_penalty) / line_norm




def optimize_line(image, start_hook, n_hooks, pixel_paths, **loss_kwargs):
    def is_vertical_or_horizontal(i, j):
        if i > j:
            i, j = j, i
        if pixel_paths[(i, j)][0, 0] == pixel_paths[(i, j)][-1, 0] or \
                pixel_paths[(i, j)][0, 1] == pixel_paths[(i, j)][-1, 1]:
            return True
        return False

    lines = [(i, start_hook) for i in range(start_hook)
             if not is_vertical_or_horizontal(start_hook, i)] + \
            [(start_hook, i) for i in range(start_hook + 1, n_hooks)
             if not is_vertical_or_horizontal(start_hook, i)]

    def get_line_loss(line):
        return loss(image, pixel_paths[line], **loss_kwargs)

    losses = Parallel(1)(delayed(get_line_loss)(line) for line in lines)
    optimal_line = sorted(zip(losses, lines), key=lambda x: x[0])
    return optimal_line[-1]


def optimize(image, n_lines, hooks, pixel_paths, darkness=160, penalty=0, weight_type='mask',
             show_plots=False):
    lines = []

    image = image.copy()
    n_hooks = len(hooks)

    if show_plots:
        fig, ax = plt.subplots(1, 1)
        ax.scatter(hooks[:, 0], hooks[:, 1], s=5)
        ax.set_xlim([-5, image.shape[1] + 5])
        ax.set_ylim([image.shape[0] + 5, -5])
        ax.set_aspect(image.shape[0] / image.shape[1])

    for i in range(n_lines):
        best_loss = -np.inf
        best_line = None

        start_hool_list = np.random.permutation(n_hooks)

        # losses = Parallel(-1)(delayed(optimize_line)(
        #     image, start_hook, n_hooks, pixel_paths, darkness=darkness,
        #     penalty=penalty)
        #     for start_hook in start_hool_list[:10]
        # )

        # losses = sorted(losses, key=lambda x: x[0])
        # best_loss, best_line = losses[-1]

        for start_hook in start_hool_list[:10]:
            loss, line = optimize_line(image, start_hook, n_hooks,
                                       pixel_paths, darkness=darkness,
                                       penalty=penalty, weight_type=weight_type)
            if loss > best_loss:
                best_loss = loss
                best_line = line

        lines.append(best_line)
        pixel_path = pixel_paths[best_line]
        image[tuple(pixel_path[:, ::-1].T)] -= darkness

        if show_plots:
            p0, p1 = hooks[best_line[0]], hooks[best_line[1]]
            ax.plot([p0[0], p1[0]], [p0[1], p1[1]],
                         c='k', lw=0.5, alpha=1)
            fig.suptitle(f'N lines: {i + 1}')
            fig.canvas.draw()

    return image, lines
