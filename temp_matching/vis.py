import numpy as np
from matplotlib import pyplot as plt
from typing import Tuple, List, Optional


def show_image(
    image: np.ndarray,
    title: Optional[str] = None,
    fig_size: Tuple[int, int] = (10, 10),
    axis: bool = False,
    show_image: bool = True,
):
    """
    A function to plot an image.

    Args:
    image: Image to plot.
    title: Title of the image.
    fig_size: Size of the figure.
    cmap: Colormap for the image.
    axis: Whether to show axis or not.

    Returns:
    fig: Figure object.

    """
    fig, ax = plt.subplots(figsize=fig_size)
    ax.imshow(image, vmax=255, vmin=0)
    if title:
        ax.set_title(title)
    if not axis:
        ax.axis("off")
    plt.tight_layout()

    if show_image:
        plt.show()
    return fig


def subplot_images(
    image: List[np.ndarray],
    titles: Optional[List[str]] = None,
    fig_size: Tuple[int, int] = (10, 10),
    order: Tuple[int, int] = (1, -1),
    axis: bool = False,
    show: bool = False,
):
    """
    A function to plot multiple images in a subplot.

    Args:
    image: List of images to plot.
    titles: List of titles for each image.
    fig_size: Size of the figure.
    cmap: Colormap for the images.
    order: Tuple of number of rows and columns for the subplot.
    axis: Whether to show axis or not.

    Returns:
    fig: Figure object.

    """

    if len(image) == 1:
        return show_image(
            image[0],
            titles[0] if titles else None,
            fig_size if fig_size else (5, 5),
            axis,
        )

    order = (
        (order[0], len(image) // order[0])
        if order[1] == -1
        else (len(image) // order[1], order[1]) if order[0] == -1 else order
    )

    fig, axs = plt.subplots(order[0], order[1], figsize=fig_size)
    if order[0] == 1 and order[1] == 1:
        axs = np.array([axs])
    for i, ax in enumerate(axs.flatten()):
        ax.imshow(image[i], vmax=255, vmin=0)
        if titles:
            ax.set_title(titles[i])
        if not axis:
            ax.axis("off")
    if show:
        plt.show()
    return fig


def figure_to_array(fig: plt.Figure) -> np.ndarray:
    fig.tight_layout(pad=1)
    # Render the figure to a canvas and get the RGB buffer
    fig.canvas.draw()

    # Get the width and height of the figure in pixels
    width, height = fig.canvas.get_width_height()

    # Convert to a NumPy array
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype="uint8")
    image = image.reshape(height, width, 3)  # Reshape to (height, width, RGB)

    return image
