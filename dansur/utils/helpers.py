"""
Path, logging, and plotting helper functions.
"""

import os
import sys
from typing import Any, Optional


def get_stdout_path() -> str:
    """
    Get the path to stdout redirection file.
    """
    return os.path.join(os.getcwd(), 'stdout.txt')


def get_stderr_path() -> str:
    """
    Get the path to stderr redirection file.
    """
    return os.path.join(os.getcwd(), 'stderr.txt')


def get_folder_from_path(path_str: str) -> str:
    """
    Extract directory name from a path string.
    """
    return os.path.basename(os.path.normpath(path_str))


def plot_hist_from_binned_statistic(bin_edges, bin_means, color: Any = None):
    """
    Step plot helper for binned statistics.
    """
    import matplotlib.pyplot as plt
    first_plot = plt.step(bin_edges[1:], bin_means, color=color, where='pre')
    if color is None:
        color = first_plot[0].get_color()
    plt.step(bin_edges[:-1], bin_means, color=color, where='post', label='Binned Statistics')
