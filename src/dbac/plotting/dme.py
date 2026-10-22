"""
dme.py

Plotting for benchmarking the different implementations of the DME channel
(matrix exponential, numpy SWAP+ptrace, qibo circuit, analytical equivalent)
against each other, as a function of the number of Trotter steps M.
"""

__all__ = ["plot_dme_channel_norms"]

import matplotlib.pyplot as plt

_NORM_YLABEL = r'$|| e^{-it\rho}\sigma e^{it\rho} - \hat{E}_t^{(\rho)}(\sigma) ||$'


def plot_dme_channel_norms(norms, *, markers=None, colors=None, show_final_value=False,
                            xlabel='M', ylabel=_NORM_YLABEL, legend=True, grid=True,
                            save_path=None):
    """
    Plot the trace-norm distance between an exact (matrix-exponential) DME
    step and one or more approximate implementations, as a function of the
    number of Trotter steps M.

    Parameters:
        norms (dict[str, sequence[float]]): label -> norm values vs. M.
        markers, colors (sequence, optional): Styling cycled over `norms`.
        show_final_value (bool): If True, appends the final norm value to each
            curve's legend entry.
        xlabel, ylabel (str): Axis labels.
        legend (bool): Whether to draw a legend.
        grid (bool): Whether to draw a grid.
        save_path (str, optional): If given, saves the figure.

    Returns:
        matplotlib.axes.Axes
    """
    markers = markers or ['o', 'x', '^', 's', 'd']
    fig, ax = plt.subplots()

    for i, (label, values) in enumerate(norms.items()):
        kwargs = dict(marker=markers[i % len(markers)], alpha=0.8)
        if colors is not None:
            kwargs['color'] = colors[i % len(colors)]
        if markers[i % len(markers)] == 'o':
            kwargs['markerfacecolor'] = 'none'
        display_label = label
        if show_final_value:
            display_label = f'{label}, final norm = {float("%.3g" % values[-1])}'
        ax.plot(values, label=display_label, **kwargs)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if grid:
        ax.grid()
    if legend and len(norms) > 1:
        ax.legend()

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)

    return ax
