"""Shared plotting style for final result figures."""

from __future__ import annotations


AXIS_COLOR = "#1f2933"
TICK_COLOR = "#263238"


def style_axes(ax, *, legend=None, xlabel_size: int = 13, ylabel_size: int = 13, tick_size: int = 11, legend_size: int = 10) -> None:
    """Make axis labels, ticks, and legends darker/bolder for paper figures."""
    ax.xaxis.label.set_size(xlabel_size)
    ax.xaxis.label.set_weight("bold")
    ax.xaxis.label.set_color(AXIS_COLOR)
    ax.yaxis.label.set_size(ylabel_size)
    ax.yaxis.label.set_weight("bold")
    ax.yaxis.label.set_color(AXIS_COLOR)
    ax.tick_params(axis="both", colors=TICK_COLOR, labelsize=tick_size, width=1.1)
    for tick_label in ax.get_xticklabels() + ax.get_yticklabels():
        tick_label.set_fontweight("bold")
        tick_label.set_color(TICK_COLOR)
    for spine in ax.spines.values():
        spine.set_color(TICK_COLOR)
        spine.set_linewidth(1.0)
    if legend is None:
        legend = ax.get_legend()
    if legend is not None:
        for text in legend.get_texts():
            text.set_fontsize(legend_size)
            text.set_fontweight("bold")
            text.set_color(AXIS_COLOR)
        title = legend.get_title()
        if title is not None:
            title.set_fontsize(legend_size)
            title.set_fontweight("bold")
            title.set_color(AXIS_COLOR)
