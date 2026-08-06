"""Composite multiple normalized traces onto one 0-1 axis plot."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def render_overlay(traces, out_path, x_axis_label="x (normalized)",
                    y_axis_label="y (normalized)", figsize=(7, 6), dpi=150):
    """traces: list of dicts with keys x_norm, y_norm, label, color (matplotlib
    color spec, e.g. "blue" or "#ff8800"), alpha (0-1)."""
    fig, ax = plt.subplots(figsize=figsize)
    for t in traces:
        ax.plot(
            t["x_norm"], t["y_norm"],
            color=t.get("color", None),
            alpha=t.get("alpha", 0.7),
            linewidth=t.get("linewidth", 2),
            label=t.get("label", None),
        )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel(x_axis_label)
    ax.set_ylabel(y_axis_label)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
