import torch 
import numpy as np
import matplotlib.pyplot as plt

def plot_shock_examples(
    x: torch.Tensor,
    sample_rate: int | float,
    k: int = 5,
    title: str | None = None,
    alpha: float = 0.7,
):
    """
    Plot a few 1D series from a tensor of shape (N, L).
    - x: torch.Tensor (N,L) or (L,) n is the number of shocks, L number of elements per shock
    - k: number of time series to plot
    - alpha: line transparency (0=transparent, 1=opaque)
    """
    if isinstance(x, tuple):
        raise TypeError("plot_shock_examples expected a Tensor, got a tuple (x, y). Pass only x.")

    if x.ndim == 1:
        x = x.unsqueeze(0)  # (1, L)
    elif x.ndim != 2:
        raise ValueError(f"Expected x to have 1 or 2 dims, got shape {tuple(x.shape)}")

    N, L = x.shape

    if k > N:
        raise ValueError(
            f"Cannot request {k} time series when only {N} have been generated. "
            "Generate more series or request fewer."
        )

    # Move to CPU and convert to numpy for matplotlib
    x_np = x.detach().cpu().numpy()
    t = np.arange(L) * (1 / sample_rate)

    fig, ax = plt.subplots()

    for i in range(k):
        ax.plot(t, x_np[i], label=f"shock {i}", alpha=alpha)

    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Acceleration (G)")
    ax.set_title(title if title else f"Examples from tensor (n={N}, L={L})")

    # Legend outside on the right
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0)

    # Make room for the outside legend
    fig.tight_layout()
    fig.subplots_adjust(right=0.78)

    plt.show()


def plot_shock_and_srs_examples(
    x: torch.Tensor,
    y_srs: torch.Tensor,
    f: np.ndarray | torch.Tensor,
    sample_rate: int | float,
    k: int = 5,
    title: str | None = None,
    alpha: float = 0.7,
):
    """
    Two subplots:
      (1) SRS (log-log): y_srs vs f
      (2) Time series: x vs t
    Legend is placed to the right of the plots.
    Suptitle is two lines: "Time series ..." then "SRS ..."
    """
    if isinstance(x, tuple) or isinstance(y_srs, tuple):
        raise TypeError("Expected tensors, got a tuple. Pass only x and y_srs tensors.")

    # Normalize shapes
    if x.ndim == 1:
        x = x.unsqueeze(0)
    elif x.ndim != 2:
        raise ValueError(f"Expected x to have 1 or 2 dims, got shape {tuple(x.shape)}")

    if y_srs.ndim == 1:
        y_srs = y_srs.unsqueeze(0)
    elif y_srs.ndim != 2:
        raise ValueError(f"Expected y_srs to have 1 or 2 dims, got shape {tuple(y_srs.shape)}")

    if isinstance(f, torch.Tensor):
        f = f.detach().cpu().numpy()
    else:
        f = np.asarray(f)

    N, L = x.shape
    Ns, F = y_srs.shape

    if Ns != N:
        raise ValueError(f"x has N={N} series but y_srs has N={Ns} series (must match).")
    # Accept f shaped (F,) or (1, F). If (1, F), collapse to (F,).
    if f.ndim == 2 and f.shape[0] == 1 and f.shape[1] == F:
        f = f.squeeze(0)
    if f.ndim != 1 or f.shape[0] != F:
        raise ValueError(f"f must be shape ({F},) or (1,{F}), got {f.shape}.")
    if k > N:
        raise ValueError(f"Requested {k} examples, but only have {N}. (k must be <= N)")

    # Move to numpy
    x_np = x.detach().cpu().numpy()
    y_np = y_srs.detach().cpu().numpy()

    t = np.arange(L, dtype=np.float64) / float(sample_rate)

    # Leave room on the right for legend
    fig, (ax_srs, ax_time) = plt.subplots(2, 1, figsize=(10, 6))
    fig.subplots_adjust(right=0.80, hspace=0.35)

    # --- Top: SRS log-log ---
    for i in range(k):
        ax_srs.plot(f, y_np[i], label=f"shock {i}")
    ax_srs.set_xscale("log")
    ax_srs.set_yscale("log")
    ax_srs.set_xlabel("Frequency (Hz)")
    ax_srs.set_ylabel("SRS (G's)")  # change to your unit
    ax_srs.grid(True, which="both", linestyle=":", linewidth=0.6)

    # --- Bottom: time series ---
    for i in range(k):
        ax_time.plot(t, x_np[i], label=f"shock {i}", alpha=alpha)
    ax_time.set_xlabel("Time (seconds)")
    ax_time.set_ylabel("Acceleration (G)")
    ax_time.grid(True, which="both", linestyle=":", linewidth=0.6)

    # Two-line suptitle
    if title:
        fig.suptitle(title, y=0.98)
    else:
        fig.suptitle(
            f"Time series examples (N={N}, L={L}, fs={float(sample_rate):g} Hz)\n"
            f"SRS examples (F={F}, f=[{float(np.min(f)):g}, {float(np.max(f)):g}] Hz)",
            y=0.98
        )

    # Legend to the right (single legend for both subplots)
    handles, labels = ax_time.get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="center left",
        bbox_to_anchor=(0.82, 0.5),
        borderaxespad=0.0,
        fontsize="small"
    )

    plt.show()
