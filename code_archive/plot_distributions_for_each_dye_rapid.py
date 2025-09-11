import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.patheffects as pe

def gaussian_pdf(x, mu, sigma, eps=1e-8):
    sigma = max(float(sigma), eps)  # avoid division by zero
    return (1.0 / (np.sqrt(2*np.pi) * sigma)) * np.exp(-0.5 * ((x - mu) / sigma)**2)

def plot_grid_gaussians(data_pairs, suptitle="Gaussian PDFs per dye"):
    """
    Plot a 6x6 grid; each cell shows two Gaussian PDFs estimated
    from two arrays of numbers.

    Parameters
    ----------
    data_pairs : list[tuple[np.ndarray, np.ndarray]]
        Up to 36 items. Each item is (arr1, arr2), arrays of numbers.
    suptitle : str
        Figure title.
    """
    if len(data_pairs) == 0:
        raise ValueError("data_pairs must contain at least one (arr1, arr2).")
    if len(data_pairs) > 36:
        raise ValueError("data_pairs can have at most 36 pairs (fits a 6x6 grid).")

    fig, axes = plt.subplots(6, 6, figsize=(14, 14), constrained_layout=True)
    axes = axes.ravel()

    # For consistent x-ranges per subplot, we compute per-cell ranges
    for i, ax in enumerate(axes):
        if i < len(data_pairs):
            a1, a2 = data_pairs[i]

            a1 = np.asarray(a1).ravel()
            a2 = np.asarray(a2).ravel()

            # Estimate mean/std
            mu1, s1 = float(np.mean(a1)), float(np.std(a1, ddof=1) if a1.size > 1 else 0.0)
            mu2, s2 = float(np.mean(a2)), float(np.std(a2, ddof=1) if a2.size > 1 else 0.0)

            # Choose an x-range that covers both distributions well
            # Use +/- 3 std around each mean; fallback if std ~ 0
            span1 = 3 * (s1 if s1 > 0 else max(1.0, abs(mu1) * 0.1 + 1.0))
            span2 = 3 * (s2 if s2 > 0 else max(1.0, abs(mu2) * 0.1 + 1.0))
            x_min = min(mu1 - span1, mu2 - span2)
            x_max = max(mu1 + span1, mu2 + span2)
            if x_min == x_max:
                x_min, x_max = x_min - 1.0, x_max + 1.0

            x = np.linspace(x_min, x_max, 400)

            # PDFs
            y1 = gaussian_pdf(x, mu1, s1)
            y2 = gaussian_pdf(x, mu2, s2)

            # Plot
            ax.plot(x, y1, label=f"Control: μ={mu1:.2f}, σ={s1:.2f}", linewidth=1.8)
            ax.plot(x, y2, label=f"MRSA: μ={mu2:.2f}, σ={s2:.2f}", linewidth=1.8)

            # Optional: light density histograms (comment out if not desired)
            # ax.hist(a1, bins="auto", density=True, alpha=0.15)
            # ax.hist(a2, bins="auto", density=True, alpha=0.15)

            ax.set_title(f"Dye {i+1}", fontsize=10)
            ax.tick_params(labelsize=8)
            ax.grid(alpha=0.2)
            ax.legend(fontsize=7, loc="upper right", frameon=False)
        else:
            ax.axis("off")

    fig.suptitle(suptitle, fontsize=16)
    plt.show()

def show_two_color_ribbons_on_ax(
    ax, A, B,
    ribbon_height=40, width_pixels=200, gap=6,
    common_scale=True, titles=None
):
    """Draw two color ribbons into a given matplotlib axis."""
    def _normalize_colors(arr, vmin=None, vmax=None, eps=1e-12):
        arr = np.asarray(arr, dtype=float)
        if arr.ndim != 2 or arr.shape[1] != 3:
            raise ValueError("Each matrix must have shape (N, 3).")
        if vmin is None: vmin, vmax = arr.min(), arr.max()
        if np.isclose(vmin, vmax):
            return np.zeros_like(arr)
        return np.clip((arr - vmin) / (vmax - vmin + eps), 0, 1)

    def _resample_colors(colors, target_len):
        n = colors.shape[0]
        if n == target_len:
            return colors
        x_old = np.linspace(0, 1, n)
        x_new = np.linspace(0, 1, target_len)
        out = np.empty((target_len, 3))
        for c in range(3):
            out[:, c] = np.interp(x_new, x_old, colors[:, c])
        return out

    A = np.asarray(A); B = np.asarray(B)

    # normalization
    if common_scale:
        vmin = min(A.min(), B.min())
        vmax = max(A.max(), B.max())
        A_norm = _normalize_colors(A, vmin, vmax)
        B_norm = _normalize_colors(B, vmin, vmax)
    else:
        A_norm = _normalize_colors(A)
        B_norm = _normalize_colors(B)

    # resample
    A_res = _resample_colors(A_norm, width_pixels)
    B_res = _resample_colors(B_norm, width_pixels)

    # build ribbon image
    height = ribbon_height * 2 + gap
    img = np.ones((height, width_pixels, 3))
    img[0:ribbon_height, :, :] = A_res[None, :, :]
    img[ribbon_height+gap:ribbon_height*2+gap, :, :] = B_res[None, :, :]

    # draw into subplot
    ax.imshow(img, aspect='auto')
    ax.axis("off")
    # label positions (in image coordinates)
    y_top    = ribbon_height / 2
    y_bottom = ribbon_height + gap + ribbon_height / 2
    x_left   = 6  # small left margin in pixels

    # nice readable text (white text with black outline)
    txt_style = dict(ha='left', va='center', fontsize=8, color='white',
                    path_effects=[pe.withStroke(linewidth=2, foreground='black')])

    ax.text(x_left, y_top,    "Control", **txt_style)  # name for top ribbon
    ax.text(x_left, y_bottom, "MRSA",    **txt_style)  # name for bottom ribbon
    if titles:
        ax.set_title(f"dye {i+1}", fontsize=8)

# -------------------------
# Example usage (remove/replace with your own arrays)
# Make 20 pairs of arrays with different means/stds (works with < 36)
if __name__ == "__main__":
    df = pd.read_csv('Cleaned_Table.csv')
    rng = np.random.default_rng(42)
    pairs_r = []
    pairs_g = []
    pairs_b = []
    
    for i in range(36):
        a_r = []
        b_r = []
        a_g = []
        b_g = []
        a_b = []
        b_b = []
        for k in range(14):
            a_r.append(df[f'Control {k+1}'][3*i]) # Red
            a_g.append(df[f'Control {k+1}'][3*i+1]) # Green
            a_b.append(df[f'Control {k+1}'][3*i+2]) # Blue
        for k in range(10):
            b_r.append(df[f'MRSA {k+1}'][3*i]) # Red
            b_g.append(df[f'MRSA {k+1}'][3*i+1]) # Green
            b_b.append(df[f'MRSA {k+1}'][3*i+2]) # Blue
        pairs_r.append((a_r, b_r))
        pairs_g.append((a_g, b_g))
        pairs_b.append((a_b, b_b))

    plot_grid_gaussians(pairs_r, suptitle="Two Gaussians per cell (Time = 120min) - Red")
    # plt.figure(figsize=(18, 18), constrained_layout=True)
    plot_grid_gaussians(pairs_g, suptitle="Two Gaussians per cell (Time = 120min) - Green")
    # plt.figure(figsize=(18, 18), constrained_layout=True)
    plot_grid_gaussians(pairs_b, suptitle="Two Gaussians per cell (Time = 120min) - Blue")
##########################################################################################
times = list(range(120, 601, 30))
mean_control = np.zeros((len(times),36, 3)) # [times, dye, rgb]
mean_MRSA = np.zeros((len(times), 36, 3))
# mean_b = np.zeros((len(times), 36, 3))
for j in range(len(times)):
    for i in range(36):
        a_r = []
        b_r = []
        a_g = []
        b_g = []
        a_b = []
        b_b = []
        for k in range(14):
            a_r.append(df[f'Control {k+1}'][108*j + 3*i]) # Red
            a_g.append(df[f'Control {k+1}'][108*j + 3*i + 1]) # Green
            a_b.append(df[f'Control {k+1}'][108*j + 3*i + 2]) # Blue
        for k in range(10):
            b_r.append(df[f'MRSA {k+1}'][108*j + 3*i]) # Red
            b_g.append(df[f'MRSA {k+1}'][108*j + 3*i + 1]) # Green
            b_b.append(df[f'MRSA {k+1}'][108*j + 3*i + 2]) # Blue
        
        mean_control[j,i,0] = np.mean(a_r)
        mean_control[j,i,1] = np.mean(a_g)
        mean_control[j,i,2] = np.mean(a_b)

        mean_MRSA[j,i,0] = np.mean(b_r)
        mean_MRSA[j,i,1] = np.mean(b_g)
        mean_MRSA[j,i,2] = np.mean(b_b)

# Example grid (6x6 for 36 items)
fig, axes = plt.subplots(6, 6, figsize=(15, 15))

for i in range(36):
    ax = axes[i // 6, i % 6]
    show_two_color_ribbons_on_ax(
        ax,
        np.squeeze(mean_control[:, i, :]),
        np.squeeze(mean_MRSA[:, i, :]),
        ribbon_height=15,   # smaller to fit grid
        width_pixels=100,   # fewer pixels to speed up
        gap=4,
        common_scale=True,
        titles=("Control", "MRSA")
    )

# plt.tight_layout()
plt.show()