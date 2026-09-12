"""
Scientific visualization utilities for CERTUS-S2 Streamlit application.
Renders research-grade matplotlib figures for reconstruction, support, risk,
proposal candidates, and certified selections.
"""

from typing import Any, Optional
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap


# Research aesthetic styling defaults
FIG_BG_COLOR = "#0e1117"
AXES_BG_COLOR = "#161b22"
TEXT_COLOR = "#e6edf3"
GRID_COLOR = "#30363d"
ACCENT_GREEN = "#10b981"
ACCENT_AMBER = "#f59e0b"
ACCENT_RED = "#ef4444"
ACCENT_BLUE = "#3b82f6"


def _setup_axis(ax: plt.Axes, title: str, subtitle: Optional[str] = None):
    """Applies clean, research-grade dark theme styling to a matplotlib axis."""
    ax.set_facecolor(AXES_BG_COLOR)
    ax.tick_params(colors=TEXT_COLOR, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color(GRID_COLOR)
        spine.set_linewidth(1.0)
    
    if subtitle:
        full_title = f"{title}\n{subtitle}"
    else:
        full_title = title
    ax.set_title(full_title, color=TEXT_COLOR, fontsize=12, fontweight="semibold", pad=10)
    ax.set_xlabel("East Pixel (2.5m grid)", color=TEXT_COLOR, fontsize=9, labelpad=6)
    ax.set_ylabel("North Pixel (2.5m grid)", color=TEXT_COLOR, fontsize=9, labelpad=6)


def plot_reconstruction(
    sr_rgb: np.ndarray,
    l2a_rgb: Optional[np.ndarray] = None,
    show_comparison: bool = False,
    view_mode: str = "SPLIT"
) -> plt.Figure:
    """
    Renders Sentinel-2 L2A (10m native) vs SEN2SR 2.5m reconstruction.
    Supports view_mode: 'SPLIT', 'ORIGINAL', or 'RECONSTRUCTION'.
    """
    mode = view_mode.upper() if view_mode else ("SPLIT" if show_comparison else "RECONSTRUCTION")
    if show_comparison:
        mode = "SPLIT"

    if mode == "SPLIT" and l2a_rgb is not None:
        fig, axes = plt.subplots(1, 2, figsize=(13, 6.2), facecolor=FIG_BG_COLOR)
        
        # Left: Native 10m Sentinel-2 L2A
        ax0 = axes[0]
        ax0.imshow(l2a_rgb)
        _setup_axis(
            ax0,
            "SOURCE: Sentinel-2 L2A Observation",
            "Native 10m Pixel Grid (128×128 px) | B04-B03-B02 RGB"
        )
        ax0.set_xlabel("East Pixel (10m grid)", color=TEXT_COLOR, fontsize=9)
        ax0.set_ylabel("North Pixel (10m grid)", color=TEXT_COLOR, fontsize=9)
        
        # Right: SEN2SR 2.5m Reconstruction
        ax1 = axes[1]
        ax1.imshow(sr_rgb)
        _setup_axis(
            ax1,
            "RECONSTRUCTION: SEN2SR 2.5m Representation",
            "4× Super-Resolved Output Grid (512×512 px) | EPSG:32630"
        )
    elif mode == "ORIGINAL" and l2a_rgb is not None:
        fig, ax0 = plt.subplots(figsize=(7.5, 6.5), facecolor=FIG_BG_COLOR)
        ax0.imshow(l2a_rgb)
        _setup_axis(
            ax0,
            "SOURCE: Sentinel-2 L2A Observation (10m Native)",
            "Observed Measurement y | 128×128 native pixels | B04-B03-B02 RGB"
        )
        ax0.set_xlabel("East Pixel (10m grid)", color=TEXT_COLOR, fontsize=9)
        ax0.set_ylabel("North Pixel (10m grid)", color=TEXT_COLOR, fontsize=9)
    else:
        fig, ax1 = plt.subplots(figsize=(7.5, 6.5), facecolor=FIG_BG_COLOR)
        ax1.imshow(sr_rgb)
        _setup_axis(
            ax1,
            "RECONSTRUCTION: SEN2SR 2.5m Representation",
            "4× Super-Resolved Output Grid (512×512 px) | EPSG:32630"
        )

    plt.tight_layout()
    return fig


def plot_support_map(support_2d: np.ndarray) -> plt.Figure:
    """
    Renders continuous measurement support map s(p) in [0, 1].
    Higher support indicates stronger observational fidelity with Sentinel-2 L2A.
    """
    fig, ax = plt.subplots(figsize=(7.8, 6.5), facecolor=FIG_BG_COLOR)
    
    im = ax.imshow(support_2d, cmap="viridis", vmin=0.0, vmax=1.0)
    _setup_axis(
        ax,
        "Measurement Support Map s(p)",
        "s(p) ∈ [0, 1] derived from observation residual e = y - D̂(x̂)"
    )
    
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Measurement Support s(p)", color=TEXT_COLOR, fontsize=10)
    cbar.ax.tick_params(colors=TEXT_COLOR, labelsize=9)
    cbar.outline.set_edgecolor(GRID_COLOR)
    
    plt.tight_layout()
    return fig


def plot_risk_map(risk_2d: np.ndarray) -> plt.Figure:
    """
    Renders continuous decision risk map r(p) in [0, 1].
    Notice: Risk is a decision-risk score, not a probability that a pixel is hallucinated.
    """
    fig, ax = plt.subplots(figsize=(7.8, 6.5), facecolor=FIG_BG_COLOR)
    
    im = ax.imshow(risk_2d, cmap="magma", vmin=0.0, vmax=1.0)
    _setup_axis(
        ax,
        "Decision Risk Map r(p)",
        "Continuous decision-risk score ∈ [0, 1] (monotonic in residual & uncertainty)"
    )
    
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Decision Risk r(p)", color=TEXT_COLOR, fontsize=10)
    cbar.ax.tick_params(colors=TEXT_COLOR, labelsize=9)
    cbar.outline.set_edgecolor(GRID_COLOR)
    
    plt.tight_layout()
    return fig


def plot_proposed_detections(
    sr_rgb: np.ndarray,
    candidate_mask: np.ndarray,
    candidates: list[dict[str, Any]]
) -> plt.Figure:
    """
    Renders proposed detections (N=109) overlaid on SEN2SR reconstruction.
    Proposal stage uses fixed c0 = 0.02, independent of certification alpha.
    """
    fig, ax = plt.subplots(figsize=(7.8, 6.5), facecolor=FIG_BG_COLOR)
    
    # Render background SR image
    ax.imshow(sr_rgb)
    
    # Overlay candidate mask in semi-transparent amber
    overlay = np.zeros((*candidate_mask.shape, 4), dtype=np.float32)
    overlay[candidate_mask > 0] = [0.96, 0.62, 0.04, 0.45]  # Amber tint
    ax.imshow(overlay)
    
    # Plot contour outlines for crisp definition
    ax.contour(candidate_mask, levels=[0.5], colors=["#f59e0b"], linewidths=1.0)
    
    _setup_axis(
        ax,
        "Fixed D1 Proposed Structures",
        f"N = {len(candidates)} Candidates | Proposal c₀ = 0.02 (Fixed prior to certification)"
    )
    
    # Add legend patch
    amber_patch = mpatches.Patch(
        edgecolor="#f59e0b", facecolor=(0.96, 0.62, 0.04, 0.45),
        label=f"Proposed Built-Structures (N={len(candidates)})"
    )
    ax.legend(
        handles=[amber_patch],
        loc="upper right",
        facecolor=AXES_BG_COLOR,
        edgecolor=GRID_COLOR,
        labelcolor=TEXT_COLOR,
        fontsize=9
    )
    
    plt.tight_layout()
    return fig


def plot_certified_detections(
    sr_rgb: np.ndarray,
    certified_mask: np.ndarray,
    certified_candidates: list[dict[str, Any]],
    method_name: str,
    alpha: float,
    threshold_label: Optional[str] = None
) -> plt.Figure:
    """
    Renders certified detections overlaid on SEN2SR reconstruction.
    If certified_candidates is empty, renders the clean SR scene with a prominent
    'NO CERTIFIED SELECTION' badge and scientific abstention explanation.
    """
    fig, ax = plt.subplots(figsize=(7.8, 6.5), facecolor=FIG_BG_COLOR)
    
    ax.imshow(sr_rgb)
    n_cert = len(certified_candidates)
    
    if n_cert > 0 and np.any(certified_mask > 0):
        # Overlay certified mask in vibrant emerald green
        overlay = np.zeros((*certified_mask.shape, 4), dtype=np.float32)
        overlay[certified_mask > 0] = [0.06, 0.73, 0.51, 0.65]  # Emerald green tint
        ax.imshow(overlay)
        ax.contour(certified_mask, levels=[0.5], colors=["#10b981"], linewidths=1.5)
        
        # Annotate centroid of certified candidates with ID badge
        for c in certified_candidates:
            centroid = c.get("centroid", c.get("centroid_yx", (256, 256)))
            cy, cx = centroid
            r_val = c.get("risk", 0.0)
            ax.scatter([cx], [cy], s=40, color="#ffffff", edgecolors="#10b981", linewidths=1.5, zorder=5)
            ax.annotate(
                f"#{c['id']} (r={r_val:.3f})",
                xy=(cx, cy),
                xytext=(cx + 8, cy - 8),
                fontsize=8,
                color="#ffffff",
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="#064e3b", edgecolor="#10b981", alpha=0.85),
                arrowprops=dict(arrowstyle="->", color="#10b981", lw=1.0)
            )
            
        subtitle = f"{method_name} (α={alpha:.2f}) | {n_cert} Candidates Certified"
        if threshold_label:
            subtitle += f" | {threshold_label}"
            
        _setup_axis(ax, "Certified Candidate Selection", subtitle)
        
        green_patch = mpatches.Patch(
            edgecolor="#10b981", facecolor=(0.06, 0.73, 0.51, 0.65),
            label=f"Certified Detections ({n_cert} selected)"
        )
        ax.legend(
            handles=[green_patch],
            loc="upper right",
            facecolor=AXES_BG_COLOR,
            edgecolor=GRID_COLOR,
            labelcolor=TEXT_COLOR,
            fontsize=9
        )
    else:
        # Honest Empty Selection / Abstention Display
        _setup_axis(
            ax,
            "Certified Candidate Selection",
            f"{method_name} (α={alpha:.2f}) | 0 Candidates Certified"
        )
        
        # Draw clean central banner indicating NO CERTIFIED SELECTION
        ax.text(
            256, 230,
            "NO CERTIFIED SELECTION",
            fontsize=15,
            fontweight="bold",
            color="#f59e0b",
            ha="center",
            va="center",
            bbox=dict(
                boxstyle="round,pad=0.6",
                facecolor="#1c1917",
                edgecolor="#f59e0b",
                linewidth=1.5,
                alpha=0.92
            )
        )
        ax.text(
            256, 280,
            "Statistical criterion not met at current α\nSystem conservatively abstains rather than inventing false discoveries.",
            fontsize=9,
            color="#e2e8f0",
            ha="center",
            va="center",
            bbox=dict(
                boxstyle="round,pad=0.4",
                facecolor="#0f172a",
                edgecolor="#334155",
                linewidth=1.0,
                alpha=0.90
            )
        )

    plt.tight_layout()
    return fig
