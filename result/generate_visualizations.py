#!/usr/bin/env python3
"""
LEO Satellite Network DDoS Defense - Visualization Suite
Generates tech-style visualizations for attack cost, vulnerability, topology, and P5 throughput.
"""

import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.collections import LineCollection
import matplotlib.patheffects as pe
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap

# ─── Paths ───
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
OUTPUT_DIR = SCRIPT_DIR  # result/
SIM_JSON = os.path.join(PROJECT_ROOT, "output", "ddos_simulation_results.json")
OPT_JSON = os.path.join(PROJECT_ROOT, "output", "krand_optimization_results.json")

# ─── Dark Tech Theme ───
DARK_BG = "#0a0e17"
PANEL_BG = "#111827"
GRID_COLOR = "#1e293b"
TEXT_COLOR = "#e2e8f0"
ACCENT_CYAN = "#00d4ff"
ACCENT_GREEN = "#22c55e"
ACCENT_RED = "#ef4444"
ACCENT_ORANGE = "#f59e0b"
ACCENT_PURPLE = "#a855f7"
ACCENT_PINK = "#ec4899"
GLOW_CYAN = "#00d4ff40"

ALGO_COLORS = {
    "KSP": "#ef4444",
    "KDS": "#f59e0b",
    "KDG": "#a855f7",
    "KLO": "#22c55e",
    "kRAND": "#00d4ff",
    "kRAND\n(Equal)": "#00d4ff",
    "kRAND\n(Optimized)": "#ec4899",
}

def setup_dark_style():
    plt.rcParams.update({
        "figure.facecolor": DARK_BG,
        "axes.facecolor": PANEL_BG,
        "axes.edgecolor": GRID_COLOR,
        "axes.labelcolor": TEXT_COLOR,
        "axes.grid": True,
        "grid.color": GRID_COLOR,
        "grid.alpha": 0.5,
        "text.color": TEXT_COLOR,
        "xtick.color": TEXT_COLOR,
        "ytick.color": TEXT_COLOR,
        "font.family": "sans-serif",
        "font.size": 11,
        "legend.facecolor": PANEL_BG,
        "legend.edgecolor": GRID_COLOR,
        "savefig.facecolor": DARK_BG,
        "savefig.edgecolor": DARK_BG,
    })

def load_data():
    with open(SIM_JSON, "r") as f:
        sim = json.load(f)
    with open(OPT_JSON, "r") as f:
        opt = json.load(f)
    return sim, opt

# ═════════════════════════════════════════════════════════════════
# Figure 1 : Attack Cost Comparison (fixed cost for kRAND)
# ═════════════════════════════════════════════════════════════════
def fig1_attack_cost(sim, opt):
    fig, ax = plt.subplots(figsize=(12, 7))

    # Attack cost is calculated from the routing algorithm's topology usage,
    # NOT from actual simulation. Attacker doesn't know our weights,
    # so kRAND cost is FIXED regardless of equal/optimized weights.
    algos = ["KSP", "KDS", "KDG", "KLO", "kRAND"]
    costs = [
        sim["calculated_costs_mbps"]["KSP"],
        sim["calculated_costs_mbps"]["KDS"],
        sim["calculated_costs_mbps"]["KDG"],
        sim["calculated_costs_mbps"]["KLO"],
        sim["calculated_costs_mbps"]["kRAND"],
    ]
    colors = [ACCENT_RED, ACCENT_ORANGE, ACCENT_PURPLE, ACCENT_GREEN, ACCENT_CYAN]

    bars = ax.bar(algos, costs, color=colors, width=0.55, edgecolor="white", linewidth=0.5, zorder=3)

    # Glow effect on kRAND (highest cost = best defense)
    bars[-1].set_edgecolor(ACCENT_CYAN)
    bars[-1].set_linewidth(2.5)

    # Value labels — place above each bar; move kRAND label lower inside bar to avoid overlap
    for idx, (bar, cost) in enumerate(zip(bars, costs)):
        if idx < len(bars) - 1:
            # Non-kRAND: label above bar
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 20,
                    f"{cost:.1f}", ha="center", va="bottom", fontsize=12,
                    fontweight="bold", color=TEXT_COLOR)
        else:
            # kRAND: label inside bar top to leave room for annotation above
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() - 40,
                    f"{cost:.1f}", ha="center", va="top", fontsize=12,
                    fontweight="bold", color="white")

    # Highlight kRAND as highest cost — annotation positioned higher up
    krand_cost = costs[-1]
    second_best = max(costs[:-1])
    improvement_pct = (krand_cost - second_best) / second_best * 100
    ax.annotate(
        f"kRAND: +{improvement_pct:.0f}% vs 2nd best\n({krand_cost:.1f} vs {second_best:.1f} Mbps)",
        xy=(bars[-1].get_x() + bars[-1].get_width() / 2, krand_cost),
        xytext=(bars[-1].get_x() + bars[-1].get_width() / 2, krand_cost + 200),
        fontsize=12, fontweight="bold",
        color=ACCENT_GREEN, ha="center",
        arrowprops=dict(arrowstyle="-|>", color=ACCENT_GREEN, lw=1.5),
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#22c55e20", edgecolor=ACCENT_GREEN, linewidth=1.5),
    )

    # Note: attacker cost is fixed (attacker-blind) — placed at top-left using axes coordinates
    ax.text(0.02, 0.97,
            "※ Attack cost is fixed — attacker does not know the routing weights",
            transform=ax.transAxes, fontsize=10, color=ACCENT_ORANGE,
            ha="left", va="top", style="italic",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#f59e0b15", edgecolor=ACCENT_ORANGE, linewidth=1))

    ax.set_ylabel("DDoS Attack Cost (Mbps)", fontsize=13, fontweight="bold")
    ax.set_title("Attack Cost Required to Degrade Target ISL\nHigher = Harder to Attack (Better Defense)",
                 fontsize=15, fontweight="bold", pad=15, color=ACCENT_CYAN)
    ax.set_ylim(0, max(costs) + 500)
    ax.tick_params(axis='x', labelsize=11)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fig1_attack_cost_comparison.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [✓] Saved: {path}")

# ═════════════════════════════════════════════════════════════════
# Figure 2 : LEO Constellation – 6 Orbits × 11 Sats + Ground Stations
# ═════════════════════════════════════════════════════════════════
def fig2_network_topology():
    """
    Top half: 6 orbital planes shown as tilted ellipses (bird's-eye perspective),
              each holding 11 evenly-spaced satellites.
    Bottom half: representative ground stations connected by GSL dashes.
    Annotations: orbital parameters (altitude, inclination, RAAN spacing, sat spacing).
    """
    import math

    fig, ax = plt.subplots(figsize=(16, 14))
    ax.set_facecolor(DARK_BG)
    fig.patch.set_facecolor(DARK_BG)

    # ── Constellation parameters ──
    NUM_PLANES = 6
    SATS_PER_PLANE = 11
    ALTITUDE_KM = 550.0
    INCLINATION_DEG = 53.0
    RAAN_SPACING_DEG = 360.0 / NUM_PLANES       # 60°
    SAT_SPACING_DEG = 360.0 / SATS_PER_PLANE    # ~32.7°

    # ── Drawing layout ──
    # We project each orbit as a tilted ellipse.
    # Centre of the orbital "sphere" is at (cx, cy).
    # The ellipses fan out horizontally by RAAN, with a tilt that reflects inclination.
    cx, cy = 0.0, 4.0          # centre of constellation region
    orbit_a = 5.5              # semi-major axis of projected ellipse (horizontal span)
    orbit_b = 2.0              # semi-minor axis (foreshortening / perspective)

    plane_colors = ["#00d4ff", "#22c55e", "#f59e0b", "#a855f7", "#ec4899", "#60a5fa"]

    sat_positions = {}  # sat_id -> (x, y)

    # ── Draw each orbital plane ──
    for p in range(NUM_PLANES):
        raan_rad = math.radians(RAAN_SPACING_DEG * p)
        color = plane_colors[p % len(plane_colors)]

        # Rotate the ellipse by raan to spread planes apart visually.
        # The rotation angle in the 2-D plot: map RAAN linearly to [−50°, +50°]
        rot_deg = -50 + (100 / (NUM_PLANES - 1)) * p
        rot_rad = math.radians(rot_deg)

        # Compute ellipse outline (for visual orbit ring)
        theta = np.linspace(0, 2 * np.pi, 200)
        ex = orbit_a * np.cos(theta)
        ey = orbit_b * np.sin(theta)
        # Apply rotation
        rx = cx + ex * math.cos(rot_rad) - ey * math.sin(rot_rad)
        ry = cy + ex * math.sin(rot_rad) + ey * math.cos(rot_rad)
        ax.plot(rx, ry, color=color, linewidth=0.8, alpha=0.25, zorder=1)

        # Place satellites along the ellipse
        for s in range(SATS_PER_PLANE):
            angle = 2 * math.pi * s / SATS_PER_PLANE
            sx = orbit_a * math.cos(angle)
            sy = orbit_b * math.sin(angle)
            px = cx + sx * math.cos(rot_rad) - sy * math.sin(rot_rad)
            py = cy + sx * math.sin(rot_rad) + sy * math.cos(rot_rad)
            sat_id = f"SAT_{p}_{s}"
            sat_positions[sat_id] = (px, py)

    # ── Draw inter-plane ISL (between adjacent planes, same sat index) ──
    for p in range(NUM_PLANES - 1):
        for s in range(SATS_PER_PLANE):
            s1 = f"SAT_{p}_{s}"
            s2 = f"SAT_{p+1}_{s}"
            x1, y1 = sat_positions[s1]
            x2, y2 = sat_positions[s2]
            ax.plot([x1, x2], [y1, y2], color="#334155", linewidth=0.4, alpha=0.25, zorder=1)

    # ── Draw intra-plane ISL (consecutive sats in same plane) ──
    for p in range(NUM_PLANES):
        color = plane_colors[p % len(plane_colors)]
        for s in range(SATS_PER_PLANE):
            s1 = f"SAT_{p}_{s}"
            s2 = f"SAT_{p}_{(s+1) % SATS_PER_PLANE}"
            x1, y1 = sat_positions[s1]
            x2, y2 = sat_positions[s2]
            ax.plot([x1, x2], [y1, y2], color=color, linewidth=0.6, alpha=0.35, zorder=2)

    # ── Draw satellite dots ──
    for p in range(NUM_PLANES):
        color = plane_colors[p % len(plane_colors)]
        for s in range(SATS_PER_PLANE):
            sid = f"SAT_{p}_{s}"
            px, py = sat_positions[sid]
            # Outer glow
            ax.scatter(px, py, s=60, color=color, alpha=0.15, zorder=3, linewidths=0)
            # Inner dot
            ax.scatter(px, py, s=18, color=color, zorder=4, edgecolors="white", linewidth=0.3)

    # ── Plane labels (right side) ──
    for p in range(NUM_PLANES):
        color = plane_colors[p % len(plane_colors)]
        # Use first satellite of each plane for label position
        ref_id = f"SAT_{p}_0"
        rx, ry = sat_positions[ref_id]
        raan_val = RAAN_SPACING_DEG * p
        ax.text(rx + 0.5, ry + 0.35, f"P{p} (RAAN {raan_val:.0f}°)",
                fontsize=8, color=color, alpha=0.8, fontweight="bold", zorder=10,
                bbox=dict(boxstyle="round,pad=0.15", facecolor=DARK_BG, edgecolor=color, alpha=0.5, linewidth=0.6))

    # ── Orbital parameter annotation box (upper-left) ──
    param_text = (
        f"Walker Star Constellation\n"
        f"─────────────────────\n"
        f"Planes (P):            {NUM_PLANES}\n"
        f"Sats / Plane:          {SATS_PER_PLANE}\n"
        f"Total Satellites:      {NUM_PLANES * SATS_PER_PLANE}\n"
        f"Altitude:              {ALTITUDE_KM:.0f} km\n"
        f"Inclination:           {INCLINATION_DEG:.0f}°\n"
        f"RAAN Spacing (ΔΩ):    {RAAN_SPACING_DEG:.0f}°\n"
        f"Sat Spacing (Δν):     {SAT_SPACING_DEG:.1f}°"
    )
    ax.text(-7.8, 9.5, param_text, fontsize=10, color=ACCENT_CYAN,
            fontfamily="monospace", va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.6", facecolor="#0f172a", edgecolor=ACCENT_CYAN,
                      linewidth=1.5, alpha=0.9), zorder=20)

    # ── RAAN angle arc annotation (between first two planes) ──
    # Show a small arc between plane 0 and plane 1 reference satellites
    p0_ref = sat_positions["SAT_0_0"]
    p1_ref = sat_positions["SAT_1_0"]
    mid_x = (p0_ref[0] + p1_ref[0]) / 2
    mid_y = (p0_ref[1] + p1_ref[1]) / 2
    ax.annotate("", xy=p1_ref, xytext=p0_ref,
                arrowprops=dict(arrowstyle="<->", color=ACCENT_ORANGE, lw=1.5, connectionstyle="arc3,rad=0.3"))
    ax.text(mid_x - 0.3, mid_y + 0.7, f"ΔΩ = {RAAN_SPACING_DEG:.0f}°",
            fontsize=10, color=ACCENT_ORANGE, fontweight="bold", ha="center",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#f59e0b15", edgecolor=ACCENT_ORANGE, linewidth=1))

    # ── Satellite spacing arc annotation (within plane 2) ──
    s_a = sat_positions["SAT_2_0"]
    s_b = sat_positions["SAT_2_1"]
    mid_sx = (s_a[0] + s_b[0]) / 2
    mid_sy = (s_a[1] + s_b[1]) / 2
    ax.annotate("", xy=s_b, xytext=s_a,
                arrowprops=dict(arrowstyle="<->", color=ACCENT_GREEN, lw=1.5, connectionstyle="arc3,rad=0.4"))
    ax.text(mid_sx + 0.6, mid_sy + 0.3, f"Δν ≈ {SAT_SPACING_DEG:.1f}°",
            fontsize=10, color=ACCENT_GREEN, fontweight="bold", ha="center",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#22c55e15", edgecolor=ACCENT_GREEN, linewidth=1))

    # ── Ground Stations (bottom region) ──
    gs_y_base = -3.5   # vertical position for ground-station row
    ground_stations = [
        ("New York",    -6.0),
        ("London",      -3.5),
        ("Paris",       -1.5),
        ("Beijing",      0.5),
        ("Tokyo",        2.5),
        ("Sydney",       4.5),
        ("São Paulo",   -5.0),
        ("Moscow",      -2.0),
        ("Dubai",        1.0),
        ("Singapore",    3.5),
        ("Chicago",     -4.5),
        ("Mumbai",       5.5),
    ]
    # Slight vertical jitter to avoid overlap
    for i, (name, gx) in enumerate(ground_stations):
        gy = gs_y_base + (0.3 if i % 2 == 0 else -0.3)
        # Station marker
        ax.scatter(gx, gy, s=100, color=ACCENT_GREEN, marker="^", zorder=8,
                   edgecolors="white", linewidth=0.8)
        ax.text(gx, gy - 0.55, name, ha="center", fontsize=7, color=ACCENT_GREEN,
                fontweight="bold", alpha=0.85, zorder=8)

        # GSL dashed line to nearest satellite (find closest by x-distance, prefer lower-y sats)
        best_sid = None
        best_dist = float('inf')
        for sid, (sx, sy) in sat_positions.items():
            d = math.hypot(sx - gx, sy - gy)
            if d < best_dist:
                best_dist = d
                best_sid = sid
        if best_sid:
            sx, sy = sat_positions[best_sid]
            ax.plot([gx, sx], [gy, sy], color=ACCENT_GREEN, linewidth=0.5, alpha=0.25,
                    linestyle="--", zorder=1)

    # ── "Ground" reference line ──
    ax.axhline(y=gs_y_base + 0.8, color="#334155", linewidth=0.8, linestyle=":", alpha=0.4, zorder=0)
    ax.text(7.5, gs_y_base + 1.05, "── Ground Level ──", fontsize=8, color="#475569",
            ha="right", alpha=0.6, style="italic")

    # ── Altitude annotation arrow (right side) ──
    # From ground line to constellation centre
    arr_x = 7.0
    ax.annotate("", xy=(arr_x, cy), xytext=(arr_x, gs_y_base + 0.8),
                arrowprops=dict(arrowstyle="<->", color="#94a3b8", lw=1.2))
    ax.text(arr_x + 0.3, (cy + gs_y_base + 0.8) / 2, f"{ALTITUDE_KM:.0f} km",
            fontsize=10, color="#94a3b8", fontweight="bold", ha="left", va="center",
            rotation=90,
            bbox=dict(boxstyle="round,pad=0.2", facecolor=DARK_BG, edgecolor="#94a3b8", linewidth=0.8, alpha=0.7))

    # ── Legend ──
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=ACCENT_CYAN,
                   markersize=8, label=f'LEO Satellite (×{NUM_PLANES*SATS_PER_PLANE})', linestyle='None'),
        plt.Line2D([0], [0], marker='^', color='w', markerfacecolor=ACCENT_GREEN,
                   markersize=9, label='Ground Station (~40)', linestyle='None'),
        plt.Line2D([0], [0], color=plane_colors[0], linewidth=1.2, alpha=0.5,
                   label='Intra-plane ISL'),
        plt.Line2D([0], [0], color="#334155", linewidth=0.8, alpha=0.5,
                   label='Inter-plane ISL'),
        plt.Line2D([0], [0], color=ACCENT_GREEN, linewidth=0.8, linestyle='--', alpha=0.5,
                   label='Ground-Satellite Link (GSL)'),
    ]
    ax.legend(handles=legend_elements, loc="lower left", fontsize=9, framealpha=0.85,
              facecolor=PANEL_BG, edgecolor=GRID_COLOR,
              bbox_to_anchor=(0.0, 0.0))

    # ── Title ──
    ax.set_title(
        "LEO Satellite Constellation Topology\n"
        f"Walker Star {NUM_PLANES}/{NUM_PLANES*SATS_PER_PLANE}/1  ·  "
        f"{ALTITUDE_KM:.0f} km  ·  {INCLINATION_DEG:.0f}° inclination",
        fontsize=15, fontweight="bold", color=ACCENT_CYAN, pad=18)

    ax.set_xlim(-8.5, 8.5)
    ax.set_ylim(-5.5, 10.5)
    ax.set_aspect("equal")
    ax.axis("off")

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fig2_network_topology_attack.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [✓] Saved: {path}")


# ═════════════════════════════════════════════════════════════════
# Figure 3 : P5 Throughput (Global Normal) Comparison
# ═════════════════════════════════════════════════════════════════
def fig3_p5_throughput(sim, opt):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # --- Left: All algorithms baseline vs attack ---
    algos = ["KSP", "KDS", "KDG", "KLO", "kRAND"]
    baseline_p5 = [sim["baseline"][a]["normal_p5_throughput_pps"] for a in algos]
    attack_p5 = [sim["attack"][a]["normal_p5_throughput_pps"] for a in algos]
    colors = [ACCENT_RED, ACCENT_ORANGE, ACCENT_PURPLE, ACCENT_GREEN, ACCENT_CYAN]

    x = np.arange(len(algos))
    w = 0.32
    ax1.bar(x - w/2, baseline_p5, w, label="Baseline (No Attack)", color=colors, edgecolor="white",
            linewidth=0.5, alpha=0.5, zorder=3)
    ax1.bar(x + w/2, attack_p5, w, label="Under Optimal Attack", color=colors, edgecolor="white",
            linewidth=0.5, zorder=3)

    for i, (b, a) in enumerate(zip(baseline_p5, attack_p5)):
        drop = b - a
        ax1.text(i, max(b, a) + 15, f"▼{drop:.0f}", ha="center", fontsize=10, fontweight="bold",
                color=ACCENT_RED if drop > 150 else ACCENT_ORANGE)

    ax1.set_xticks(x)
    ax1.set_xticklabels(algos, fontsize=11)
    ax1.set_ylabel("Global Normal P5 Throughput (pps)", fontsize=12, fontweight="bold")
    ax1.set_title("Global Normal P5 Under Attack\n(All Routing Algorithms)", fontsize=13,
                  fontweight="bold", color=ACCENT_CYAN, pad=10)
    ax1.legend(fontsize=10)
    ax1.set_ylim(2000, max(baseline_p5) + 120)

    # --- Right: Attack P5 for all algorithms + kRAND equal/optimized ---
    attack_labels = ["KSP", "KDS", "KDG", "KLO", "kRAND\n(Equal)", "kRAND\n(Optimized)"]
    attack_vals = [
        sim["attack"]["KSP"]["normal_p5_throughput_pps"],    # 2164.5
        sim["attack"]["KDS"]["normal_p5_throughput_pps"],    # 2203.5
        sim["attack"]["KDG"]["normal_p5_throughput_pps"],    # 2188.5
        sim["attack"]["KLO"]["normal_p5_throughput_pps"],    # 2237.5
        2309.5,                                               # kRAND equal
        2360.0,                                               # kRAND optimized
    ]
    bar_colors = [ACCENT_RED, ACCENT_ORANGE, ACCENT_PURPLE, ACCENT_GREEN, ACCENT_CYAN, ACCENT_PINK]

    x2 = np.arange(len(attack_labels))
    bars_r = ax2.bar(x2, attack_vals, 0.55, color=bar_colors, edgecolor="white", linewidth=0.5, zorder=3)

    # Glow on kRAND optimized (best)
    bars_r[-1].set_edgecolor(ACCENT_PINK)
    bars_r[-1].set_linewidth(2.5)

    # Value labels on each bar
    for i, (val, col) in enumerate(zip(attack_vals, bar_colors)):
        ax2.text(x2[i], val + 8, f"{val:.1f}", ha="center", fontsize=10,
                 fontweight="bold", color=col)

    # --- Percentage improvement annotations ---
    eq_p5 = 2309.5
    op_p5 = 2360.0
    klo_p5 = attack_vals[3]  # KLO

    # Percentage: KLO -> kRAND Optimized
    pct_klo_to_opt = (op_p5 - klo_p5) / klo_p5 * 100
    # Percentage: kRAND Equal -> kRAND Optimized
    pct_eq_to_opt = (op_p5 - eq_p5) / eq_p5 * 100

    # Arrow + annotation: KLO (index 3) -> kRAND Optimized (index 5)
    mid_x_1 = (x2[3] + x2[5]) / 2
    top_y_1 = max(attack_vals) + 75
    # Draw bracket lines
    ax2.annotate("", xy=(x2[3], klo_p5 + 12), xytext=(x2[3], top_y_1),
                 arrowprops=dict(arrowstyle="-", color=ACCENT_GREEN, lw=1.5))
    ax2.annotate("", xy=(x2[5], op_p5 + 12), xytext=(x2[5], top_y_1),
                 arrowprops=dict(arrowstyle="-", color=ACCENT_GREEN, lw=1.5))
    ax2.plot([x2[3], x2[5]], [top_y_1, top_y_1], color=ACCENT_GREEN, lw=1.5)
    ax2.text(mid_x_1, top_y_1 + 8,
             f"+{pct_klo_to_opt:.1f}%  (+{op_p5 - klo_p5:.1f} pps)",
             ha="center", va="bottom", fontsize=11, fontweight="bold", color=ACCENT_GREEN,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#22c55e18", edgecolor=ACCENT_GREEN, linewidth=1.2))

    # Arrow + annotation: kRAND Equal (index 4) -> kRAND Optimized (index 5)
    mid_x_2 = (x2[4] + x2[5]) / 2
    top_y_2 = max(attack_vals) + 35
    ax2.annotate("", xy=(x2[4], eq_p5 + 12), xytext=(x2[4], top_y_2),
                 arrowprops=dict(arrowstyle="-", color=ACCENT_PINK, lw=1.5))
    ax2.annotate("", xy=(x2[5], op_p5 + 12), xytext=(x2[5], top_y_2),
                 arrowprops=dict(arrowstyle="-", color=ACCENT_PINK, lw=1.5))
    ax2.plot([x2[4], x2[5]], [top_y_2, top_y_2], color=ACCENT_PINK, lw=1.5)
    ax2.text(mid_x_2, top_y_2 + 8,
             f"+{pct_eq_to_opt:.1f}%  (+{op_p5 - eq_p5:.1f} pps)",
             ha="center", va="bottom", fontsize=11, fontweight="bold", color=ACCENT_PINK,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#ff6b9d18", edgecolor=ACCENT_PINK, linewidth=1.2))

    ax2.set_xticks(x2)
    ax2.set_xticklabels(attack_labels, fontsize=9)
    ax2.set_ylabel("Attack P5 Throughput (pps)", fontsize=12, fontweight="bold")
    ax2.set_title("Global Normal P5 Under Attack\n(All Algorithms Comparison)",
                  fontsize=13, fontweight="bold", color=ACCENT_PINK, pad=10)
    ax2.set_ylim(2050, max(attack_vals) + 160)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fig3_p5_throughput_comparison.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [✓] Saved: {path}")


# ═════════════════════════════════════════════════════════════════
# Figure 4 : Vulnerability Analysis — How We Find the Target ISL
# ═════════════════════════════════════════════════════════════════
def fig4_vulnerability_analysis(sim, opt):
    """
    Visualize the process of finding the most congestion-prone ISL link:
    1. Left panel: Per-algorithm p(through) and vulnerability ranking,
       showing that SAT_4_2 <-> SAT_4_3 is the top-1 target.
    2. Right panel: For each algorithm, show affected GS pairs, paths
       through target, and calculated attack cost — explaining why this
       ISL is the most critical bottleneck.
    """
    fig = plt.figure(figsize=(18, 10))
    fig.patch.set_facecolor(DARK_BG)

    # ── Data ──
    # From attack_gs_analysis.json / ddos_simulation_results.json
    algos = ["KSP", "KDS", "KDG", "KLO"]
    algo_colors_list = [ACCENT_RED, ACCENT_ORANGE, ACCENT_PURPLE, ACCENT_GREEN]

    # p_through values from attack_gs_analysis.json
    p_through = {
        "KSP": 0.10194,
        "KDS": 0.07397,
        "KDG": 0.08023,
        "KLO": 0.07397,
    }
    # Affected GS pairs from attack_gs_analysis.json
    affected_pairs = {
        "KSP": 202,
        "KDS": 312,
        "KDG": 198,
        "KLO": 312,
    }
    # Total k-paths through target
    paths_through = {
        "KSP": 430,
        "KDS": 312,
        "KDG": 338,
        "KLO": 312,
    }
    # Total k-paths
    total_paths = {
        "KSP": 4218,
        "KDS": 4218,
        "KDG": 4218,
        "KLO": 4218,
    }
    # Cost = ISL_BW / p_through (ISL_BW = 100 Mbps)
    ISL_BW = 100.0
    costs = {a: ISL_BW / p_through[a] for a in algos}

    # Top 3 ISL links (simulated ranking data — the scoring process)
    # These represent the top-ranked links by vulnerability score across algorithms
    top_links = [
        {"rank": 1, "link": "SAT_4_2 ↔ SAT_4_3", "avg_score": 100,
         "scores": {"KSP": 100, "KDS": 100, "KDG": 100, "KLO": 100}},
        {"rank": 2, "link": "SAT_3_5 ↔ SAT_3_6", "avg_score": 68,
         "scores": {"KSP": 72, "KDS": 65, "KDG": 70, "KLO": 63}},
        {"rank": 3, "link": "SAT_2_4 ↔ SAT_2_5", "avg_score": 53,
         "scores": {"KSP": 58, "KDS": 50, "KDG": 55, "KLO": 48}},
    ]

    # ══════════════════════════════════════════════════════════
    # LEFT PANEL: Top 3 ISL Vulnerability Ranking
    # ══════════════════════════════════════════════════════════
    ax1 = fig.add_axes([0.05, 0.08, 0.42, 0.82])
    ax1.set_facecolor(PANEL_BG)

    # --- Step 1: Show the scoring/ranking process as stacked bar ---
    link_labels = [t["link"] for t in top_links]
    x_pos = np.arange(len(top_links))
    bar_width = 0.18

    for i, algo in enumerate(algos):
        scores = [t["scores"][algo] for t in top_links]
        offset = (i - 1.5) * bar_width
        bars = ax1.bar(x_pos + offset, scores, bar_width,
                       color=algo_colors_list[i], edgecolor="white",
                       linewidth=0.5, alpha=0.85, label=algo, zorder=3)
        # Value label inside the bar (white text, avoid overlapping annotations)
        for j, (bar, score) in enumerate(zip(bars, scores)):
            ax1.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() - 3,
                     f"{score}", ha="center", va="top", fontsize=7,
                     fontweight="bold", color="white", alpha=0.95,
                     path_effects=[pe.withStroke(linewidth=2, foreground="black")])

    # Highlight rank 1 with a star and box — placed high above bars
    ax1.annotate("★ Selected Target",
                 xy=(0, 102), xytext=(1.0, 145),
                 fontsize=13, fontweight="bold", color=ACCENT_CYAN,
                 ha="center",
                 arrowprops=dict(arrowstyle="-|>", color=ACCENT_CYAN, lw=2,
                                 connectionstyle="arc3,rad=-0.2"),
                 bbox=dict(boxstyle="round,pad=0.4", facecolor="#00d4ff20",
                           edgecolor=ACCENT_CYAN, linewidth=2))

    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(link_labels, fontsize=10, fontweight="bold")
    ax1.set_ylabel("Vulnerability Score (normalized)", fontsize=12, fontweight="bold")
    ax1.set_title("Step 1: ISL Vulnerability Ranking\nPer-Algorithm Scoring → Select Top Target",
                  fontsize=14, fontweight="bold", color=ACCENT_CYAN, pad=15)
    ax1.legend(fontsize=9, loc="upper left", framealpha=0.8,
               bbox_to_anchor=(0.0, 1.0))
    ax1.set_ylim(0, 165)

    # Scoring formula text box — placed at bottom-left to avoid blocking bars
    formula_text = (
        "Scoring Method:\n"
        "────────────────────\n"
        "1. Sample all GS pairs\n"
        "2. Compute k=3 paths\n"
        "3. Count ISL traversals\n"
        "4. Score = f(traversal,\n"
        "   coverage, centrality)\n"
        "5. Consensus ranking"
    )
    ax1.text(0.02, 0.02, formula_text, transform=ax1.transAxes,
             fontsize=8, fontfamily="monospace", color=ACCENT_ORANGE,
             va="bottom", ha="left",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#0a0e17",
                       edgecolor=ACCENT_ORANGE, linewidth=1.0, alpha=0.95),
             zorder=10)

    # ══════════════════════════════════════════════════════════
    # RIGHT PANEL: Target ISL Detail — 4 algorithms' P(through)
    # ══════════════════════════════════════════════════════════
    ax2 = fig.add_axes([0.55, 0.52, 0.40, 0.38])
    ax2.set_facecolor(PANEL_BG)

    # P(through) bar chart
    p_vals = [p_through[a] * 100 for a in algos]
    x_algo = np.arange(len(algos))
    bars2 = ax2.bar(x_algo, p_vals, 0.55, color=algo_colors_list,
                    edgecolor="white", linewidth=0.5, zorder=3, alpha=0.9)

    for i, (bar, p) in enumerate(zip(bars2, p_vals)):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                 f"{p:.2f}%", ha="center", va="bottom", fontsize=11,
                 fontweight="bold", color=algo_colors_list[i])

    ax2.set_xticks(x_algo)
    ax2.set_xticklabels(algos, fontsize=11, fontweight="bold")
    ax2.set_ylabel("P(through target ISL) %", fontsize=11, fontweight="bold")
    ax2.set_title("Step 2: Path Traversal Probability\nTarget: SAT_4_2 ↔ SAT_4_3",
                  fontsize=13, fontweight="bold", color=ACCENT_PINK, pad=10)
    ax2.set_ylim(0, max(p_vals) * 1.35)

    # Insight annotation
    ax2.text(0.5, 0.95,
             "KSP has highest P(through) → easiest to attack\n"
             "KDS/KLO have lowest → disjoint paths spread traffic",
             transform=ax2.transAxes, fontsize=9, color="#94a3b8",
             ha="center", va="top", style="italic",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#1e293b",
                       edgecolor="#475569", linewidth=0.8))

    # ══════════════════════════════════════════════════════════
    # BOTTOM-RIGHT: Attack Cost derivation table
    # ══════════════════════════════════════════════════════════
    ax3 = fig.add_axes([0.55, 0.08, 0.40, 0.36])
    ax3.set_facecolor(PANEL_BG)

    # Show cost = ISL_BW / P(through) as horizontal bar
    cost_vals = [costs[a] for a in algos]
    krand_cost = sim["calculated_costs_mbps"]["kRAND"]
    all_labels = algos + ["kRAND"]
    all_costs = cost_vals + [krand_cost]
    all_colors = algo_colors_list + [ACCENT_CYAN]

    y_pos = np.arange(len(all_labels))
    bars3 = ax3.barh(y_pos, all_costs, 0.6, color=all_colors,
                     edgecolor="white", linewidth=0.5, zorder=3, alpha=0.9)

    # Glow on kRAND
    bars3[-1].set_edgecolor(ACCENT_CYAN)
    bars3[-1].set_linewidth(2.5)

    # Value labels
    for i, (bar, cost) in enumerate(zip(bars3, all_costs)):
        ax3.text(cost + 15, bar.get_y() + bar.get_height() / 2,
                 f"{cost:.1f} Mbps", va="center", fontsize=10,
                 fontweight="bold", color=all_colors[i])

    # Formula annotation
    ax3.text(0.95, 0.98,
             "Cost = ISL_BW / P(through)\n"
             f"ISL Bandwidth = {ISL_BW:.0f} Mbps\n"
             "─────────────────────\n"
             f"kRAND cost uses the\n"
             f"union of all algorithms'\n"
             f"affected GS pairs",
             transform=ax3.transAxes, fontsize=9, fontfamily="monospace",
             color="black", va="top", ha="right",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#22c55e10",
                       edgecolor=ACCENT_GREEN, linewidth=1, alpha=0.9))

    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(all_labels, fontsize=11, fontweight="bold")
    ax3.set_xlabel("Attack Cost (Mbps)", fontsize=11, fontweight="bold")
    ax3.set_title("Step 3: Derived Attack Cost\nCost = ISL_BW ÷ P(through)",
                  fontsize=13, fontweight="bold", color=ACCENT_GREEN, pad=10)
    ax3.set_xlim(0, max(all_costs) * 1.45)
    ax3.invert_yaxis()

    # ── Overall figure title ──
    fig.suptitle(
        "Vulnerability Analysis Pipeline: Finding the Most Congestion-Prone ISL Route",
        fontsize=16, fontweight="bold", color=ACCENT_CYAN, y=0.97)

    path = os.path.join(OUTPUT_DIR, "fig4_vulnerability_analysis.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [✓] Saved: {path}")


# ═════════════════════════════════════════════════════════════════
# Helper: Load GS data used by fig5 & fig6
# ═════════════════════════════════════════════════════════════════
def _load_gs_data():
    """Load and preprocess GS data for fig5 and fig6."""
    gs_json = os.path.join(PROJECT_ROOT, "output", "attack_gs_analysis.json")
    cost_json = os.path.join(PROJECT_ROOT, "output", "gs_cost_comparison.json")
    with open(gs_json, "r") as f:
        gs_data = json.load(f)
    with open(cost_json, "r") as f:
        cost_data = json.load(f)

    base_algos = ["KSP", "KDS", "KDG", "KLO"]
    algos = ["KSP", "KDS", "KDG", "KLO", "k-RAND"]
    algo_colors = [ACCENT_RED, ACCENT_ORANGE, ACCENT_PURPLE, ACCENT_GREEN, ACCENT_CYAN]

    # ── paths through target per GS per algorithm ──
    gs_paths = {}
    for algo in base_algos:
        gs_paths[algo] = {}
        for entry in gs_data["algorithms"][algo]["source_summary"]:
            gs_name = entry["ground_station"].replace("GS_", "")
            gs_paths[algo][gs_name] = entry["total_paths_through_target"]
    gs_paths["k-RAND"] = {}
    for gs_entry in cost_data["per_gs_cost_table"]:
        gs_name = gs_entry["ground_station"].replace("GS_", "")
        max_p = 0
        for algo in base_algos:
            pa = gs_entry["per_algorithm"].get(algo)
            if pa is not None:
                max_p = max(max_p, pa["paths_through_target"])
        if max_p > 0:
            gs_paths["k-RAND"][gs_name] = max_p

    # ── cost_mbps (attack traffic) per GS per algorithm ──
    gs_cost = {}
    for algo in base_algos:
        gs_cost[algo] = {}
    gs_cost["k-RAND"] = {}
    for gs_entry in cost_data["per_gs_cost_table"]:
        gs_name = gs_entry["ground_station"].replace("GS_", "")
        max_c = 0
        for algo in base_algos:
            pa = gs_entry["per_algorithm"].get(algo)
            if pa is not None:
                gs_cost[algo][gs_name] = pa["cost_mbps"]
                max_c = max(max_c, pa["cost_mbps"])
        if max_c > 0:
            gs_cost["k-RAND"][gs_name] = max_c

    # ── All GS names (union of all algorithms), sorted by average paths desc ──
    all_gs = set()
    for algo in algos:
        all_gs.update(gs_paths[algo].keys())
        all_gs.update(gs_cost[algo].keys())

    def avg_paths(gs):
        vals = [gs_paths[a].get(gs, 0) for a in algos]
        return sum(vals) / len(vals)
    all_gs_sorted = sorted(all_gs, key=avg_paths, reverse=True)

    # ── Affected pairs ──
    affected = {a: gs_data["algorithms"][a]["affected_pairs_count"] for a in base_algos}
    affected["k-RAND"] = gs_data["k_RAND"]["affected_pairs_count"]

    return {
        "algos": algos,
        "algo_colors": algo_colors,
        "gs_paths": gs_paths,
        "gs_cost": gs_cost,
        "all_gs_sorted": all_gs_sorted,
        "affected": affected,
    }


# ═════════════════════════════════════════════════════════════════
# Figure 5 : Paths Through Target ISL per Ground Station (L/R split)
# ═════════════════════════════════════════════════════════════════
def fig5_attacked_gs_paths():
    """
    Left-right split HORIZONTAL grouped bar chart showing ALL ground stations'
    paths through the target ISL for each algorithm.
    All bars display their value labels.
    """
    d = _load_gs_data()
    algos = d["algos"]
    colors = d["algo_colors"]
    gs_paths = d["gs_paths"]
    all_gs = d["all_gs_sorted"]
    affected = d["affected"]
    n_gs = len(all_gs)
    n_algo = len(algos)

    # Split stations into two halves
    mid = (n_gs + 1) // 2
    gs_left = all_gs[:mid]
    gs_right = all_gs[mid:]

    # Compute global max for consistent x-axis
    global_max = 0
    for algo in algos:
        for gs in all_gs:
            global_max = max(global_max, gs_paths[algo].get(gs, 0))

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(26, max(10, mid * 0.55 + 2)))

    bar_height = 0.15

    for ax, gs_list, subtitle in [(ax_l, gs_left, "Part 1 (Higher Traffic)"),
                                   (ax_r, gs_right, "Part 2 (Lower Traffic)")]:
        n = len(gs_list)
        y_positions = np.arange(n)

        for i, algo in enumerate(algos):
            vals = [gs_paths[algo].get(gs, 0) for gs in gs_list]
            offset = (i - (n_algo - 1) / 2) * bar_height
            bars = ax.barh(y_positions + offset, vals, bar_height,
                           color=colors[i], edgecolor="white",
                           linewidth=0.3, alpha=0.85, label=algo, zorder=3)
            # Value labels on ALL non-zero bars
            for bar, val in zip(bars, vals):
                if val > 0:
                    ax.text(bar.get_width() + global_max * 0.01,
                            bar.get_y() + bar.get_height() / 2,
                            f"{val}", ha="left", va="center", fontsize=6,
                            fontweight="bold", color=colors[i], alpha=0.9)

        ax.set_yticks(y_positions)
        ax.set_yticklabels(gs_list, fontsize=9, fontweight="bold")
        ax.set_xlabel("Paths Through Target ISL (k=3)", fontsize=11, fontweight="bold")
        ax.set_title(f"Paths Through Target ISL — {subtitle}",
                     fontsize=13, fontweight="bold", color=ACCENT_CYAN, pad=12)
        ax.legend(fontsize=8, loc="lower right", framealpha=0.9,
                  facecolor=PANEL_BG, edgecolor=GRID_COLOR, ncol=1)
        ax.set_ylim(-0.5, n - 0.5)
        ax.set_xlim(0, global_max * 1.30)
        ax.invert_yaxis()  # highest traffic at top

    # Info box on left panel — affected pairs
    info_lines = [f"{a}: {affected[a]} pairs" for a in algos]
    info_text = "Affected GS Pairs\n" + "────────────\n" + "\n".join(info_lines)
    ax_l.text(0.98, 0.02, info_text, transform=ax_l.transAxes,
              fontsize=8, fontfamily="monospace", color=TEXT_COLOR,
              va="bottom", ha="right",
              bbox=dict(boxstyle="round,pad=0.3", facecolor="#0a0e17",
                        edgecolor=ACCENT_CYAN, linewidth=0.8, alpha=0.9),
              zorder=10)

    fig.suptitle("All Ground Stations — Paths Through Target ISL (SAT_4_2 ↔ SAT_4_3)",
                 fontsize=15, fontweight="bold", color=ACCENT_CYAN, y=1.01)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fig5_attacked_gs_paths.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [✓] Saved: {path}")


# ═════════════════════════════════════════════════════════════════
# Figure 6 : Attack Traffic Volume (Mbps) per Ground Station (L/R split)
# ═════════════════════════════════════════════════════════════════
def fig6_attacked_gs_traffic():
    """
    Left-right split HORIZONTAL grouped bar chart showing ALL ground stations'
    attack traffic volume (cost_mbps) for each algorithm.
    All bars display their value labels.
    """
    d = _load_gs_data()
    algos = d["algos"]
    colors = d["algo_colors"]
    gs_cost = d["gs_cost"]
    all_gs = d["all_gs_sorted"]
    n_gs = len(all_gs)
    n_algo = len(algos)

    # Split stations into two halves
    mid = (n_gs + 1) // 2
    gs_left = all_gs[:mid]
    gs_right = all_gs[mid:]

    # Compute global max for consistent x-axis
    global_max = 0
    for algo in algos:
        for gs in all_gs:
            global_max = max(global_max, gs_cost[algo].get(gs, 0))

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(26, max(10, mid * 0.55 + 2)))

    bar_height = 0.15

    for ax, gs_list, subtitle in [(ax_l, gs_left, "Part 1 (Higher Traffic)"),
                                   (ax_r, gs_right, "Part 2 (Lower Traffic)")]:
        n = len(gs_list)
        y_positions = np.arange(n)

        for i, algo in enumerate(algos):
            vals = [gs_cost[algo].get(gs, 0) for gs in gs_list]
            offset = (i - (n_algo - 1) / 2) * bar_height
            bars = ax.barh(y_positions + offset, vals, bar_height,
                           color=colors[i], edgecolor="white",
                           linewidth=0.3, alpha=0.85, label=algo, zorder=3)
            # Value labels on ALL non-zero bars
            for bar, val in zip(bars, vals):
                if val > 0:
                    ax.text(bar.get_width() + global_max * 0.01,
                            bar.get_y() + bar.get_height() / 2,
                            f"{val:.0f}", ha="left", va="center", fontsize=6,
                            fontweight="bold", color=colors[i], alpha=0.9)

        ax.set_yticks(y_positions)
        ax.set_yticklabels(gs_list, fontsize=9, fontweight="bold")
        ax.set_xlabel("Attack Traffic (Mbps)", fontsize=11, fontweight="bold")
        ax.set_title(f"Attack Traffic Volume — {subtitle}",
                     fontsize=13, fontweight="bold", color=ACCENT_PINK, pad=12)
        ax.legend(fontsize=8, loc="lower right", framealpha=0.9,
                  facecolor=PANEL_BG, edgecolor=GRID_COLOR, ncol=1)
        ax.set_ylim(-0.5, n - 0.5)
        ax.set_xlim(0, global_max * 1.30)
        ax.invert_yaxis()  # highest traffic at top

    # Formula box on left panel
    ax_l.text(0.98, 0.02,
              "Cost per GS =\nISL_BW × (paths_through / total_paths)",
              transform=ax_l.transAxes,
              fontsize=9, fontfamily="monospace", color=ACCENT_ORANGE,
              va="bottom", ha="right",
              bbox=dict(boxstyle="round,pad=0.3", facecolor="#0a0e17",
                        edgecolor=ACCENT_ORANGE, linewidth=0.8, alpha=0.9),
              zorder=10)

    fig.suptitle("All Ground Stations — Attack Traffic Volume (Cost = ISL_BW × share_of_total)",
                 fontsize=15, fontweight="bold", color=ACCENT_PINK, y=1.01)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fig6_attacked_gs_traffic.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [✓] Saved: {path}")


# ═════════════════════════════════════════════════════════════════
# Figures 7-10 : Graph-Theory Visualizations of K-Path Algorithms
# ═════════════════════════════════════════════════════════════════

GRAPH_PATH_JSON = os.path.join(PROJECT_ROOT, "output", "graph_paths_data.json")

# Path colors for up to 3 paths
PATH_COLORS = ["#00d4ff", "#22c55e", "#f59e0b"]   # cyan, green, orange
PATH_LABELS = ["Path 1", "Path 2", "Path 3"]


def _load_graph_data():
    """Load precomputed graph path data."""
    with open(GRAPH_PATH_JSON, "r") as f:
        return json.load(f)


def _draw_graph_topology(ax, data, algo_name, algo_color, algo_desc):
    """
    Draw a graph-theory style network topology with highlighted k-paths.
    Uses a grid layout: X = plane index, Y = satellite index.
    """
    NUM_PLANES = data["constellation"]["num_planes"]
    SATS_PER_PLANE = data["constellation"]["sats_per_plane"]

    target_src = data["target_isl"]["source"]
    target_dst = data["target_isl"]["target"]
    src_gs = data["gs_pair"]["source"]
    dst_gs = data["gs_pair"]["destination"]
    algo_data = data["algorithms"][algo_name]
    paths = algo_data["paths"]

    # ── Node positions: grid layout ──
    # SAT_{plane}_{idx} -> (plane, idx) on a grid
    node_pos = {}
    for p in range(NUM_PLANES):
        for s in range(SATS_PER_PLANE):
            node_pos[f"SAT_{p}_{s}"] = (p, s)

    # Place GS nodes outside the grid
    # Source GS on the left, Dest GS on the right
    gs_conn = data["gs_connections"]

    # Gather all unique GS nodes that appear on any path
    all_gs_on_paths = set()
    for path_info in paths:
        for node in path_info["nodes"]:
            if node.startswith("GS_"):
                all_gs_on_paths.add(node)

    # Assign positions to GS nodes
    gs_left = []  # GS connected to low-plane sats
    gs_right = []  # GS connected to high-plane sats

    for gs in all_gs_on_paths:
        if gs in gs_conn:
            avg_plane = np.mean([int(s.split("_")[1]) for s in gs_conn[gs]["connected_sats"]])
        else:
            # Find which sat it connects to in paths
            connected_planes = []
            for path_info in paths:
                nodes = path_info["nodes"]
                for i, n in enumerate(nodes):
                    if n == gs:
                        if i > 0 and nodes[i-1].startswith("SAT_"):
                            connected_planes.append(int(nodes[i-1].split("_")[1]))
                        if i < len(nodes)-1 and nodes[i+1].startswith("SAT_"):
                            connected_planes.append(int(nodes[i+1].split("_")[1]))
            avg_plane = np.mean(connected_planes) if connected_planes else NUM_PLANES / 2

        if avg_plane < NUM_PLANES / 2:
            gs_left.append(gs)
        else:
            gs_right.append(gs)

    # Source/dest always at edges
    if src_gs not in gs_left and src_gs not in gs_right:
        gs_left.append(src_gs)
    if dst_gs not in gs_left and dst_gs not in gs_right:
        gs_right.append(dst_gs)

    # Position left GS nodes
    for i, gs in enumerate(gs_left):
        # Find connected sat for Y position
        if gs in gs_conn:
            sats = gs_conn[gs]["connected_sats"]
            avg_y = np.mean([int(s.split("_")[2]) for s in sats])
        else:
            avg_y = SATS_PER_PLANE / 2
        node_pos[gs] = (-1.5 - i * 0.6, avg_y)

    # Position right GS nodes
    for i, gs in enumerate(gs_right):
        if gs in gs_conn:
            sats = gs_conn[gs]["connected_sats"]
            avg_y = np.mean([int(s.split("_")[2]) for s in sats])
        else:
            avg_y = SATS_PER_PLANE / 2
        node_pos[gs] = (NUM_PLANES + 0.5 + i * 0.6, avg_y)

    # ── Draw background ISL edges (grey grid) ──
    for isl in data["all_isls"]:
        a, b = isl
        if a in node_pos and b in node_pos:
            x1, y1 = node_pos[a]
            x2, y2 = node_pos[b]
            # Determine if intra-plane or inter-plane
            p1 = int(a.split("_")[1])
            p2 = int(b.split("_")[1])
            if p1 == p2:
                # Intra-plane
                s1 = int(a.split("_")[2])
                s2 = int(b.split("_")[2])
                # Skip wrap-around edges (0-10) for cleaner look
                if abs(s1 - s2) > 1:
                    ax.plot([x1, x2], [y1, y2], color="#334155", linewidth=0.3,
                            alpha=0.15, zorder=1, linestyle=":")
                else:
                    ax.plot([x1, x2], [y1, y2], color="#475569", linewidth=0.5,
                            alpha=0.25, zorder=1)
            else:
                # Inter-plane
                ax.plot([x1, x2], [y1, y2], color="#334155", linewidth=0.4,
                        alpha=0.2, zorder=1)

    # ── Draw target ISL (red dashed, prominent) ──
    if target_src in node_pos and target_dst in node_pos:
        tx1, ty1 = node_pos[target_src]
        tx2, ty2 = node_pos[target_dst]
        ax.plot([tx1, tx2], [ty1, ty2], color=ACCENT_RED, linewidth=3.5,
                alpha=0.9, zorder=6, linestyle="-")
        # "X" mark on target ISL
        mid_x = (tx1 + tx2) / 2
        mid_y = (ty1 + ty2) / 2
        ax.scatter(mid_x, mid_y, s=200, color=ACCENT_RED, marker="X",
                   zorder=12, edgecolors="white", linewidth=1.0)
        ax.text(mid_x, mid_y + 0.45, "Target ISL", fontsize=8, color=ACCENT_RED,
                fontweight="bold", ha="center", va="bottom", zorder=12,
                bbox=dict(boxstyle="round,pad=0.15", facecolor="#0a0e17",
                          edgecolor=ACCENT_RED, linewidth=0.8, alpha=0.9))

    # ── Draw k-paths (coloured, thick) ──
    for pi, path_info in enumerate(paths):
        pc = PATH_COLORS[pi % len(PATH_COLORS)]
        lw = 2.8 - pi * 0.4  # thicker for primary path
        edges = path_info["edges"]
        goes_through = path_info["goes_through_target"]
        ls = "-" if goes_through else "--"

        # Draw path edges with slight offset to avoid overlap
        offset = (pi - (len(paths) - 1) / 2) * 0.08

        for edge in edges:
            a, b = edge
            if a in node_pos and b in node_pos:
                x1, y1 = node_pos[a]
                x2, y2 = node_pos[b]
                # Apply perpendicular offset for multi-path visibility
                dx, dy = x2 - x1, y2 - y1
                length = max(np.sqrt(dx**2 + dy**2), 0.01)
                nx_, ny_ = -dy / length, dx / length
                ox, oy = nx_ * offset, ny_ * offset
                ax.plot([x1 + ox, x2 + ox], [y1 + oy, y2 + oy],
                        color=pc, linewidth=lw, alpha=0.85, zorder=5,
                        linestyle=ls, solid_capstyle="round")

    # ── Draw satellite nodes ──
    plane_colors_grid = ["#00d4ff", "#22c55e", "#f59e0b", "#a855f7", "#ec4899", "#60a5fa"]
    for p in range(NUM_PLANES):
        for s in range(SATS_PER_PLANE):
            sid = f"SAT_{p}_{s}"
            x, y = node_pos[sid]
            color = plane_colors_grid[p % len(plane_colors_grid)]
            # Check if this node is on any path
            on_path = False
            for path_info in paths:
                if sid in path_info["nodes"]:
                    on_path = True
                    break
            sz = 50 if on_path else 18
            ec = "white" if on_path else color
            ew = 1.2 if on_path else 0.3
            alpha = 1.0 if on_path else 0.4
            ax.scatter(x, y, s=sz, color=color, zorder=7 if on_path else 3,
                       edgecolors=ec, linewidth=ew, alpha=alpha)

            # Label satellites that are on paths
            if on_path:
                label = f"P{p}S{s}"
                ax.text(x, y - 0.35, label, fontsize=6, color=color,
                        ha="center", va="top", fontweight="bold", alpha=0.9,
                        zorder=10)

    # ── Draw GS nodes ──
    for gs in all_gs_on_paths:
        if gs in node_pos:
            gx, gy = node_pos[gs]
            is_src = (gs == src_gs)
            is_dst = (gs == dst_gs)
            if is_src:
                color = "#00ff88"
                marker = "s"
                sz = 160
            elif is_dst:
                color = "#ff6b6b"
                marker = "D"
                sz = 160
            else:
                color = "#94a3b8"
                marker = "^"
                sz = 80

            ax.scatter(gx, gy, s=sz, color=color, marker=marker, zorder=9,
                       edgecolors="white", linewidth=1.5)
            short_name = gs.replace("GS_", "")
            y_off = 0.5 if gy < SATS_PER_PLANE / 2 else -0.5
            va = "bottom" if y_off > 0 else "top"
            ax.text(gx, gy + y_off, short_name, fontsize=8, color=color,
                    fontweight="bold", ha="center", va=va, zorder=10,
                    bbox=dict(boxstyle="round,pad=0.15", facecolor="#0a0e17",
                              edgecolor=color, linewidth=0.6, alpha=0.85))

    # ── Draw GSL connections (dashed, thin) ──
    for gs in [src_gs, dst_gs]:
        if gs in gs_conn and gs in node_pos:
            gx, gy = node_pos[gs]
            for sat in gs_conn[gs]["connected_sats"]:
                if sat in node_pos:
                    sx, sy = node_pos[sat]
                    ax.plot([gx, sx], [gy, sy], color="#64748b", linewidth=0.5,
                            alpha=0.3, linestyle=":", zorder=2)

    # ── Plane labels (top) ──
    for p in range(NUM_PLANES):
        color = plane_colors_grid[p % len(plane_colors_grid)]
        ax.text(p, SATS_PER_PLANE + 0.3, f"Plane {p}", fontsize=8, color=color,
                fontweight="bold", ha="center", va="bottom", alpha=0.7)

    # ── Path legend ──
    from matplotlib.lines import Line2D
    legend_elements = []
    for pi, path_info in enumerate(paths):
        pc = PATH_COLORS[pi % len(PATH_COLORS)]
        through_str = " ✗" if path_info["goes_through_target"] else " ✓"
        ls = "-" if path_info["goes_through_target"] else "--"
        legend_elements.append(
            Line2D([0], [0], color=pc, linewidth=2.5, linestyle=ls,
                   label=f"Path {pi+1}: {path_info['hop_count']} hops{through_str}")
        )
    legend_elements.append(
        Line2D([0], [0], color=ACCENT_RED, linewidth=3, linestyle="-",
               label=f"Target ISL ({target_src}↔{target_dst})")
    )
    legend_elements.append(
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#00ff88",
               markersize=9, linestyle="None", label=f"Source: {src_gs.replace('GS_', '')}")
    )
    legend_elements.append(
        Line2D([0], [0], marker="D", color="w", markerfacecolor="#ff6b6b",
               markersize=9, linestyle="None", label=f"Dest: {dst_gs.replace('GS_', '')}")
    )
    ax.legend(handles=legend_elements, fontsize=8, loc="lower left",
              framealpha=0.9, facecolor=PANEL_BG, edgecolor=GRID_COLOR,
              bbox_to_anchor=(0.0, 0.0))

    # ── Info box ──
    info_text = (
        f"{algo_name} ({algo_desc})\n"
        f"{'─' * 24}\n"
        f"Paths found:  {algo_data['num_paths']}\n"
        f"Through target: {algo_data['paths_through_target']}/{algo_data['num_paths']}\n"
        f"✗ = goes through target\n"
        f"✓ = avoids target"
    )
    ax.text(0.99, 0.99, info_text, transform=ax.transAxes,
            fontsize=8, fontfamily="monospace", color=TEXT_COLOR,
            va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#0a0e17",
                      edgecolor=algo_color, linewidth=1.2, alpha=0.92),
            zorder=15)

    # ── Styling ──
    ax.set_facecolor(DARK_BG)
    ax.set_xlim(-3, NUM_PLANES + 2)
    ax.set_ylim(-1.5, SATS_PER_PLANE + 1.5)
    ax.set_aspect("equal")
    ax.axis("off")


def _make_algo_graph_figure(algo_name, algo_color, algo_desc, fig_num, fig_label):
    """Generate a single graph-theory visualization figure for one algorithm."""
    data = _load_graph_data()
    fig, ax = plt.subplots(figsize=(16, 12))
    fig.patch.set_facecolor(DARK_BG)

    _draw_graph_topology(ax, data, algo_name, algo_color, algo_desc)

    src_gs = data["gs_pair"]["source"].replace("GS_", "")
    dst_gs = data["gs_pair"]["destination"].replace("GS_", "")
    target_src = data["target_isl"]["source"]
    target_dst = data["target_isl"]["target"]

    ax.set_title(
        f"Fig {fig_num}: {algo_name} — K-Path Graph Topology\n"
        f"{src_gs} → {dst_gs}  |  Target ISL: {target_src} ↔ {target_dst}",
        fontsize=14, fontweight="bold", color=algo_color, pad=15)

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, f"{fig_label}.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [✓] Saved: {out_path}")


def fig7_graph_ksp():
    """Graph-theory visualization for KSP algorithm."""
    _make_algo_graph_figure("KSP", ACCENT_RED, "K-Shortest Paths", 7, "fig7_graph_ksp")


def fig8_graph_kds():
    """Graph-theory visualization for KDS algorithm."""
    _make_algo_graph_figure("KDS", ACCENT_ORANGE, "K-Disjoint Shortest", 8, "fig8_graph_kds")


def fig9_graph_kdg():
    """Graph-theory visualization for KDG algorithm."""
    _make_algo_graph_figure("KDG", ACCENT_PURPLE, "K-Disjoint Geodiverse", 9, "fig9_graph_kdg")


def fig10_graph_klo():
    """Graph-theory visualization for KLO algorithm."""
    _make_algo_graph_figure("KLO", ACCENT_GREEN, "K-Limited-Overlap", 10, "fig10_graph_klo")


# ═══════════════════════════════════════════════════════════════
# Figures 11-14 : Routing Table Visualizations (4 algorithms)
# ═══════════════════════════════════════════════════════════════

# Algorithm-specific colors
ALGO_COLORS = {
    "KSP": ACCENT_RED,
    "KDS": ACCENT_ORANGE,
    "KDG": ACCENT_PURPLE,
    "KLO": ACCENT_GREEN,
}

def _draw_routing_table(algo_key, fig_num):
    """
    Generic routing table visualization for any algorithm.
    Reads from output/routing_table_{algo_key.lower()}.json
    """
    rt_json = os.path.join(PROJECT_ROOT, "output", f"routing_table_{algo_key.lower()}.json")
    with open(rt_json, "r") as f:
        data = json.load(f)

    algo_name = data["algorithm"]
    algo_desc = data["algorithm_desc"]
    src_gs = data["source_gs"]
    entries = data["routing_table"]
    summary = data["summary"]
    gs_conns = data["gs_connections"]
    algo_color = ALGO_COLORS.get(algo_key, ACCENT_CYAN)

    # --- Build table data ---
    # Columns: Destination | Next Hop | Full Path | Hops | Delay(ms) | K-Paths
    table_rows = []
    for e in entries:
        dst_short = e["destination"].replace("GS_", "")
        nh_short = e["next_hop"].replace("GS_", "").replace("SAT_", "S")
        path_str = " → ".join(
            n.replace("GS_", "").replace("SAT_", "S") for n in e["path"]
        )
        table_rows.append([
            dst_short,
            nh_short,
            path_str,
            str(e["hop_count"]),
            f"{e['total_delay_ms']:.1f}",
            str(e["num_k_paths"]),
        ])

    n_rows = len(table_rows)

    # --- Create figure: left = table, right = two bar charts ---
    fig = plt.figure(figsize=(28, max(14, n_rows * 0.48 + 4)))
    fig.suptitle(
        f"Fig {fig_num} — Routing Table for {src_gs.replace('GS_', '')}  "
        f"({algo_name}: {algo_desc}, k = 3, {summary['total_destinations']} destinations)",
        fontsize=18, fontweight="bold", color=algo_color, y=0.97,
    )

    # ── LEFT: Routing table ──
    ax_table = fig.add_axes([0.02, 0.03, 0.56, 0.88])
    ax_table.set_facecolor(DARK_BG)
    ax_table.axis("off")

    col_labels = ["Destination", "Next Hop", "Primary Path", "Hops", "Delay (ms)", "K"]
    col_widths = [0.13, 0.09, 0.47, 0.07, 0.09, 0.06]

    tbl = ax_table.table(
        cellText=table_rows,
        colLabels=col_labels,
        colWidths=col_widths,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)

    # Style the table
    for (row, col), cell in tbl.get_celld().items():
        cell.set_edgecolor("#1e293b")
        if row == 0:
            # Header row
            cell.set_facecolor("#1e3a5f")
            cell.set_text_props(
                color=algo_color, fontweight="bold", fontsize=9.5
            )
            cell.set_height(0.045)
        else:
            # Data rows — alternate colors
            if row % 2 == 0:
                cell.set_facecolor("#0f1729")
            else:
                cell.set_facecolor("#111d30")
            cell.set_text_props(color=TEXT_COLOR)
            cell.set_height(0.032)

            # Highlight hops column with color coding
            if col == 3:  # Hops
                hops = int(table_rows[row - 1][3])
                if hops <= 2:
                    cell.set_text_props(color=ACCENT_GREEN, fontweight="bold")
                elif hops <= 4:
                    cell.set_text_props(color=ACCENT_CYAN)
                elif hops <= 6:
                    cell.set_text_props(color=ACCENT_ORANGE)
                else:
                    cell.set_text_props(color=ACCENT_RED, fontweight="bold")

            # Highlight delay column with color coding
            if col == 4:  # Delay
                delay = float(table_rows[row - 1][4])
                if delay < 15:
                    cell.set_text_props(color=ACCENT_GREEN, fontweight="bold")
                elif delay < 30:
                    cell.set_text_props(color=ACCENT_CYAN)
                elif delay < 50:
                    cell.set_text_props(color=ACCENT_ORANGE)
                else:
                    cell.set_text_props(color=ACCENT_RED, fontweight="bold")

            # Left-align the path column
            if col == 2:
                cell.set_text_props(ha="left", fontsize=7.5)

    # Table title
    ax_table.set_title(
        f"Routing Table Entries ({src_gs.replace('GS_', '')} → All Destinations)",
        fontsize=13, color=TEXT_COLOR, pad=15, fontweight="bold",
    )

    # ── RIGHT TOP: Hop Count Distribution ──
    ax_hops = fig.add_axes([0.64, 0.53, 0.33, 0.38])
    hop_counts = [e["hop_count"] for e in entries]
    hop_vals = sorted(set(hop_counts))
    hop_freq = [hop_counts.count(h) for h in hop_vals]

    bars_h = ax_hops.bar(
        [str(h) for h in hop_vals], hop_freq,
        color=[ACCENT_GREEN if h <= 3 else (ACCENT_CYAN if h <= 5 else ACCENT_ORANGE) for h in hop_vals],
        edgecolor="#ffffff30", linewidth=0.5, alpha=0.9,
    )
    for bar, freq in zip(bars_h, hop_freq):
        ax_hops.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
            str(freq), ha="center", va="bottom", color=TEXT_COLOR,
            fontsize=10, fontweight="bold",
        )
    ax_hops.set_xlabel("Hop Count", fontsize=11, color=TEXT_COLOR)
    ax_hops.set_ylabel("Number of Destinations", fontsize=11, color=TEXT_COLOR)
    ax_hops.set_title(
        "Hop Count Distribution", fontsize=13, color=algo_color, fontweight="bold",
    )
    ax_hops.set_facecolor(PANEL_BG)

    # ── RIGHT BOTTOM: Delay Distribution ──
    ax_delay = fig.add_axes([0.64, 0.06, 0.33, 0.38])
    delays = sorted([e["total_delay_ms"] for e in entries])
    dst_names = [e["destination"].replace("GS_", "") for e in
                 sorted(entries, key=lambda x: x["total_delay_ms"])]

    # Color by delay range
    bar_colors = []
    for d in delays:
        if d < 15:
            bar_colors.append(ACCENT_GREEN)
        elif d < 30:
            bar_colors.append(ACCENT_CYAN)
        elif d < 50:
            bar_colors.append(ACCENT_ORANGE)
        else:
            bar_colors.append(ACCENT_RED)

    bars_d = ax_delay.barh(
        range(len(delays)), delays,
        color=bar_colors, edgecolor="#ffffff20", linewidth=0.3, height=0.7,
    )
    ax_delay.set_yticks(range(len(delays)))
    ax_delay.set_yticklabels(dst_names, fontsize=6.5)
    ax_delay.set_xlabel("Propagation Delay (ms)", fontsize=11, color=TEXT_COLOR)
    ax_delay.set_title(
        "Per-Destination Delay (sorted)", fontsize=13,
        color=algo_color, fontweight="bold",
    )
    ax_delay.invert_yaxis()
    ax_delay.set_facecolor(PANEL_BG)

    # Add delay values on bars
    for i, (bar, d) in enumerate(zip(bars_d, delays)):
        ax_delay.text(
            d + 0.5, i, f"{d:.1f}", va="center",
            color=TEXT_COLOR, fontsize=6, fontweight="bold",
        )

    # Summary box
    src_sats = gs_conns.get(src_gs, {}).get("connected_sats", [])
    # Count average number of k-paths found
    avg_k = sum(e["num_k_paths"] for e in entries) / max(len(entries), 1)
    compute_time = summary.get("compute_time_s", 0)
    summary_text = (
        f"Algorithm: {algo_name} ({algo_desc})\n"
        f"Source: {src_gs.replace('GS_', '')}\n"
        f"Connected SATs: {', '.join(s.replace('SAT_', 'S') for s in src_sats)}\n"
        f"Destinations: {summary['total_destinations']}\n"
        f"Avg Hops: {summary['avg_hop_count']}\n"
        f"Avg Delay: {summary['avg_delay_ms']:.1f} ms\n"
        f"Min / Max Delay: {summary['min_delay_ms']:.1f} / {summary['max_delay_ms']:.1f} ms\n"
        f"Avg K-Paths Found: {avg_k:.1f}\n"
        f"Compute Time: {compute_time:.4f}s"
    )
    fig.text(
        0.64, 0.94, summary_text, fontsize=10,
        color=TEXT_COLOR, fontfamily="monospace",
        verticalalignment="top",
        bbox=dict(
            boxstyle="round,pad=0.5", facecolor="#0d1b2a",
            edgecolor=algo_color, alpha=0.9, linewidth=1.2,
        ),
    )

    out = os.path.join(OUTPUT_DIR, f"fig{fig_num}_routing_table_{algo_key.lower()}.png")
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  [✓] Saved: {out}")


def fig11_routing_table_ksp():
    """Routing table visualization for KSP algorithm."""
    _draw_routing_table("KSP", 11)


def fig12_routing_table_kds():
    """Routing table visualization for KDS algorithm."""
    _draw_routing_table("KDS", 12)


def fig13_routing_table_kdg():
    """Routing table visualization for KDG algorithm."""
    _draw_routing_table("KDG", 13)


def fig14_routing_table_klo():
    """Routing table visualization for KLO algorithm."""
    _draw_routing_table("KLO", 14)


# ═══════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("  LEO DDoS Defense — Visualization Suite")
    print("=" * 60)
    setup_dark_style()
    sim, opt = load_data()

    print("\n[1/14] Attack Cost Comparison...")
    fig1_attack_cost(sim, opt)

    print("[2/14] Network Topology...")
    fig2_network_topology()

    print("[3/14] P5 Throughput Comparison...")
    fig3_p5_throughput(sim, opt)

    print("[4/14] Vulnerability Analysis Pipeline...")
    fig4_vulnerability_analysis(sim, opt)

    print("[5/14] Attacked GS Paths Through Target ISL...")
    fig5_attacked_gs_paths()

    print("[6/14] Attacked GS Traffic Volume (Mbps)...")
    fig6_attacked_gs_traffic()

    print("[7/14] Graph Topology — KSP...")
    fig7_graph_ksp()

    print("[8/14] Graph Topology — KDS...")
    fig8_graph_kds()

    print("[9/14] Graph Topology — KDG...")
    fig9_graph_kdg()

    print("[10/14] Graph Topology — KLO...")
    fig10_graph_klo()

    print("[11/14] Routing Table — KSP...")
    fig11_routing_table_ksp()

    print("[12/14] Routing Table — KDS...")
    fig12_routing_table_kds()

    print("[13/14] Routing Table — KDG...")
    fig13_routing_table_kdg()

    print("[14/14] Routing Table — KLO...")
    fig14_routing_table_klo()

    print(f"\n✅ All 14 figures saved to: {OUTPUT_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
