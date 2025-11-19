#!/usr/bin/env python3
"""
visualise_grid_results_exec.py

Exec-friendly visualisations:

  - normalised_total_bars_top5_all_horizons.png
      -> absolute total Magnums/units (was 'bars'), with labels and % vs best.
  - time_breakdown_states_top5_all_horizons.png
      -> absolute hours by STATE BUCKET (not normalised).
  - relative_gain_top5_all_horizons.png
      -> single bar per horizon: best vs average and best vs worst (Magnums).
  - gantt_top5_horizon_{H}h.png
      -> Gantt with grouped TASK BUCKETS (production / changeovers /
         wrapper changes / daily / weekly / fortnightly / pre-post / other).

Top 5 schedules only per horizon.
"""

import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.patches import Patch

SUMMARY_CSV = "grid_demand_summary.csv"
EVENTS_CSV = "grid_demand_events.csv"
OUTPUT_DIR = "plots"

TOP_K = 5  # top schedules per horizon

# ---------------------------------------------------------------------
# State buckets (from summary CSV columns)
# ---------------------------------------------------------------------
STATE_BUCKETS = {
    "Production": ["Production"],
    "Changeover": ["Changeover"],
    "Packaging": ["Packaging"],
    "Routine Cleaning": ["RoutineSan", "ResourceCleaning"],
    "Deep Cleaning": ["WeeklySan", "FortnightlySan"],
    "Other": ["PreProduction", "PostProduction", "Other"],
}

BUCKET_COLORS = {
    "Production": "#1f77b4",
    "Changeover": "#ff7f0e",
    "Packaging": "#9467bd",
    "Routine Cleaning": "#8c564b",
    "Deep Cleaning": "#e377c2",
    "Other": "#7f7f7f",
}

# ---------------------------------------------------------------------
# Task GROUP buckets for Gantt chart
# ---------------------------------------------------------------------
TASK_GROUP_COLORS = {
    "Production": "#1f77b4",
    "Changeover cleaning (A–R)": "#ff7f0e",
    "Wrapper change": "#2ca02c",
    "Daily sanitisation": "#8c564b",
    "Weekly cleaning (N2 + AM/PM)": "#9467bd",
    "Fortnightly cleaning (tunnel + N2 + sauce)": "#e377c2",
    "Pre / post production": "#7f7f7f",
    "Other": "#17becf",
}

TASK_GROUP_ORDER = [
    "Production",
    "Changeover cleaning (A–R)",
    "Wrapper change",
    "Daily sanitisation",
    "Weekly cleaning (N2 + AM/PM)",
    "Fortnightly cleaning (tunnel + N2 + sauce)",
    "Pre / post production",
    "Other",
]

SHORT_GROUP_LABELS = {
    "Production": None,                     # we’ll still use SKU codes for long runs
    "Changeover cleaning (A–R)": "CO",      # Changeover
    "Wrapper change": "WP",                 # Wrapper change
    "Daily sanitisation": "DS",             # Daily sanitisation
    "Weekly cleaning (N2 + AM/PM)": "WC",   # Weekly cleaning
    "Fortnightly cleaning (tunnel + N2 + sauce)": "FC",  # Fortnightly cleaning
    "Pre / post production": "PP",          # Pre/Post production
    "Other": "OT",                          # Other
}

# thresholds for inline text (in hours)
PROD_ANNOT_THRESHOLD = 24.0      # label SKU if production run ≥ 1 day
NONPROD_ANNOT_THRESHOLD = 4.0    # label cleaning blocks if ≥ 4h

# small visual gap between steps (in hours)
STEP_GAP_H = 0.25


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def ensure_output_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path)


def load_data():
    summary = pd.read_csv(SUMMARY_CSV)
    events = pd.read_csv(EVENTS_CSV)
    return summary, events


def get_total_col(df: pd.DataFrame) -> str:
    """
    Decide which total output column to use:
    - Prefer 'total_magnums' (new naming)
    - Fall back to 'total_bars' for backward compatibility
    """
    if "total_magnums" in df.columns:
        return "total_magnums"
    if "total_bars" in df.columns:
        return "total_bars"
    raise ValueError(
        "Summary CSV must contain either 'total_magnums' or 'total_bars'."
    )


def add_schedule_ranks(summary: pd.DataFrame) -> pd.DataFrame:
    summary = summary.copy()
    if "time_horizon_h" not in summary.columns:
        raise ValueError("Missing 'time_horizon_h' column in summary CSV.")

    total_col = get_total_col(summary)

    summary["schedule_rank"] = (
        summary.groupby("time_horizon_h")[total_col]
        .rank(method="first", ascending=False)
        .astype(int)
    )
    return summary


def bucketize_state_row(row: pd.Series) -> pd.Series:
    bucket_vals = {b: 0.0 for b in STATE_BUCKETS}
    for bucket, cols in STATE_BUCKETS.items():
        for c in cols:
            if c in row:
                bucket_vals[bucket] += float(row[c])
    return pd.Series(bucket_vals)


# -------------------------- TASK GROUPING -----------------------------
def map_task_to_group(task: str) -> str:
    """
    Map detailed event `task` text to high-level groups, based on:
      - cleaning codes A–R (change-over cleaning)
      - daily / weekly / fortnightly cleans
      - wrapper change
      - pre/post production
      - production
    """
    t = task.lower().strip()

    # 1) Production
    if t.startswith("produce "):
        return "Production"

    # 2) Wrapper change (size + wrapper)
    if t.startswith("wrapper/volume change"):
        return "Wrapper change"

    # 3) Product changeovers A–R
    if t.startswith("change-over cleaning"):
        return "Changeover cleaning (A–R)"

    # 4) Daily routine sanitisation
    if t.startswith("routine sanitisation"):
        return "Daily sanitisation"

    # 5) Weekly block = Nitrogen bath cleaning + AM/PM cleaning
    if t.startswith("nitrogen bath cleaning") or t.startswith("am/pm cleaning"):
        return "Weekly cleaning (N2 + AM/PM)"

    # 6) Fortnightly block
    if t.startswith("tunnel & nitrogen & sauce tank washing"):
        return "Fortnightly cleaning (tunnel + N2 + sauce)"

    # 7) Pre / post production sanitisation and procedures
    if (
        t.startswith("planned sanitisation before start")
        or t.startswith("production start procedures")
        or t.startswith("planned sanitisation after end of production")
        or t.startswith("end of production procedures")
    ):
        return "Pre / post production"

    # 8) Fallback
    return "Other"


# ---------------------------------------------------------------------
# Plot 1: Total Magnums (absolute)
# ---------------------------------------------------------------------
def plot_total_bars_top5(summary: pd.DataFrame):
    total_col = get_total_col(summary)

    horizons = sorted(summary["time_horizon_h"].unique())
    n = len(horizons)
    cols = min(2, n)
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 7 * rows), squeeze=False)
    axes_flat = axes.flatten()

    for ax, horizon in zip(axes_flat, horizons):
        grp = (
            summary[summary["time_horizon_h"] == horizon]
            .copy()
            .sort_values("schedule_rank")
            .head(TOP_K)
        )

        if grp.empty:
            ax.axis("off")
            continue

        x = np.arange(len(grp))
        totals = grp[total_col].values
        best = totals.max()

        bars = ax.bar(x, totals, color="#1f77b4")

        # -------------------------------
        # Add padding above tallest bar
        # -------------------------------
        ax.set_ylim(top=best * 1.12)

        ax.set_title(f"Horizon = {int(horizon / 168)} weeks", fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels([f"#{r}" for r in grp["schedule_rank"]], fontsize=8)
        ax.set_ylabel("Total Magnums (millions)")
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v/1e6:.2f}M"))
        ax.grid(axis="y", alpha=0.3)

        # Labels on each bar
        for bar, sched_id, total in zip(bars, grp["schedule_id"], totals):
            pct_vs_best = (total / best - 1) * 100 if best > 0 else 0
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"S{int(sched_id)}\n{total/1e6:.2f}M\n{pct_vs_best:+.1f}%",
                ha="center",
                va="bottom",
                fontsize=7,
            )

    for ax in axes_flat[len(horizons):]:
        ax.axis("off")

    fig.suptitle("Total Magnums – Top 5 Schedules per Horizon", fontsize=14)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fname = os.path.join(OUTPUT_DIR, "normalised_total_bars_top5_all_horizons.png")
    fig.savefig(fname, dpi=200)
    plt.close(fig)
    print("[saved]", fname)


# ---------------------------------------------------------------------
# Plot 2: State breakdown (absolute hours)
# ---------------------------------------------------------------------
def plot_state_breakdown_top5(summary: pd.DataFrame):
    horizons = sorted(summary["time_horizon_h"].unique())
    n = len(horizons)
    cols = min(2, n)
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 7 * rows), squeeze=False)
    axes_flat = axes.flatten()

    state_names = list(STATE_BUCKETS.keys())
    num_buckets = len(state_names)

    for ax, horizon in zip(axes_flat, horizons):
        grp = (
            summary[summary["time_horizon_h"] == horizon]
            .copy()
            .sort_values("schedule_rank")
            .head(TOP_K)
        )

        if grp.empty:
            ax.axis("off")
            continue

        # Hours per state bucket
        bucket_df = grp.apply(bucketize_state_row, axis=1)

        # Total time per schedule
        total_time = bucket_df.sum(axis=1).replace(0, np.nan)

        # Normalised to % of total time
        bucket_pct = bucket_df.div(total_time, axis=0) * 100.0

        x = np.arange(len(grp))  # one group per schedule
        bar_width = 0.12
        offset = (num_buckets - 1) / 2.0

        production_positions = None
        production_values = None

        # Plot grouped bars: one bar per state bucket per schedule
        for b_idx, bucket in enumerate(state_names):
            if bucket not in bucket_pct.columns:
                continue

            vals = bucket_pct[bucket].values
            positions = x + (b_idx - offset) * bar_width

            ax.bar(
                positions,
                vals,
                width=bar_width,
                label=bucket,
                color=BUCKET_COLORS.get(bucket, "#7f7f7f"),
                alpha=0.9 if bucket == "Production" else 0.7,
            )

            # Remember production bars for annotation
            if bucket == "Production":
                production_positions = positions
                production_values = vals

        ax.set_title(f"Horizon = {int(horizon / 168)} weeks (Top 5)", fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [f"#{r}\nS{int(s)}" for r, s in zip(grp["schedule_rank"], grp["schedule_id"])],
            fontsize=7,
        )
        ax.set_ylabel("Share of time [%]")
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:.1f}"))
        ax.grid(axis="y", alpha=0.3)

        # Headroom so annotations don't clip
        ymax = bucket_pct.max().max() * 1.12
        ax.set_ylim(top=ymax)

        # Annotate production bars with % and Δ vs best (if we have them)
        if production_positions is not None and production_values is not None:
            best_prod_pct = np.nanmax(production_values)
            for pos, pct in zip(production_positions, production_values):
                delta_pp = pct - best_prod_pct  # best = 0, others negative
                ax.text(
                    pos,
                    pct,
                    f"{pct:.1f}%\n{delta_pp:+.1f} pp",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )

    # Hide unused axes
    for ax in axes_flat[len(horizons):]:
        ax.axis("off")

    # Legend (state buckets)
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", title="State bucket", fontsize=8)

    fig.suptitle(
        "Time Breakdown by State (% of total time) – Top 5 Schedules per Horizon",
        fontsize=14,
    )
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    fname = os.path.join(OUTPUT_DIR, "time_breakdown_states_top5_all_horizons.png")
    fig.savefig(fname, dpi=200)
    plt.close(fig)
    print("[saved]", fname)



# ---------------------------------------------------------------------
# Plot 3: Relative gain summary (Magnums)
# ---------------------------------------------------------------------
def plot_relative_gain_summary(summary: pd.DataFrame):
    total_col = get_total_col(summary)

    rows = []
    for horizon, grp in summary.groupby("time_horizon_h"):
        grp = grp.copy()

        # Sort schedules by performance (Magnums)
        grp_sorted = grp.sort_values(total_col, ascending=False)

        best = grp_sorted[total_col].iloc[0]
        # "Next best" = 2nd schedule, or same as best if only one schedule exists
        next_best = grp_sorted[total_col].iloc[1] if len(grp_sorted) > 1 else best

        worst = grp_sorted[total_col].min()
        avg = grp_sorted[total_col].mean()

        gain_vs_next = (best - next_best) / next_best * 100 if next_best > 0 else 0
        gain_vs_avg = (best - avg) / avg * 100 if avg > 0 else 0
        gain_vs_worst = (best - worst) / worst * 100 if worst > 0 else 0

        rows.append(
            {
                "time_horizon_h": horizon,
                "gain_vs_next_pct": gain_vs_next,
                "gain_vs_avg_pct": gain_vs_avg,
                "gain_vs_worst_pct": gain_vs_worst,
            }
        )

    df = pd.DataFrame(rows).sort_values("time_horizon_h")

    if df.empty:
        return

    x = np.arange(len(df))
    width = 0.25  # thinner bars now that we have 3 per group

    fig, ax = plt.subplots(figsize=(10, 5))

    bars_next = ax.bar(
        x - width,
        df["gain_vs_next_pct"],
        width,
        label="Best vs 2nd best",
    )
    bars_avg = ax.bar(
        x,
        df["gain_vs_avg_pct"],
        width,
        label="Best vs Avg",
    )
    bars_worst = ax.bar(
        x + width,
        df["gain_vs_worst_pct"],
        width,
        label="Best vs Worst",
    )

    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(h / 168)} weeks" for h in df["time_horizon_h"]])
    ax.set_ylabel("Gain [%]")
    ax.set_title("Relative Gain of Best Schedule (Magnums) per Horizon")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:.2f}"))
    ax.grid(axis="y", alpha=0.3)
    ax.legend()

    # Annotate all three bar groups
    for bars in (bars_next, bars_avg, bars_worst):
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h,
                f"{h:.1f}%",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    fig.tight_layout()
    fname = os.path.join(OUTPUT_DIR, "relative_gain_top5_all_horizons.png")
    fig.savefig(fname, dpi=200)
    plt.close(fig)
    print("[saved]", fname)


SKU_COLORS = {
    "B3":  "#1f77b4",
    "B4":  "#ff7f0e",
    "Bm":  "#2ca02c",
    "CAB": "#9467bd",
    "S3":  "#8c564b",
    "S4":  "#e377c2",
    "Sm":  "#7f7f7f",
    "C":   "#bcbd22",
    "Ch3": "#17becf",
    "Ch4": "#aec7e8",
    "Hm":  "#ffbb78",
}

# ---------------------------------------------------------------------
# Plot 4: Gantt – grouped task buckets, with spacing & inline labels
# ---------------------------------------------------------------------
def plot_gantt_top5(summary: pd.DataFrame, events: pd.DataFrame):
    """
    Same as before, but:
      - Production bars colored per SKU (using SKU_COLORS)
      - Other task groups keep existing colors
      - No inline annotations
      - Y-labels unchanged
    """

    total_col = get_total_col(summary)
    horizons = sorted(summary["time_horizon_h"].unique())

    for horizon in horizons:

        grp = (
            summary[summary["time_horizon_h"] == horizon]
            .copy()
            .sort_values("schedule_rank")
            .head(TOP_K)
        )
        if grp.empty:
            continue

        sched_ids = grp["schedule_id"].astype(int).tolist()

        rank_map = {int(r["schedule_id"]): int(r["schedule_rank"]) for _, r in grp.iterrows()}
        total_units_map = {int(r["schedule_id"]): float(r[total_col]) for _, r in grp.iterrows()}

        ev = events[
            (events["time_horizon_h"] == horizon) &
            (events["schedule_id"].isin(sched_ids))
        ].copy()

        if ev.empty:
            continue

        ev = ev.sort_values(["schedule_id", "start_h"])
        max_time = float(ev["end_h"].max())

        bucket_to_y = {b: i for i, b in enumerate(TASK_GROUP_ORDER)}
        y_positions = list(bucket_to_y.values())
        y_labels = list(bucket_to_y.keys())

        n_schedules = len(sched_ids)
        fig_height = max(6, 3 + n_schedules * 3.5)

        fig, axes = plt.subplots(n_schedules, 1, figsize=(26, fig_height), sharex=True)
        if n_schedules == 1:
            axes = [axes]

        for ax, sid in zip(axes, sched_ids):

            ev_s = ev[ev["schedule_id"] == sid].sort_values("start_h")

            for _, r in ev_s.iterrows():
                start = float(r["start_h"])
                duration = float(r["duration_h"])
                task = r["task"]

                bucket = map_task_to_group(task)
                if bucket not in bucket_to_y:
                    continue

                y = bucket_to_y[bucket]

                # ----------- PRODUCTION COLORING -----------
                if bucket == "Production" and task.lower().startswith("produce "):
                    sku = task.split(" ", 1)[1].strip()
                    color = SKU_COLORS.get(sku, "#000000")  # fallback black if missing
                else:
                    color = TASK_GROUP_COLORS[bucket]
                # -------------------------------------------

                ax.barh(
                    y,
                    duration,
                    left=start,
                    height=0.8,
                    color=color,
                    edgecolor="none"
                )

            ax.set_yticks(y_positions)
            ax.set_yticklabels(y_labels, fontsize=10)
            ax.grid(axis="x", alpha=0.35)

            rank = rank_map.get(sid, "?")
            tu = total_units_map.get(sid, 0.0)

            # Extract SKU sequence for this schedule
            ev_prod = ev_s[ev_s["task"].str.lower().str.startswith("produce ")]
            sku_sequence = [
                t.split(" ", 1)[1].strip()
                for t in ev_prod["task"].tolist()
            ]
            sku_sequence_str = "→".join(sku_sequence[:12]) if sku_sequence else "|"

            # Updated title including production sequence
            ax.set_title(
                f"Schedule {sid} – Rank #{rank} – Total Magnums: {tu:,.0f}\n"
                f"Sequence: {sku_sequence_str}",
                fontsize=12,
                loc="left"
            )

        # X-axis ticks converted to weeks
        step = 7 * 24
        xticks = np.arange(0, max_time + step, step)
        xlabels = [int(x / 168) for x in xticks]

        axes[-1].set_xticks(xticks, labels=xlabels)
        axes[-1].set_xlabel("Time [weeks]")

        for lbl in axes[-1].get_xticklabels():
            lbl.set_rotation(90)
            lbl.set_ha("right")

        # Legend remains by task group (not SKU)
        # --- Global legends ---

        # 1) Task Group Legend (existing)
        handles_task = [
            Patch(facecolor=TASK_GROUP_COLORS[b], label=b)
            for b in TASK_GROUP_ORDER
        ]

        task_legend = fig.legend(
            handles_task,
            [h.get_label() for h in handles_task],
            loc="upper left",
            bbox_to_anchor=(0.80, 0.98),
            fontsize=10,
            title="Task Groups",
            frameon=True,
        )

        # 2) SKU legend for production colors (new)
        handles_sku = [
            Patch(facecolor=col, label=sku)
            for sku, col in SKU_COLORS.items()
        ]

        sku_legend = fig.legend(
            handles_sku,
            [h.get_label() for h in handles_sku],
            loc="upper left",
            bbox_to_anchor=(0.80, 0.55),
            fontsize=9,
            title="SKU Production Colors",
            frameon=True,
        )

        # Ensure both legends are drawn
        fig.add_artist(task_legend)
        fig.add_artist(sku_legend)

        fig.suptitle(
            f"Gantt Timelines – Top {TOP_K} Schedules (Horizon {int(horizon/168)} weeks)",
            fontsize=18
        )

        fig.tight_layout(rect=[0.02, 0.02, 0.78, 0.94])

        out_name = os.path.join(OUTPUT_DIR, f"gantt_top5_horizon_{int(horizon)}h.png")
        fig.savefig(out_name, dpi=250)
        plt.close(fig)

        print("[saved]", out_name)



# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    ensure_output_dir(OUTPUT_DIR)
    summary, events = load_data()
    summary = add_schedule_ranks(summary)

    plot_total_bars_top5(summary)
    plot_state_breakdown_top5(summary)
    plot_relative_gain_summary(summary)
    plot_gantt_top5(summary, events)

    print("\nAll plots generated in:", os.path.abspath(OUTPUT_DIR))


if __name__ == "__main__":
    main()
