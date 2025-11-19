#!/usr/bin/env python3
"""
compare_schedule_to_baseline.py

Post-processes `grid_demand_summary.csv` and compares the BEST schedule
(at each horizon) to the SECOND-BEST schedule (used as baseline).

It reports, per horizon:

- Extra PRODUCTION HOURS and % vs second-best
- Extra MAGNUMS/UNITS and % vs second-best

This matches the requirement: baseline = 2nd best schedule.

Outputs
-------

- Human-readable summary to stdout
- CSV file `schedule_gain_vs_baseline.csv` with per-horizon metrics
"""

import csv
from collections import defaultdict
from typing import Dict, List, Tuple


SummaryRow = Dict[str, str]


SUMMARY_CSV_PATH = "grid_demand_summary.csv"
OUTPUT_CSV_PATH = "schedule_gain_vs_baseline.csv"


def read_summary(summary_path: str) -> List[SummaryRow]:
    rows: List[SummaryRow] = []
    with open(summary_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def group_by_horizon(rows: List[SummaryRow]) -> Dict[float, List[SummaryRow]]:
    grouped: Dict[float, List[SummaryRow]] = defaultdict(list)
    for r in rows:
        h = float(r["time_horizon_h"])
        grouped[h].append(r)
    return grouped


def pick_best_and_second_best(rows: List[SummaryRow]) -> Tuple[SummaryRow, SummaryRow]:
    """
    Given all rows for a single horizon, return:

    - best_row: row with max total_magnums
    - baseline_row: row with SECOND-HIGHEST total_magnums
      (if there is only one row, best == baseline)
    """
    def units(r: SummaryRow) -> float:
        return float(r["total_magnums"])

    rows_sorted = sorted(rows, key=units, reverse=True)
    best_row = rows_sorted[0]
    baseline_row = rows_sorted[1] if len(rows_sorted) > 1 else best_row
    return best_row, baseline_row


def horizon_weeks(time_horizon_h: float) -> float:
    # 1 week = 7 * 24 = 168 hours
    return time_horizon_h / 168.0


def compute_gain(best: SummaryRow, base: SummaryRow) -> Dict[str, float]:
    prod_best = float(best["Production"])
    prod_base = float(base["Production"])
    units_best = float(best["total_magnums"])
    units_base = float(base["total_magnums"])

    prod_diff = prod_best - prod_base
    units_diff = units_best - units_base

    prod_pct = (prod_diff / prod_base * 100.0) if prod_base > 0 else 0.0
    units_pct = (units_diff / units_base * 100.0) if units_base > 0 else 0.0

    return {
        "prod_best": prod_best,
        "prod_base": prod_base,
        "prod_diff": prod_diff,
        "prod_pct": prod_pct,
        "units_best": units_best,
        "units_base": units_base,
        "units_diff": units_diff,
        "units_pct": units_pct,
    }


def main() -> None:
    rows = read_summary(SUMMARY_CSV_PATH)
    grouped = group_by_horizon(rows)

    # Prepare CSV output
    out_fields = [
        "time_horizon_h",
        "horizon_weeks",
        "best_schedule_id",
        "best_sequence",
        "baseline_schedule_id",
        "baseline_sequence",
        "prod_hours_baseline",
        "prod_hours_best",
        "prod_hours_gain",
        "prod_hours_gain_pct",
        "magnums_baseline",
        "magnums_best",
        "magnums_gain",
        "magnums_gain_pct",
    ]

    out_rows = []

    print("\n=== Gain of best schedule vs SECOND-BEST schedule (per horizon) ===\n")

    for time_h in sorted(grouped.keys()):
        rows_h = grouped[time_h]
        best, baseline = pick_best_and_second_best(rows_h)
        gain = compute_gain(best, baseline)

        w = horizon_weeks(time_h)
        best_id = int(best["schedule_id"])
        base_id = int(baseline["schedule_id"])
        best_seq = best["sequence"]
        base_seq = baseline["sequence"]

        print(f"Horizon: {time_h:.0f} h  (~{w:.1f} weeks)")
        print(f"  Best schedule:       ID {best_id}, seq = {best_seq}")
        print(f"  Baseline (2nd best): ID {base_id}, seq = {base_seq}")
        print(
            f"  Production hours: "
            f"{gain['prod_base']:.1f} → {gain['prod_best']:.1f}  "
            f"(+{gain['prod_diff']:.1f} h, {gain['prod_pct']:.1f}%)"
        )
        print(
            f"  Magnums/units:    "
            f"{gain['units_base']:.0f} → {gain['units_best']:.0f}  "
            f"(+{gain['units_diff']:.0f}, {gain['units_pct']:.1f}%)"
        )
        print()

        out_rows.append({
            "time_horizon_h": f"{time_h:.0f}",
            "horizon_weeks": f"{w:.3f}",
            "best_schedule_id": str(best_id),
            "best_sequence": best_seq,
            "baseline_schedule_id": str(base_id),
            "baseline_sequence": base_seq,
            "prod_hours_baseline": f"{gain['prod_base']:.3f}",
            "prod_hours_best": f"{gain['prod_best']:.3f}",
            "prod_hours_gain": f"{gain['prod_diff']:.3f}",
            "prod_hours_gain_pct": f"{gain['prod_pct']:.3f}",
            "magnums_baseline": f"{gain['units_base']:.3f}",
            "magnums_best": f"{gain['units_best']:.3f}",
            "magnums_gain": f"{gain['units_diff']:.3f}",
            "magnums_gain_pct": f"{gain['units_pct']:.3f}",
        })

    with open(OUTPUT_CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=out_fields)
        writer.writeheader()
        for r in out_rows:
            writer.writerow(r)

    print(f"Summary CSV written to: {OUTPUT_CSV_PATH}\n")


if __name__ == "__main__":
    main()
