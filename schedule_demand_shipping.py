#!/usr/bin/env python3
"""
icecream_schedule_extended_inventory_timelimited.py

Inventory-aware + HARD time horizon

- Weekly demand per SKU (from 6-week capacity).
- For each horizon (in weeks), we compute a time limit:
      time_limit_h = weeks * 7 * 24
- We simulate week-by-week, trying to produce weekly_demand for each SKU.
- BUT: we NEVER start an event (production or cleaning) that would
  end beyond time_limit_h. We simply stop scheduling at that point.

Outputs:

1) grid_demand_summary.csv
2) grid_demand_events.csv

Columns as before, but now:
  - total_hours <= time_horizon_h
  - scale_of_demand <= 1 reflects how much of the planned demand could
    actually be produced within the given time horizon.
"""

from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any, Iterable

from ortools.constraint_solver import pywrapcp, routing_enums_pb2


# ------------------------------
# Data structures
# ------------------------------

@dataclass
class SKU:
    code: str
    name: str
    allergens: List[str]
    volume_ml: int  # 85 (standard) or 55 (mini)


@dataclass
class Resource:
    name: str
    max_run_time: float       # hours of continuous use before cleaning
    clean_time: float         # hours to clean
    max_storage_time: float = float('inf')  # hours idle before rinse/flush


# ------------------------------
# SKU definitions & matrices
# ------------------------------

def build_skus() -> Dict[str, SKU]:
    skus: Dict[str, SKU] = {
        'B3':  SKU('B3',  'Magnum Billionaire Standard 3MP',                 ['M','G','J','D','S'], 85),
        'B4':  SKU('B4',  'Magnum Billionaire Standard 4MP',                 ['M','G','J','D','S'], 85),
        'Bm':  SKU('Bm',  'Magnum Billionaire Mini 6MP',                     ['M','G','J','D','S'], 55),
        'CAB': SKU('CAB', 'Magnum Double Caramel Almond & Billionaire Mini 6MP', ['M','G','J','D','S','Z'], 55),
        'S3':  SKU('S3',  'Magnum Double Starchaser 3MP',                    ['M','G'], 85),
        'S4':  SKU('S4',  'Magnum Double Starchaser 4MP',                    ['M','G'], 85),
        'Sm':  SKU('Sm',  'Magnum Double Starchaser Mini 6MP',               ['M','G'], 55),
        'C':   SKU('C',   'Magnum Double Caramel 4MP',                       ['M','G'], 85),
        'Ch3': SKU('Ch3', 'Magnum Double Cherry 3MP',                        ['M','G'], 85),
        'Ch4': SKU('Ch4', 'Magnum Double Cherry 4MP',                        ['M','G'], 85),
        'Hm':  SKU('Hm',  'Magnum Double Hazelnut Mini 6MP',                 ['M','D','Se'], 55),
    }
    return skus


def build_changeover_and_tasks() -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, str]]]:
    skus = ['B3','B4','Bm','CAB','S3','S4','Sm','C','Ch3','Ch4','Hm']
    times: Dict[str, Dict[str, float]] = {k: {} for k in skus}
    tasks: Dict[str, Dict[str, str]] = {k: {} for k in skus}

    def row(from_sku: str, entries: List[Tuple[str, float, str]]) -> None:
        for to_sku, t, code in entries:
            times[from_sku][to_sku] = t
            tasks[from_sku][to_sku] = code

    row('B3', [('B3',1,'-'),('B4',2,'-'),('Bm',2,'-'),('CAB',10,'B*'),
               ('S3',13,'C*'),('S4',13,'C*'),('Sm',13,'C*'),
               ('C',15,'D*'),('Ch3',14,'E*'),('Ch4',14,'E*'),('Hm',14,'E*')])
    row('B4', [('B3',2,'-'),('B4',1,'-'),('Bm',2,'-'),('CAB',10,'B*'),
               ('S3',13,'C*'),('S4',13,'C*'),('Sm',13,'C*'),
               ('C',15,'D*'),('Ch3',14,'E*'),('Ch4',14,'E*'),('Hm',14,'E*')])
    row('Bm', [('B3',6,'A'),('B4',6,'A'),('Bm',1,'-'),('CAB',10,'B*'),
               ('S3',13,'C*'),('S4',13,'C*'),('Sm',13,'C*'),
               ('C',15,'D*'),('Ch3',14,'E*'),('Ch4',14,'E*'),('Hm',14,'E*')])
    row('CAB',[('B3',14,'G'),('B4',14,'G'),('Bm',14,'G'),('CAB',1,'-'),
               ('S3',14,'G'),('S4',14,'G'),('Sm',14,'G'),
               ('C',16,'H'),('Ch3',15,'I'),('Ch4',15,'I'),('Hm',16,'J')])
    row('S3', [('B3',10,'B'),('B4',10,'B'),('Bm',10,'B'),('CAB',10,'B'),
               ('S3',1,'-'),('S4',2,'-'),('Sm',2,'-'),
               ('C',13,'C'),('Ch3',14,'E'),('Ch4',14,'E'),('Hm',15,'F')])
    row('S4', [('B3',10,'B'),('B4',10,'B'),('Bm',10,'B'),('CAB',10,'B'),
               ('S3',2,'-'),('S4',1,'-'),('Sm',2,'-'),
               ('C',13,'C'),('Ch3',14,'E'),('Ch4',14,'E'),('Hm',15,'F')])
    row('Sm', [('B3',10,'B'),('B4',10,'B'),('Bm',10,'B'),('CAB',10,'B'),
               ('S3',6,'A'),('S4',6,'A'),('Sm',1,'-'),
               ('C',13,'C'),('Ch3',14,'E'),('Ch4',14,'E'),('Hm',15,'F')])
    row('C',  [('B3',11,'K'),('B4',11,'K'),('Bm',11,'K'),('CAB',11,'K'),
               ('S3',8,'L'),('S4',8,'L'),('Sm',8,'L'),
               ('C',1,'-'),('Ch3',16,'M'),('Ch4',16,'M'),('Hm',17,'N')])
    row('Ch3',[('B3',14,'E'),('B4',14,'E'),('Bm',14,'E'),('CAB',14,'E'),
               ('S3',14,'E'),('S4',14,'E'),('Sm',14,'E'),
               ('C',16,'M'),('Ch3',1,'-'),('Ch4',2,'-'),('Hm',15,'F')])
    row('Ch4',[('B3',14,'E'),('B4',14,'E'),('Bm',14,'E'),('CAB',14,'E'),
               ('S3',14,'E'),('S4',14,'E'),('Sm',14,'E'),
               ('C',16,'M'),('Ch3',2,'-'),('Ch4',1,'-'),('Hm',15,'F')])
    row('Hm', [('B3',16,'O'),('B4',16,'O'),('Bm',16,'O'),('CAB',17,'P'),
               ('S3',17,'P'),('S4',17,'P'),('Sm',17,'P'),
               ('C',18,'R'),('Ch3',17,'P'),('Ch4',17,'P'),('Hm',1,'-')])
    return times, tasks


def build_tasks_description() -> Dict[str, str]:
    return {
        'A': 'Trays cleaning; wash freezers & extruders (Mini→Double)',
        'B': 'Trays cleaning; wash freezers, extruders, GMPs & chocolate bath',
        'C': 'Trays cleaning; + slat conveyor, GPMs, N2 bath, chocolate, sauce, P&P robots, GMW',
        'D': 'As C + sauce tank, cover bath & tank M2',
        'E': 'As C + sauce tank',
        'F': 'As E + chocolate tank M1',
        'G': 'As C + allergen inspection',
        'H': 'As G + cover bath & tank M1',
        'I': 'As G + sauce tank',
        'J': 'As I + chocolate tank M1',
        'K': 'Trays cleaning; freezers, extruders, GPMs, chocolate & cover bath, cover tank M2',
        'L': 'Trays cleaning; freezers, extruders, GPMs, cover bath, cover tank M2',
        'M': 'Full wash including cover tank M2',
        'N': 'Full wash including chocolate tank M1 and cover tank M2',
        'O': 'Tunnel wash + allergen inspection (incl. tunnel)',
        'P': 'Tunnel wash + choc tank M1 + allergen inspection',
        'R': 'Tunnel wash + choc tank M1 + cover tank M2 + allergen inspection',
    }


def build_packaging_change_matrix(skus: Dict[str, SKU]) -> Dict[str, Dict[str, float]]:
    volumes: Dict[str, int] = {code: sku.volume_ml for code, sku in skus.items()}
    change_time: Dict[str, Dict[str, float]] = {a: {} for a in skus}
    for a in skus:
        for b in skus:
            if a == b:
                change_time[a][b] = 0.0
            else:
                v_from, v_to = volumes[a], volumes[b]
                if v_from == v_to:
                    change_time[a][b] = 0.0
                elif v_from == 85 and v_to == 55:
                    change_time[a][b] = 2.0 + 1.0  # 2h size + 1h wrapper
                elif v_from == 55 and v_to == 85:
                    change_time[a][b] = 6.0 + 1.0  # 6h size + 1h wrapper
                else:
                    change_time[a][b] = 0.0
    return change_time


# ------------------------------
# Rates & demand from capacity
# ------------------------------

def build_sku_rate_map(skus: Dict[str, SKU]) -> Dict[str, float]:
    """
    Per-SKU rate (bars/hour) from 720 bars/min baseline (43,200 bars/h) and OE by pack:
      - 3MP → OE 0.80
      - 4MP → OE 0.85
      - 6MP → OE 0.85
    """
    nominal_bars_per_hour = 720.0 * 60.0  # 43,200
    rate_map: Dict[str, float] = {}
    for code, sku in skus.items():
        nm = sku.name
        if "6MP" in nm:
            oe = 0.85
        elif "3MP" in nm:
            oe = 0.80
        elif "4MP" in nm:
            oe = 0.85
        else:
            oe = 0.85
        rate_map[code] = nominal_bars_per_hour * oe
    return rate_map


def build_6week_demand_from_capacity(
    rate_map: Dict[str, float],
    horizon_weeks: float = 6.0,
    utilisation: Optional[Dict[str, float]] = None
) -> Dict[str, float]:
    """
    demand[sku] = rate_map[sku] * horizon_hours * utilisation_share[sku]

    If utilisation is None, split time equally across SKUs.
    This is the *nominal* 6-week demand profile (capacity-based),
    not yet weekly.
    """
    horizon_hours = horizon_weeks * 7.0 * 24.0
    keys = list(rate_map.keys())
    if utilisation is None:
        share = 1.0 / len(keys) if keys else 0.0
        utilisation = {k: share for k in keys}
    else:
        s = sum(utilisation.values()) or 1.0
        utilisation = {k: utilisation.get(k, 0.0) / s for k in keys}
    return {k: rate_map[k] * horizon_hours * utilisation[k] for k in keys}


# ------------------------------
# Costs & solver (sequence optimisation)
# ------------------------------

def compute_processing_times_by_rate(demand: Dict[str, float],
                                     rate_map: Dict[str, float]) -> Dict[str, float]:
    return {sku: demand[sku] / rate_map[sku] for sku in demand}


def build_cost_matrix(skus: List[str], base_times: Dict[str, Dict[str, float]],
                      tasks: Dict[str, Dict[str, str]],
                      packaging_change: Dict[str, Dict[str, float]],
                      processing_time: Dict[str, float]) -> List[List[float]]:
    n = len(skus)
    cost: List[List[float]] = [[0.0] * n for _ in range(n)]
    for i, a in enumerate(skus):
        for j, b in enumerate(skus):
            base = base_times[a][b]
            code = tasks[a][b]
            extra = 1.0 if code.endswith('*') else 0.0
            pack = packaging_change[a][b]
            pt = processing_time[b]
            cost[i][j] = base + extra + pack + pt
    return cost


def solve_schedule(cost_matrix: List[List[float]]) -> Optional[List[int]]:
    n = len(cost_matrix)
    manager = pywrapcp.RoutingIndexManager(n, 1, 0)
    routing = pywrapcp.RoutingModel(manager)

    def transit_cb(from_index: int, to_index: int) -> int:
        fi = manager.IndexToNode(from_index)
        ti = manager.IndexToNode(to_index)
        return int(cost_matrix[fi][ti] * 1000)

    tidx = routing.RegisterTransitCallback(transit_cb)
    routing.SetArcCostEvaluatorOfAllVehicles(tidx)

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    params.time_limit.seconds = 100
    params.log_search = False

    sol = routing.SolveWithParameters(params)
    if sol is None:
        return None
    idx = routing.Start(0)
    order: List[int] = []
    while not routing.IsEnd(idx):
        node = manager.IndexToNode(idx)
        order.append(node)
        idx = sol.Value(routing.NextVar(idx))
    return order


# ------------------------------
# Timeline helpers (state buckets, transitions)
# ------------------------------

def _summarize_timeline_simple(timeline: List[Dict[str, Any]]) -> Dict[str, float]:
    buckets = {
        "PreProduction": 0.0, "Production": 0.0, "Changeover": 0.0, "Packaging": 0.0,
        "RoutineSan": 0.0, "WeeklySan": 0.0, "FortnightlySan": 0.0, "ResourceCleaning": 0.0,
        "PostProduction": 0.0, "Other": 0.0
    }
    for ev in timeline:
        d = float(ev["end"]) - float(ev["start"])
        t = ev["task"]
        if t.startswith("Planned sanitisation before"):
            buckets["PreProduction"] += d
        elif t.startswith("Production start"):
            buckets["PreProduction"] += d
        elif t.startswith("Produce "):
            buckets["Production"] += d
        elif t.startswith("Wrapper/volume change"):
            buckets["Packaging"] += d
        elif t.startswith("Change-over cleaning"):
            buckets["Changeover"] += d
        elif t.startswith("Routine sanitisation"):
            buckets["RoutineSan"] += d
        elif t.startswith("Nitrogen bath cleaning") or t.startswith("AM/PM cleaning"):
            buckets["WeeklySan"] += d
        elif t.startswith("Tunnel & nitrogen & sauce tank washing"):
            buckets["FortnightlySan"] += d
        elif t.startswith("Resource clean "):
            buckets["ResourceCleaning"] += d
        elif t.startswith("Planned sanitisation after end"):
            buckets["PostProduction"] += d
        elif t.startswith("End of production"):
            buckets["PostProduction"] += d
        else:
            buckets["Other"] += d
    return buckets


def _transition_durations_after_each_product(
    order: List[int],
    sku_order: List[str],
    timeline: List[Dict[str, Any]],
) -> Dict[str, float]:
    """
    Measure time between end of first run of SKU a and start of first run of SKU b
    for each (a,b) adjacent in the order. (Coarse, but enough for summary.)
    """
    produce_spans: Dict[str, List[Dict[str, float]]] = {}
    for ev in timeline:
        if isinstance(ev.get('task'), str) and ev['task'].startswith('Produce '):
            sku = ev['task'].split(' ', 1)[1]
            produce_spans.setdefault(sku, []).append({'start': ev['start'], 'end': ev['end']})
    cols: Dict[str, float] = {}
    for i in range(len(order) - 1):
        a = sku_order[order[i]]
        b = sku_order[order[i+1]]
        if not produce_spans.get(a) or not produce_spans.get(b):
            cols[f'after_{a}_to_{b}_transition_h'] = 0.0
            continue
        end_a = float(produce_spans[a][0]['end'])
        start_b = float(produce_spans[b][0]['start'])
        cols[f'after_{a}_to_{b}_transition_h'] = max(0.0, start_b - end_a)
    return cols


def _all_transition_headers(order_candidates: List[List[int]], sku_order: List[str]) -> List[str]:
    pairs = set()
    for order in order_candidates:
        for i in range(len(order) - 1):
            a = sku_order[order[i]]
            b = sku_order[order[i+1]]
            pairs.add((a, b))
    idx = {sku: i for i, sku in enumerate(sku_order)}
    pairs_sorted = sorted(list(pairs), key=lambda ab: (idx[ab[0]], idx[ab[1]]))
    return [f'after_{a}_to_{b}_transition_h' for a, b in pairs_sorted]


def sample_random_orders(order_indices: List[int], k: int, seed: int = 42) -> List[List[int]]:
    rng = random.Random(seed)
    seen = set()
    orders = []
    ident = tuple(order_indices)
    seen.add(ident)
    orders.append(list(ident))
    attempts, cap = 0, 50 * k
    while len(orders) < k and attempts < cap:
        perm = order_indices[:]
        rng.shuffle(perm)
        key = tuple(perm)
        if key not in seen:
            seen.add(key)
            orders.append(perm)
        attempts += 1
    return orders


# ------------------------------
# Weekly inventory-aware simulation with TIME LIMIT
# ------------------------------

def simulate_weekly_schedule_with_inventory(
    order: List[int],
    sku_order: List[str],
    weekly_demand: Dict[str, float],
    weeks: int,
    rate_map: Dict[str, float],
    base_times: Dict[str, Dict[str, float]],
    tasks: Dict[str, Dict[str, str]],
    packaging_change: Dict[str, Dict[str, float]],
    resource_map: Dict[str, Iterable[str]],
    resources: Dict[str, Resource],
    time_limit_h: float,   # <<< HARD TIME LIMIT >>>
) -> Tuple[List[Dict[str, Any]], float, Dict[str, float]]:
    """
    Build a timeline for up to `weeks` repeated runs of the same SKU order,
    but DO NOT exceed time_limit_h.

    We *aim* to produce weekly_demand for each SKU per week, but if we
    run out of time, we stop immediately and return partial production.

    Returns:
      - timeline: list of events
      - total_hours: final time (<= time_limit_h)
      - bars_by_sku: dict sku -> total bars actually produced
    """
    timeline: List[Dict[str, Any]] = []
    current = 0.0
    eps = 1e-9

    def can_add(duration: float) -> bool:
        return current + duration <= time_limit_h + eps

    time_exhausted = False

    # Pre-production once at start
    if can_add(5.0):
        timeline.append({'task': 'Planned sanitisation before start', 'start': current, 'end': current + 5.0})
        current += 5.0
    else:
        return timeline, current, {sku: 0.0 for sku in sku_order}

    if can_add(1.0):
        timeline.append({'task': 'Production start procedures', 'start': current, 'end': current + 1.0})
        current += 1.0
    else:
        return timeline, current, {sku: 0.0 for sku in sku_order}

    # Periodic cleaning thresholds
    next_daily = 24.0
    next_weekly = 168.0
    next_fortnightly = 336.0

    def insert_periodic():
        nonlocal current, next_daily, next_weekly, next_fortnightly, time_exhausted
        # Daily
        while current >= next_daily and not time_exhausted:
            if not can_add(1.0):
                time_exhausted = True
                return
            timeline.append({
                'task': 'Routine sanitisation (rinse x3 / knives & seals)',
                'start': current,
                'end': current + 1.0
            })
            current += 1.0
            next_daily += 24.0
        # Weekly
        while current >= next_weekly and not time_exhausted:
            # Nitrogen bath cleaning (4h)
            if not can_add(4.0):
                time_exhausted = True
                return
            timeline.append({
                'task': 'Nitrogen bath cleaning',
                'start': current,
                'end': current + 4.0
            })
            current += 4.0
            if not can_add(6.0):
                time_exhausted = True
                return
            # AM/PM cleaning (6h)
            timeline.append({
                'task': 'AM/PM cleaning',
                'start': current,
                'end': current + 6.0
            })
            current += 6.0
            next_weekly += 168.0
        # Fortnightly
        while current >= next_fortnightly and not time_exhausted:
            if not can_add(14.0):
                time_exhausted = True
                return
            timeline.append({
                'task': 'Tunnel & nitrogen & sauce tank washing',
                'start': current,
                'end': current + 14.0
            })
            current += 14.0
            next_fortnightly += 336.0

    # Resource tracking
    res_run: Dict[str, float] = {r: 0.0 for r in resources}
    res_last_active: Dict[str, float] = {r: 0.0 for r in resources}
    res_last_clean: Dict[str, float] = {r: 0.0 for r in resources}

    def clean_resource(rname: str, reason: str):
        nonlocal current, time_exhausted
        res = resources[rname]
        if not can_add(res.clean_time):
            time_exhausted = True
            return
        timeline.append({
            'task': f'Resource clean {rname} ({reason})',
            'start': current,
            'end': current + res.clean_time
        })
        current += res.clean_time
        insert_periodic()
        res_run[rname] = 0.0
        res_last_clean[rname] = current

    last_sku: Optional[str] = None
    bars_by_sku: Dict[str, float] = {sku: 0.0 for sku in sku_order}

    # Simulate week-by-week, but stop when time_limit_h is reached
    for week_idx in range(weeks):
        if time_exhausted or current >= time_limit_h - eps:
            break

        for idx in order:
            if time_exhausted or current >= time_limit_h - eps:
                break

            sku = sku_order[idx]
            weekly_qty = weekly_demand.get(sku, 0.0)
            if weekly_qty <= 0:
                continue

            insert_periodic()
            if time_exhausted or current >= time_limit_h - eps:
                break

            # Changeovers between SKUs
            if last_sku is not None and last_sku != sku:
                # Packaging / wrapper/volume change
                pack = packaging_change[last_sku][sku]
                if pack > 0:
                    if not can_add(pack):
                        time_exhausted = True
                        break
                    timeline.append({
                        'task': f'Wrapper/volume change {last_sku} → {sku}',
                        'start': current,
                        'end': current + pack
                    })
                    current += pack
                    insert_periodic()
                    if time_exhausted or current >= time_limit_h - eps:
                        break

                # Cleaning changeover A,B,C...
                code = tasks[last_sku][sku]
                extra = 1.0 if code.endswith('*') else 0.0
                base = base_times[last_sku][sku]
                dur = base + extra
                if dur > 0:
                    if not can_add(dur):
                        time_exhausted = True
                        break
                    timeline.append({
                        'task': f'Change-over cleaning {last_sku} → {sku} (code {code.strip("*")}'
                                f'{" + extra" if extra else ""})',
                        'start': current,
                        'end': current + dur
                    })
                    current += dur
                    insert_periodic()
                    if time_exhausted or current >= time_limit_h - eps:
                        break

            if time_exhausted or current >= time_limit_h - eps:
                break

            # Production run for this week's demand for this SKU
            run = weekly_qty / rate_map[sku]
            if run <= 0:
                continue

            # Check if we can fit the production run
            if not can_add(run):
                # We choose NOT to start a partial run; horizon is tight.
                time_exhausted = True
                break

            used_resources = list(resource_map.get(sku, []))

            # Resource constraints
            for rname in used_resources:
                if time_exhausted:
                    break
                res = resources[rname]
                idle = current - res_last_active[rname]
                # Idle storage limit
                if idle > res.max_storage_time:
                    clean_resource(rname, f'idle>{res.max_storage_time}h')
                    if time_exhausted or current >= time_limit_h - eps:
                        break
                # Max continuous run time
                if res.max_run_time > 0 and (res_run[rname] + run > res.max_run_time):
                    clean_resource(rname, f'run>{res.max_run_time}h')
                    if time_exhausted or current >= time_limit_h - eps:
                        break

            if time_exhausted or current >= time_limit_h - eps:
                break

            # Final check before actually producing
            if not can_add(run):
                time_exhausted = True
                break

            # Produce
            timeline.append({
                'task': f'Produce {sku}',
                'start': current,
                'end': current + run
            })
            current += run
            insert_periodic()
            if time_exhausted or current >= time_limit_h - eps:
                # We've used up horizon during/after production
                bars_by_sku[sku] += weekly_qty  # produced full weekly demand
                last_sku = sku
                break

            for rname in used_resources:
                res_run[rname] += run
                res_last_active[rname] = current

            bars_by_sku[sku] += weekly_qty
            last_sku = sku

        if time_exhausted or current >= time_limit_h - eps:
            break

    # Post-production (once), only if there is still time
    if can_add(6.0):
        timeline.append({
            'task': 'Planned sanitisation after end of production',
            'start': current,
            'end': current + 6.0
        })
        current += 6.0
    # try end-of-production procedures
    if can_add(0.5):
        timeline.append({
            'task': 'End of production procedures',
            'start': current,
            'end': current + 0.5
        })
        current += 0.5

    total_hours = min(current, time_limit_h)
    return timeline, total_hours, bars_by_sku


# ------------------------------
# GRID: weekly inventory → CSV (time-limited)
# ------------------------------

def grid_search_inventory_vs_horizon_to_csv(
    weeks_grid: List[int],
    order_candidates: List[List[int]],
    sku_order: List[str],
    weekly_demand: Dict[str, float],
    rate_map: Dict[str, float],
    base_times: Dict[str, Dict[str, float]],
    tasks: Dict[str, Dict[str, str]],
    packaging_change: Dict[str, Dict[str, float]],
    resource_map: Dict[str, Iterable[str]],
    resources: Dict[str, Any],
    summary_csv_path: str = "grid_demand_summary.csv",
    events_csv_path: str = "grid_demand_events.csv",
) -> None:
    """
    For each horizon (in WEEKS) and each candidate order:

      - Simulate a week-by-week schedule, but stop when total time reaches
        time_limit_h = weeks * 7 * 24.
      - weekly_demand is the *target*; actual production may be less if
        the horizon is tight.
    """
    transition_headers = _all_transition_headers(order_candidates, sku_order)
    per_sku_bars_headers = [f"bars_{sku}" for sku in sku_order]
    per_sku_time_headers = [f"prod_time_{sku}_h" for sku in sku_order]

    summary_headers = [
        "time_horizon_h",      # nominal: weeks * 7 * 24
        "schedule_id",
        "sequence",
        "scale_of_demand",     # produced / planned demand over horizon
        "total_bars",
        "total_hours",
        "PreProduction","Production","Changeover","Packaging",
        "RoutineSan","WeeklySan","FortnightlySan","ResourceCleaning",
        "PostProduction","Other",
    ] + per_sku_bars_headers + per_sku_time_headers + transition_headers

    events_headers = [
        "time_horizon_h",
        "schedule_id",
        "step_index",
        "task",
        "sku",
        "start_h",
        "end_h",
        "duration_h",
    ]

    with open(summary_csv_path, "w", newline="") as fsum, open(events_csv_path, "w", newline="") as fevt:
        sw = csv.writer(fsum); ew = csv.writer(fevt)
        sw.writerow(summary_headers); ew.writerow(events_headers)

        for weeks in weeks_grid:
            time_limit_h = weeks * 7.0 * 24.0

            for sid, order in enumerate(order_candidates, start=1):
                # 1) simulate week-by-week schedule with HARD time limit
                timeline, total_h, bars_by_sku = simulate_weekly_schedule_with_inventory(
                    order,
                    sku_order,
                    weekly_demand,
                    weeks,
                    rate_map,
                    base_times,
                    tasks,
                    packaging_change,
                    resource_map,
                    resources,
                    time_limit_h=time_limit_h,
                )

                # 2) state buckets & transitions
                states = _summarize_timeline_simple(timeline)
                trans_cols = _transition_durations_after_each_product(order, sku_order, timeline)

                total_bars = sum(bars_by_sku.values())
                total_demand_horizon = sum(weekly_demand[s] * weeks for s in sku_order)
                scale_of_demand = (total_bars / total_demand_horizon) if total_demand_horizon > 0 else 0.0

                # per-SKU production times from bars_by_sku & rates
                time_by_sku = {
                    sku: (bars_by_sku[sku] / rate_map[sku]) if rate_map[sku] > 0 else 0.0
                    for sku in sku_order
                }

                # 3) summary row
                row = [
                    time_limit_h,
                    sid,
                    " -> ".join(sku_order[i] for i in order),
                    round(scale_of_demand, 6),
                    round(total_bars, 3),
                    round(total_h, 3),
                    round(states.get("PreProduction", 0.0), 3),
                    round(states.get("Production", 0.0), 3),
                    round(states.get("Changeover", 0.0), 3),
                    round(states.get("Packaging", 0.0), 3),
                    round(states.get("RoutineSan", 0.0), 3),
                    round(states.get("WeeklySan", 0.0), 3),
                    round(states.get("FortnightlySan", 0.0), 3),
                    round(states.get("ResourceCleaning", 0.0), 3),
                    round(states.get("PostProduction", 0.0), 3),
                    round(states.get("Other", 0.0), 3),
                ]

                # per-SKU bars
                row += [round(bars_by_sku.get(sku, 0.0), 3) for sku in sku_order]
                # per-SKU production time
                row += [round(time_by_sku.get(sku, 0.0), 3) for sku in sku_order]
                # transition columns
                for hname in transition_headers:
                    row.append(round(trans_cols.get(hname, 0.0), 3))

                sw.writerow(row)

                # 4) events CSV
                for j, ev in enumerate(timeline):
                    task = ev.get("task", "")
                    sku = task.split(' ', 1)[1] if task.startswith("Produce ") and len(task.split(' ', 1)) == 2 else ""
                    ew.writerow([
                        time_limit_h,
                        sid,
                        j,
                        task,
                        sku,
                        round(float(ev["start"]), 3),
                        round(float(ev["end"]), 3),
                        round(float(ev["end"]) - float(ev["start"]), 3),
                    ])


# ------------------------------
# Main
# ------------------------------

def main() -> None:
    skus = build_skus()
    base_times, task_codes = build_changeover_and_tasks()
    tasks_description = build_tasks_description()
    packaging_change = build_packaging_change_matrix(skus)

    # Per-SKU rates from 720 bars/min + OE by pack size
    rate_map = build_sku_rate_map(skus)

    # Nominal 6-week capacity-based demand profile
    demand_6w = build_6week_demand_from_capacity(rate_map, horizon_weeks=6.0, utilisation=None)

    # Convert to weekly demand
    weekly_demand = {sku: demand_6w[sku] / 6.0 for sku in demand_6w}

    # Processing time (for solver) for the *full 6-week* demand
    processing_time = compute_processing_times_by_rate(demand_6w, rate_map)

    # Solve order with OR-Tools
    sku_order = list(skus.keys())  # keep defined order index mapping
    cost_matrix = build_cost_matrix(sku_order, base_times, task_codes, packaging_change, processing_time)
    order_indices = solve_schedule(cost_matrix)
    if order_indices is None:
        print('No feasible order found')
        return

    ordered_skus = [sku_order[i] for i in order_indices]
    print('Optimal production order (used as base sequence):')
    print(' -> '.join(ordered_skus))

    # Print cleaning legend
    print('\nCleaning activities legend:')
    for code, desc in tasks_description.items():
        print(f"{code}: {desc}")

    # Resources (constraints from client)
    resources: Dict[str, Resource] = {
        'chocolate_tub':    Resource('chocolate_tub',    max_run_time=72.0, clean_time=1.0, max_storage_time=21*24),
        'chocolate_buffer': Resource('chocolate_buffer', max_run_time=72.0, clean_time=1.0, max_storage_time=21*24),
        'coating_tub':      Resource('coating_tub',      max_run_time=72.0, clean_time=1.0, max_storage_time=21*24),
        'coating_buffer':   Resource('coating_buffer',   max_run_time=72.0, clean_time=1.0, max_storage_time=21*24),
        'caramel_tub':      Resource('caramel_tub',      max_run_time=14*24, clean_time=3.0, max_storage_time=14*24),
        'caramel_buffer':   Resource('caramel_buffer',   max_run_time=10*24, clean_time=3.0, max_storage_time=10*24),
        'chocolate_tank':   Resource('chocolate_tank',   max_run_time=0.0,  clean_time=1.0, max_storage_time=10*24),
    }

    # Mapping of which resources are used by which SKU
    resource_map: Dict[str, Iterable[str]] = {}
    for code in skus:
        default_res = ['chocolate_tub', 'chocolate_buffer', 'coating_tub', 'coating_buffer']
        if code in ['C', 'CAB']:
            resource_map[code] = default_res + ['caramel_tub', 'caramel_buffer']
        else:
            resource_map[code] = default_res

    # Horizon grid expressed in WEEKS (inventory driven)
    weeks_grid = [
        6,   # 6 weeks
        12,  # 12 weeks
        24,  # 24 weeks
        48,  # 48 weeks
        52,  # 52 weeks
    ]

    # Sample schedule candidates (start with optimal order, plus random permutations)
    order_candidates = sample_random_orders(order_indices, k=50, seed=123)

    # Generate CSVs with inventory-aware, time-limited weekly schedule
    grid_search_inventory_vs_horizon_to_csv(
        weeks_grid=weeks_grid,
        order_candidates=order_candidates,
        sku_order=sku_order,
        weekly_demand=weekly_demand,
        rate_map=rate_map,
        base_times=base_times,
        tasks=task_codes,
        packaging_change=packaging_change,
        resource_map=resource_map,
        resources=resources,
        summary_csv_path="grid_demand_summary.csv",
        events_csv_path="grid_demand_events.csv",
    )

    print("\nCSV written: grid_demand_summary.csv, grid_demand_events.csv")


if __name__ == '__main__':
    main()
