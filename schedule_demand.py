#!/usr/bin/env python3
"""
icecream_schedule_extended.py

Version: horizon grid + actual demand

- Uses actual demand per SKU (capacity-based) for a nominal 6-week demand.
- For each time horizon in a grid and for each candidate sequence:
    * Finds scale_of_demand (0–1) that fits within that time.
    * Computes how many bars per SKU that corresponds to.
    * Computes time spent producing each SKU.
    * Writes a summary CSV row and step-wise events.

Outputs:

1) grid_demand_summary.csv
   Columns:
     - time_horizon_h
     - schedule_id
     - sequence
     - scale_of_demand
     - total_bars   <-- total number of bars produced in the given time
     - total_hours
     - time in each state bucket
     - bars_<SKU> (bars actually produced per SKU in that horizon)
     - prod_time_<SKU>_h
     - transition times after each product

2) grid_demand_events.csv
   Columns:
     - time_horizon_h
     - schedule_id
     - step_index
     - task
     - sku  (for Produce steps)
     - start_h, end_h, duration_h
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
        'Bm':  SKU('Bm',  'Magnum Billionaire Mini 6MP',                      ['M','G','J','D','S'], 55),
        'CAB': SKU('CAB', 'Magnum Double Caramel Almond & Billionaire Mini 6MP', ['M','G','J','D','S','Z'], 55),
        'S3':  SKU('S3',  'Magnum Double Starchaser 3MP',                    ['M','G'], 85),
        'S4':  SKU('S4',  'Magnum Double Starchaser 4MP',                    ['M','G'], 85),
        'Sm':  SKU('Sm',  'Magnum Double Starchaser Mini 6MP',               ['M','G'], 55),
        'C':   SKU('C',   'Magnum Double Caramel 4MP',                        ['M','G'], 85),
        'Ch3': SKU('Ch3', 'Magnum Double Cherry 3MP',                         ['M','G'], 85),
        'Ch4': SKU('Ch4', 'Magnum Double Cherry 4MP',                         ['M','G'], 85),
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
# Costs & solver
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
# Baseline timeline
# ------------------------------

def generate_timeline(order: List[int], skus: List[str], processing_time: Dict[str, float],
                      base_times: Dict[str, Dict[str, float]],
                      tasks: Dict[str, Dict[str, str]],
                      packaging_change: Dict[str, Dict[str, float]]) -> List[Dict[str, Any]]:
    timeline: List[Dict[str, Any]] = []
    current = 0.0

    # Pre
    timeline.append({'task':'Planned sanitisation before start','start':current,'end':current+5.0})
    current += 5.0
    timeline.append({'task':'Production start procedures','start':current,'end':current+1.0})
    current += 1.0

    next_daily = 24.0
    next_weekly = 168.0
    next_fortnightly = 336.0

    def insert_periodic():
        nonlocal current, next_daily, next_weekly, next_fortnightly
        while current >= next_daily:
            timeline.append({'task':'Routine sanitisation (rinse x3 / knives & seals)',
                             'start':current,'end':current+1.0})
            current += 1.0
            next_daily += 24.0
        while current >= next_weekly:
            timeline.append({'task':'Nitrogen bath cleaning','start':current,'end':current+4.0})
            current += 4.0
            timeline.append({'task':'AM/PM cleaning','start':current,'end':current+6.0})
            current += 6.0
            next_weekly += 168.0
        while current >= next_fortnightly:
            timeline.append({'task':'Tunnel & nitrogen & sauce tank washing',
                             'start':current,'end':current+14.0})
            current += 14.0
            next_fortnightly += 336.0

    last: Optional[str] = None
    for idx in order:
        sku = skus[idx]
        insert_periodic()
        if last is not None:
            pack = packaging_change[last][sku]
            if pack > 0:
                timeline.append({'task':f'Wrapper/volume change {last} → {sku}',
                                 'start':current,'end':current+pack})
                current += pack
                insert_periodic()
            code = tasks[last][sku]
            extra = 1.0 if code.endswith('*') else 0.0
            base = base_times[last][sku]
            if base + extra > 0:
                timeline.append({'task':f'Change-over cleaning {last} → {sku} (code {code.strip("*")}'
                                      f'{" + extra" if extra else ""})',
                                 'start':current,'end':current+base+extra})
                current += base + extra
                insert_periodic()

        run = processing_time[sku]
        timeline.append({'task':f'Produce {sku}','start':current,'end':current+run})
        current += run
        last = sku

    # Post
    timeline.append({'task':'Planned sanitisation after end of production','start':current,'end':current+6.0})
    current += 6.0
    timeline.append({'task':'End of production procedures','start':current,'end':current+0.5})
    current += 0.5
    return timeline


# ------------------------------
# Resource-aware simulators
# ------------------------------

def simulate_schedule_duration_with_resources(
    order: List[int],
    sku_order: List[str],
    scale: float,
    demand: Dict[str, float],
    rate_map: Dict[str, float],
    base_times: Dict[str, Dict[str, float]],
    tasks: Dict[str, Dict[str, str]],
    packaging_change: Dict[str, Dict[str, float]],
    resource_map: Dict[str, Iterable[str]],
    resources: Dict[str, Resource],
    include_pre_post: bool = True
) -> float:
    current = 0.0
    if include_pre_post:
        current += 5.0
        current += 1.0

    next_daily = 24.0
    next_weekly = 168.0
    next_fortnightly = 336.0

    res_run: Dict[str, float] = {r: 0.0 for r in resources}
    res_last_active: Dict[str, float] = {r: 0.0 for r in resources}
    res_last_clean: Dict[str, float] = {r: 0.0 for r in resources}

    def periodic():
        nonlocal current, next_daily, next_weekly, next_fortnightly
        while current >= next_daily:
            current += 1.0
            next_daily += 24.0
        while current >= next_weekly:
            current += 4.0
            current += 6.0
            next_weekly += 168.0
        while current >= next_fortnightly:
            current += 14.0
            next_fortnightly += 336.0

    last: Optional[str] = None
    for idx in order:
        sku = sku_order[idx]
        periodic()

        if last is not None:
            pack = packaging_change[last][sku]
            if pack > 0:
                current += pack
                periodic()
            code = tasks[last][sku]
            extra = 1.0 if code.endswith('*') else 0.0
            current += base_times[last][sku] + extra
            periodic()

        run = (demand[sku] * scale) / rate_map[sku]
        used = list(resource_map.get(sku, []))

        for rname in used:
            res = resources[rname]
            idle = current - res_last_active[rname]
            if idle > res.max_storage_time:
                current += res.clean_time
                res_run[rname] = 0.0
                res_last_clean[rname] = current
                periodic()
            if res.max_run_time > 0 and (res_run[rname] + run > res.max_run_time):
                current += res.clean_time
                res_run[rname] = 0.0
                res_last_clean[rname] = current
                periodic()

        current += run
        for rname in used:
            res_run[rname] += run
            res_last_active[rname] = current
        last = sku

    if include_pre_post:
        current += 6.0
        current += 0.5
    return current


# ------------------------------
# Scaling & max fraction of demand
# ------------------------------

def find_max_scale_with_resources(
    time_horizon: float,
    order: List[int],
    sku_order: List[str],
    demand: Dict[str, float],
    rate_map: Dict[str, float],
    base_times: Dict[str, Dict[str, float]],
    tasks: Dict[str, Dict[str, str]],
    packaging_change: Dict[str, Dict[str, float]],
    resource_map: Dict[str, Iterable[str]],
    resources: Dict[str, Resource]
) -> Tuple[float, float]:
    low, high, best = 0.0, 1.0, 0.0
    for _ in range(30):
        mid = (low + high) / 2
        tt = simulate_schedule_duration_with_resources(
            order, sku_order, mid, demand, rate_map,
            base_times, tasks, packaging_change, resource_map, resources, True
        )
        if tt <= time_horizon + 1e-6:
            best = mid
            low = mid
        else:
            high = mid
    total_bars = best * sum(demand.values())
    return best, total_bars


# ------------------------------
# Helpers for reporting
# ------------------------------

def _build_timeline_for_scaled_demand(
    order: List[int],
    sku_order: List[str],
    scale: float,
    demand: Dict[str, float],
    rate_map: Dict[str, float],
    base_times: Dict[str, Dict[str, float]],
    tasks: Dict[str, Dict[str, str]],
    packaging_change: Dict[str, Dict[str, float]],
) -> List[Dict[str, Any]]:
    """Generate a simple (resource-agnostic) timeline for scaled demand."""
    proc_time = {
        sku: (demand[sku] * scale) / rate_map[sku] for sku in demand
    }
    return generate_timeline(order, sku_order, proc_time, base_times, tasks, packaging_change)


def _transition_durations_after_each_product(
    order: List[int],
    sku_order: List[str],
    timeline: List[Dict[str, Any]],
) -> Dict[str, float]:
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
        elif t.startswith("Planned sanitisation after end"):
            buckets["PostProduction"] += d
        elif t.startswith("End of production"):
            buckets["PostProduction"] += d
        else:
            buckets["Other"] += d
    return buckets


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
# New: demand + horizon GRID → CSV
# ------------------------------

def grid_search_demand_vs_horizon_to_csv(
    time_horizon_grid: List[float],
    order_candidates: List[List[int]],
    sku_order: List[str],
    demand: Dict[str, float],
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
    For each time horizon in time_horizon_grid and each candidate order:

      - Interpret `demand` as a NOMINAL 6-week demand.
      - Scale that demand linearly with the horizon length:
            demand_h[sku] = demand_6w[sku] * (time_horizon_h / (6 * 7 * 24))
      - Find scale_of_demand in [0, 1] that fits within the horizon.
        (scale_of_demand = 1 means "we met 100% of the horizon's demand".)
      - Compute total_bars, per-SKU bars & production time.
      - Write summary & events rows.

    This makes longer horizons have proportionally larger demand as well
    as more time, so `total_bars` and utilisation meaningfully vary with
    the horizon instead of saturating at the 6-week demand.
    """
    # Treat the incoming `demand` as a 6-week demand profile
    six_week_hours = 6.0 * 7.0 * 24.0
    demand_6w = demand

    transition_headers = _all_transition_headers(order_candidates, sku_order)
    per_sku_bars_headers = [f"bars_{sku}" for sku in sku_order]
    per_sku_time_headers = [f"prod_time_{sku}_h" for sku in sku_order]

    summary_headers = [
        "time_horizon_h",
        "schedule_id",
        "sequence",
        "scale_of_demand",
        "total_bars",     # total number of bars produced in the given time
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

        for time_horizon_h in time_horizon_grid:
            # Scale the 6-week demand up/down to match this horizon
            horizon_scale = time_horizon_h / six_week_hours
            demand_h = {sku: demand_6w[sku] * horizon_scale for sku in sku_order}

            for sid, order in enumerate(order_candidates, start=1):
                # 1) resource-aware scale + total bars for THIS horizon's demand
                scale, total_bars = find_max_scale_with_resources(
                    time_horizon_h,
                    order,
                    sku_order,
                    demand_h,
                    rate_map,
                    base_times,
                    tasks,
                    packaging_change,
                    resource_map,
                    resources,
                )

                # 2) recompute total hours with that scale
                total_h = simulate_schedule_duration_with_resources(
                    order,
                    sku_order,
                    scale,
                    demand_h,
                    rate_map,
                    base_times,
                    tasks,
                    packaging_change,
                    resource_map,
                    resources,
                    include_pre_post=True,
                )

                # 3) simple timeline for states / transitions & events log
                timeline = _build_timeline_for_scaled_demand(
                    order, sku_order, scale, demand_h, rate_map,
                    base_times, tasks, packaging_change
                )
                states = _summarize_timeline_simple(timeline)
                trans_cols = _transition_durations_after_each_product(order, sku_order, timeline)

                # per-SKU bars/time (from horizon demand & scale)
                bars_by_sku = {sku: scale * demand_h[sku] for sku in sku_order}
                time_by_sku = {
                    sku: (bars_by_sku[sku] / rate_map[sku]) if rate_map[sku] > 0 else 0.0
                    for sku in sku_order
                }

                # 4) summary row
                row = [
                    time_horizon_h,
                    sid,
                    " -> ".join(sku_order[i] for i in order),
                    round(scale, 6),
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

                # 5) events CSV
                for j, ev in enumerate(timeline):
                    task = ev.get("task", "")
                    sku = task.split(' ', 1)[1] if task.startswith("Produce ") and len(task.split(' ',1)) == 2 else ""
                    ew.writerow([
                        time_horizon_h,
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

    # Actual demand for a nominal 6-week period (capacity-based, equal utilisation share)
    demand = build_6week_demand_from_capacity(rate_map, horizon_weeks=6.0, utilisation=None)

    # Processing time for solver's cost matrix (for whole 6-week demand)
    processing_time = compute_processing_times_by_rate(demand, rate_map)

    # Solve order with OR-Tools
    sku_order = list(skus.keys())  # keep defined order index mapping
    cost_matrix = build_cost_matrix(sku_order, base_times, task_codes, packaging_change, processing_time)
    order_indices = solve_schedule(cost_matrix)
    if order_indices is None:
        print('No feasible order found'); return

    ordered_skus = [sku_order[i] for i in order_indices]
    print('Optimal production order:')
    print(' -> '.join(ordered_skus))

    # Print legend (cleaning activities)
    print('\nCleaning activities legend:')
    for code, desc in tasks_description.items():
        print(f"{code}: {desc}")

    # Resources
    resources: Dict[str, Resource] = {
        'chocolate_tub':    Resource('chocolate_tub',    max_run_time=72.0, clean_time=1.0, max_storage_time=21*24),
        'chocolate_buffer': Resource('chocolate_buffer', max_run_time=72.0, clean_time=1.0, max_storage_time=21*24),
        'coating_tub':      Resource('coating_tub',      max_run_time=72.0, clean_time=1.0, max_storage_time=21*24),
        'coating_buffer':   Resource('coating_buffer',   max_run_time=72.0, clean_time=1.0, max_storage_time=21*24),
        'caramel_tub':      Resource('caramel_tub',      max_run_time=14*24, clean_time=3.0, max_storage_time=14*24),
        'caramel_buffer':   Resource('caramel_buffer',   max_run_time=10*24, clean_time=3.0, max_storage_time=10*24),
        'chocolate_tank':   Resource('chocolate_tank',   max_run_time=0.0,  clean_time=1.0, max_storage_time=10*24),
    }

    resource_map: Dict[str, Iterable[str]] = {}
    for code in skus:
        default_res = ['chocolate_tub','chocolate_buffer','coating_tub','coating_buffer']
        if code in ['C','CAB']:
            resource_map[code] = default_res + ['caramel_tub','caramel_buffer']
        else:
            resource_map[code] = default_res

    # Example time-horizon grid: 1 week, 3 weeks, 6 weeks
    time_horizon_grid = [
        6 * 7 * 24,   # 6 weeks
        12 * 7 * 24,  # 12 weeks
    24 * 7 * 24,  # 24 weeks
    48 * 7 * 24,  # 48 weeks
        52 * 7 * 24,  # 52 weeks
    ]

    # Quick print for optimal order & 6-week horizon
    time_horizon_hours = 6 * 7 * 24
    best_scale, max_bars = find_max_scale_with_resources(
        time_horizon_hours, order_indices, sku_order, demand, rate_map,
        base_times, task_codes, packaging_change, resource_map, resources
    )
    print(f"\nFor 6-week horizon ({time_horizon_hours} h):")
    print(f"  Fraction of 6-week demand produced (optimal order): {best_scale:.3f}")
    print(f"  Total bars produced in this time: {max_bars:.0f}")

    # Sample ~10 random sequences (incl. optimal as first)
    order_candidates = sample_random_orders(order_indices, k=50, seed=123)

    # Generate CSVs over the time-horizon grid
    grid_search_demand_vs_horizon_to_csv(
        time_horizon_grid=time_horizon_grid,
        order_candidates=order_candidates,
        sku_order=sku_order,
        demand=demand,
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
