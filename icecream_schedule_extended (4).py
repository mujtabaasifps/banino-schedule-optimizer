#!/usr/bin/env python3
"""
icecream_schedule_extended.py
=============================

This program extends the ice‑cream scheduling example to take into
account allergens, detailed cleaning activities and routine
sanitisation tasks.  It synthesises information from two tables:

* A matrix of change‑over times between different Magnum SKUs
  (Figure 1).  Each cell lists both the time (in hours) required to
  change from one product to another and a letter code describing the
  cleaning activities needed during the change‑over【396900509045137†screenshot】.
  Some codes include an asterisk (e.g. ``C*``); these indicate that an
  additional hour must be added because the tank must be checked and
  possibly drained before switching chocolate types【396900509045137†screenshot】.
* A sanitisation schedule (Figure 2) that specifies how often deep
  cleaning tasks must occur, such as routine daily rinses and
  periodic tunnel or nitrogen‑bath washes【771340528377493†screenshot】.

The script constructs a single‑machine scheduling problem in which
each SKU has a demand (number of units to produce) and a production
rate (units per hour).  It then builds a cost matrix that combines
processing time, change‑over time, additional packaging change time
(depending on whether the products use 85 ml or 55 ml wrappers) and
the extra hour required for starred cleaning activities.  The cost
matrix is passed to OR‑Tools’ routing solver to find an order of
production that minimises the total time.  After the optimal sequence
is computed, the script produces a detailed timeline that includes
pre‑production and post‑production sanitisation, change‑over cleaning
activities with their descriptions, packaging/volume changes, wrapper
changes and periodic maintenance tasks.

The solution assumes the following:

* ``3MP`` or ``4MP`` SKUs are produced in 85 ml bars, while ``6MP``
  (mini) SKUs are 55 ml.  Changing the wrapper size from 85 ml to
  55 ml incurs a 2 h delay, whereas switching from 55 ml back to
  85 ml incurs a 6 h delay; a wrapper change itself requires an
  additional 1 h【771340528377493†screenshot】.
* Heavy allergens (peanuts ``Z``, nuts ``D`` and sesame ``Se``) should
  be processed last in order to avoid the most extensive cleaning
  operations【396900509045137†screenshot】.
* Routine sanitisation (1 h) occurs every 24 h of accumulated run time;
  nitrogen‑bath cleaning (4 h) and AM/PM checks (6 h) occur every
  168 h; tunnel/nitrogen/sauce tank washing (14 h) occurs every 336 h
  (14 days)【771340528377493†screenshot】.
* A planned sanitisation (5 h) and production start (1 h) occur before
  the first run, and a planned sanitisation (6 h) followed by a
  0.5 h shutdown occur after the final run【771340528377493†screenshot】.

Users should adjust demands, production rates, wrapper sizes and
maintenance frequencies to suit their specific factory.  The module is
written for clarity rather than absolute performance and serves as a
template for more complex scheduling.
"""

from __future__ import annotations
from itertools import combinations, permutations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any, Iterable
import numpy as np
import random
from ortools.constraint_solver import pywrapcp, routing_enums_pb2


@dataclass
class SKU:
    """Representation of a stock‑keeping unit (SKU)."""
    code: str
    name: str
    allergens: List[str]
    volume_ml: int  # nominal volume of the bar (e.g. 85 or 55)


@dataclass
class Resource:
    """Represent a production resource such as a chocolate tub or caramel buffer.

    Each resource has a maximum continuous run time (in hours) before a
    cleaning cycle must be triggered, a cleaning duration (in hours),
    and an optional maximum storage time (in hours) during which it
    may sit idle before requiring a rinse prior to reuse.  These
    parameters are derived from the constraints provided by the user:

    * Chocolate tub, chocolate buffer, coating tub and coating buffer
      can remain at temperature (>50 °C) for at most 72 hours before
      they must be drained and rinsed with fresh material【771340528377493†screenshot】.
    * Caramel tub must be cleaned more frequently (≤14 days) and
      different procedures apply depending on the length of the
      production break; here we conservatively assume a single
      cleaning duration applies once the run time threshold is
      exceeded.
    * Chocolate tanks (GMP) stored dry for up to 10 days must be
      rinsed with raw material prior to use.  This is modelled via
      ``max_storage_time``.
    """
    name: str
    max_run_time: float
    clean_time: float
    max_storage_time: float = float('inf')


def build_skus() -> Dict[str, SKU]:
    """Define all SKUs with their allergens and nominal volumes.

    Allergens are based on the column header of the change‑over matrix
    where each letter corresponds to an ingredient category【396900509045137†screenshot】:

    * ``M`` – milk and dairy products
    * ``G`` – grains containing gluten
    * ``J`` – eggs
    * ``Z`` – peanuts
    * ``S`` – soy
    * ``D`` – tree nuts
    * ``Se`` – sesame seeds (appears only on Hazelnut)

    Nominal volumes are deduced from the product names:
    multipacks with 3 MP or 4 MP are assumed to use the standard
    85 ml bars, whereas mini multipacks with 6 MP use the 55 ml format.
    """
    skus: Dict[str, SKU] = {
        'B3': SKU('B3', 'Magnum Billionaire Standard 3MP', ['M', 'G', 'J', 'D', 'S'], 85),
        'B4': SKU('B4', 'Magnum Billionaire Standard 4MP', ['M', 'G', 'J', 'D', 'S'], 85),
        'Bm': SKU('Bm', 'Magnum Billionaire Mini 6MP',     ['M', 'G', 'J', 'D', 'S'], 55),
        'CAB': SKU('CAB', 'Magnum Double Caramel Almond & Billionaire Mini 6MP', ['M', 'G', 'J', 'D', 'S', 'Z'], 55),
        'S3': SKU('S3', 'Magnum Double Starchaser 3MP',    ['M', 'G'], 85),
        'S4': SKU('S4', 'Magnum Double Starchaser 4MP',    ['M', 'G'], 85),
        'Sm': SKU('Sm', 'Magnum Double Starchaser Mini 6MP', ['M', 'G'], 55),
        'C':  SKU('C',  'Magnum Double Caramel 4MP',       ['M', 'G'], 85),
        'Ch3': SKU('Ch3', 'Magnum Double Cherry 3MP',       ['M', 'G'], 85),
        'Ch4': SKU('Ch4', 'Magnum Double Cherry 4MP',       ['M', 'G'], 85),
        'Hm': SKU('Hm', 'Magnum Double Hazelnut Mini 6MP', ['M', 'D', 'Se'], 55),
    }
    return skus


def build_changeover_and_tasks() -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, str]]]:
    """Construct base change‑over times and cleaning activity codes for each pair.

    Returns
    -------
    times : dict
        times[a][b] is the number of hours required to clean the line
        when switching from SKU ``a`` to ``b``.  These times exclude
        any extra hours for starred operations (the stars are handled
        separately when building the cost matrix).
    tasks : dict
        tasks[a][b] is the letter code describing the cleaning
        activities necessary when transitioning from ``a`` to ``b``
       【396900509045137†screenshot】.  Starred codes are returned literally (e.g. ``C*``).
    """
    skus = [
        'B3', 'B4', 'Bm', 'CAB', 'S3', 'S4', 'Sm', 'C', 'Ch3', 'Ch4', 'Hm'
    ]
    # Base times (without star bonus) and task codes extracted from the matrix【396900509045137†screenshot】.
    times: Dict[str, Dict[str, float]] = {sku: {} for sku in skus}
    tasks: Dict[str, Dict[str, str]] = {sku: {} for sku in skus}

    # Helper to assign entries for a single row.
    def row(from_sku: str, entries: List[Tuple[str, float, str]]) -> None:
        for to_sku, t, code in entries:
            times[from_sku][to_sku] = t
            tasks[from_sku][to_sku] = code

    # Populate using the first figure. See analysis for details.
    row('B3', [
        ('B3', 1, '-'), ('B4', 2, '-'), ('Bm', 2, '-'), ('CAB', 10, 'B*'),
        ('S3', 13, 'C*'), ('S4', 13, 'C*'), ('Sm', 13, 'C*'),
        ('C',  15, 'D*'), ('Ch3', 14, 'E*'), ('Ch4', 14, 'E*'), ('Hm', 14, 'E*')
    ])
    row('B4', [
        ('B3', 2, '-'), ('B4', 1, '-'), ('Bm', 2, '-'), ('CAB', 10, 'B*'),
        ('S3', 13, 'C*'), ('S4', 13, 'C*'), ('Sm', 13, 'C*'),
        ('C',  15, 'D*'), ('Ch3', 14, 'E*'), ('Ch4', 14, 'E*'), ('Hm', 14, 'E*')
    ])
    row('Bm', [
        ('B3', 6, 'A'), ('B4', 6, 'A'), ('Bm', 1, '-'), ('CAB', 10, 'B*'),
        ('S3', 13, 'C*'), ('S4', 13, 'C*'), ('Sm', 13, 'C*'),
        ('C',  15, 'D*'), ('Ch3', 14, 'E*'), ('Ch4', 14, 'E*'), ('Hm', 14, 'E*')
    ])
    row('CAB', [
        ('B3', 14, 'G'), ('B4', 14, 'G'), ('Bm', 14, 'G'), ('CAB', 1, '-'),
        ('S3', 14, 'G'), ('S4', 14, 'G'), ('Sm', 14, 'G'),
        ('C',  16, 'H'), ('Ch3', 15, 'I'), ('Ch4', 15, 'I'), ('Hm', 16, 'J')
    ])
    row('S3', [
        ('B3', 10, 'B'), ('B4', 10, 'B'), ('Bm', 10, 'B'), ('CAB', 10, 'B'),
        ('S3', 1,  '-'), ('S4', 2,  '-'), ('Sm', 2,  '-'),
        ('C',  13, 'C'), ('Ch3', 14, 'E'), ('Ch4', 14, 'E'), ('Hm', 15, 'F')
    ])
    row('S4', [
        ('B3', 10, 'B'), ('B4', 10, 'B'), ('Bm', 10, 'B'), ('CAB', 10, 'B'),
        ('S3', 2,  '-'), ('S4', 1,  '-'), ('Sm', 2,  '-'),
        ('C',  13, 'C'), ('Ch3', 14, 'E'), ('Ch4', 14, 'E'), ('Hm', 15, 'F')
    ])
    row('Sm', [
        ('B3', 10, 'B'), ('B4', 10, 'B'), ('Bm', 10, 'B'), ('CAB', 10, 'B'),
        ('S3', 6,  'A'), ('S4', 6,  'A'), ('Sm', 1,  '-'),
        ('C',  13, 'C'), ('Ch3', 14, 'E'), ('Ch4', 14, 'E'), ('Hm', 15, 'F')
    ])
    row('C', [
        ('B3', 11, 'K'), ('B4', 11, 'K'), ('Bm', 11, 'K'), ('CAB', 11, 'K'),
        ('S3', 8,  'L'), ('S4', 8,  'L'), ('Sm', 8,  'L'),
        ('C',  1,  '-'), ('Ch3', 16, 'M'), ('Ch4', 16, 'M'), ('Hm', 17, 'N')
    ])
    row('Ch3', [
        ('B3', 14, 'E'), ('B4', 14, 'E'), ('Bm', 14, 'E'), ('CAB', 14, 'E'),
        ('S3', 14, 'E'), ('S4', 14, 'E'), ('Sm', 14, 'E'),
        ('C',  16, 'M'), ('Ch3', 1,  '-'), ('Ch4', 2,  '-'), ('Hm', 15, 'F')
    ])
    row('Ch4', [
        ('B3', 14, 'E'), ('B4', 14, 'E'), ('Bm', 14, 'E'), ('CAB', 14, 'E'),
        ('S3', 14, 'E'), ('S4', 14, 'E'), ('Sm', 14, 'E'),
        ('C',  16, 'M'), ('Ch3', 2,  '-'), ('Ch4', 1,  '-'), ('Hm', 15, 'F')
    ])
    row('Hm', [
        ('B3', 16, 'O'), ('B4', 16, 'O'), ('Bm', 16, 'O'), ('CAB', 17, 'P'),
        ('S3', 17, 'P'), ('S4', 17, 'P'), ('Sm', 17, 'P'),
        ('C',  18, 'R'), ('Ch3', 17, 'P'), ('Ch4', 17, 'P'), ('Hm', 1,  '-')
    ])

    return times, tasks


def build_tasks_description() -> Dict[str, str]:
    """Map cleaning activity codes to human‑readable descriptions.【396900509045137†screenshot】"""
    return {
        'A': 'Trays cleaning; wash freezers and extruders (Mini → Double format)',
        'B': 'Trays cleaning; wash freezers, extruders, GMPs and chocolate bath',
        'C': 'Trays cleaning; wash freezers, extruders, slat conveyor, GPMs, nitrogen bath, chocolate bath, sauce bath, P&P Robots, and GMW wrapper',
        'D': 'Trays cleaning; wash freezers, extruders, slat conveyor, GPMs, nitrogen bath, chocolate bath, sauce bath, sauce tank, cover bath, cover tank M2, P&P Robots, and GMW wrapper',
        'E': 'Trays cleaning; wash freezers, extruders, slat conveyor, GPMs, nitrogen bath, chocolate bath, sauce bath, sauce tank, P&P Robots, and GMW wrapper',
        'F': 'Trays cleaning; wash freezers, extruders, slat conveyor, GPMs, nitrogen bath, chocolate bath, chocolate tank M1, sauce bath, sauce tank, P&P Robots, and GMW wrapper',
        'G': 'Trays cleaning; wash freezers, extruders, slat conveyor, GPMs, nitrogen bath, chocolate bath, sauce bath, P&P Robots, and GMW wrapper + allergen inspection',
        'H': 'Trays cleaning; wash freezers, extruders, slat conveyor, GPMs, nitrogen bath, chocolate bath, sauce bath, cover bath, cover tank M1, P&P Robots, and GMW wrapper + allergen inspection',
        'I': 'Trays cleaning; wash freezers, extruders, slat conveyor, GPMs, nitrogen bath, chocolate bath, sauce bath, sauce tank, P&P Robots, and GMW wrapper + allergen inspection',
        'J': 'Trays cleaning; wash freezers, extruders, slat conveyor, GPMs, nitrogen bath, chocolate bath, chocolate tank M1, sauce bath, sauce tank, P&P Robots, and GMW wrapper + allergen inspection',
        'K': 'Trays cleaning; wash freezers, extruders, GPMs, chocolate bath, cover bath, cover tank M2',
        'L': 'Trays cleaning; wash freezers, extruders, GPMs, cover bath, cover tank M2',
        'M': 'Trays cleaning; wash freezers, extruders, slat conveyor, GPMs, nitrogen bath, chocolate bath, sauce bath, sauce tank, cover bath, cover tank M2, P&P Robots, and GMW wrapper',
        'N': 'Trays cleaning; wash freezers, extruders, slat conveyor, GPMs, nitrogen bath, chocolate bath, chocolate tank M1, sauce bath, sauce tank, cover bath, cover tank M2, P&P Robots, and GMW wrapper',
        'O': 'Tunnel washing; wash freezers, extruders, slat conveyor, GPMs, nitrogen bath, chocolate bath, sauce bath, sauce tank, P&P Robots, and GMW wrapper + allergen inspection (including the tunnel)',
        'P': 'Tunnel washing; wash freezers, extruders, slat conveyor, GPMs, nitrogen bath, chocolate bath, chocolate tank M1, sauce bath, sauce tank, P&P Robots, and GMW wrapper + allergen inspection (including the tunnel)',
        'R': 'Tunnel washing; wash freezers, extruders, slat conveyor, GPMs, nitrogen bath, chocolate bath, chocolate tank M1, sauce bath, sauce tank, cover bath, cover tank M2, P&P Robots, and GMW wrapper + allergen inspection (including the tunnel)',
    }


def build_packaging_change_matrix(skus: Dict[str, SKU]) -> Dict[str, Dict[str, float]]:
    """Calculate extra time when switching wrapper sizes.

    If the two SKUs have the same nominal bar volume the time is zero.  If
    switching from 85 ml → 55 ml the additional time is 2 h; from
    55 ml → 85 ml it is 6 h.  A wrapper change itself always takes
    1 h, so the returned time includes the 1 h wrapper change【771340528377493†screenshot】.
    """
    volumes: Dict[str, int] = {code: sku.volume_ml for code, sku in skus.items()}
    change_time: Dict[str, Dict[str, float]] = {a: {} for a in skus}
    for a in skus:
        for b in skus:
            if a == b:
                change_time[a][b] = 0.0
            else:
                v_from = volumes[a]
                v_to = volumes[b]
                if v_from == v_to:
                    change_time[a][b] = 0.0
                elif v_from == 85 and v_to == 55:
                    change_time[a][b] = 2.0 + 1.0  # 2 h size change + 1 h wrapper
                elif v_from == 55 and v_to == 85:
                    change_time[a][b] = 6.0 + 1.0  # 6 h size change + 1 h wrapper
                else:
                    # For unexpected volumes assign zero; update if new sizes appear.
                    change_time[a][b] = 0.0
    return change_time


def compute_processing_times(demand: Dict[str, float], rate: float) -> Dict[str, float]:
    """Compute the time (hours) required to produce each SKU based on demand and rate."""
    return {sku: demand[sku] / rate for sku in demand}


def build_cost_matrix(skus: List[str], base_times: Dict[str, Dict[str, float]],
                      tasks: Dict[str, Dict[str, str]],
                      packaging_change: Dict[str, Dict[str, float]],
                      processing_time: Dict[str, float]) -> List[List[float]]:
    """Assemble the cost matrix for the routing solver.

    The cost of moving from SKU ``a`` to ``b`` equals:

    * base change‑over time (base_times)
    * +1 h if the activity code ends with ``*`` (star)
    * + packaging change time (if the bar volumes differ)
    * + processing time for SKU ``b``
    """
    n = len(skus)
    cost: List[List[float]] = [[0.0] * n for _ in range(n)]
    for i, a in enumerate(skus):
        for j, b in enumerate(skus):
            # base cleaning time
            base = base_times[a][b]
            code = tasks[a][b]
            extra = 1.0 if code.endswith('*') else 0.0
            pack = packaging_change[a][b]
            pt = processing_time[b]
            cost[i][j] = base + extra + pack + pt
    return cost


def solve_schedule(cost_matrix: List[List[float]]) -> Optional[List[int]]:
    """Find an ordering of SKUs that minimises total cost using OR‑Tools."""
    n = len(cost_matrix)
    manager = pywrapcp.RoutingIndexManager(n, 1, 0)
    routing = pywrapcp.RoutingModel(manager)

    def transit_cb(from_index: int, to_index: int) -> int:
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return int(cost_matrix[from_node][to_node] * 1000)

    transit_idx = routing.RegisterTransitCallback(transit_cb)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_idx)

    search_params = pywrapcp.DefaultRoutingSearchParameters()
    search_params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    search_params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    search_params.time_limit.seconds = 10
    search_params.log_search = False

    solution = routing.SolveWithParameters(search_params)
    if solution is None:
        return None
    index = routing.Start(0)
    order: List[int] = []
    while not routing.IsEnd(index):
        node = manager.IndexToNode(index)
        order.append(node)
        index = solution.Value(routing.NextVar(index))
    return order


def generate_timeline(order: List[int], skus: List[str], processing_time: Dict[str, float],
                      base_times: Dict[str, Dict[str, float]], tasks: Dict[str, Dict[str, str]],
                      packaging_change: Dict[str, Dict[str, float]]) -> List[Dict[str, Any]]:
    """Produce a detailed timeline for the production run.

    The timeline includes:

    * Pre‑production sanitisation and start‑up (5 h + 1 h).
    * For each change‑over: packaging change (if applicable), extra hour for star codes,
      cleaning tasks with descriptive labels and durations; wrapper changes are
      included in the packaging change.
    * Production time for each SKU.
    * Routine sanitisation tasks inserted every 24 h of accumulated time.
    * Nitrogen bath cleaning (4 h) and AM/PM checks (6 h) every 168 h.
    * Tunnel washing + nitrogen bath + sauce tank washing (14 h) every 336 h.
    * Post‑production sanitisation and shutdown (6 h + 0.5 h).

    Returns a list of events with start and end times and a description.
    """
    timeline: List[Dict[str, Any]] = []
    current_time = 0.0
    # Pre‑production tasks
    timeline.append({'task': 'Planned sanitisation before start', 'start': current_time, 'end': current_time + 5.0})
    current_time += 5.0
    timeline.append({'task': 'Production start procedures', 'start': current_time, 'end': current_time + 1.0})
    current_time += 1.0

    # Times at which periodic tasks last executed
    next_daily = 24.0
    next_weekly = 168.0
    next_fortnightly = 336.0

    last_sku: Optional[str] = None
    for idx in order:
        sku = skus[idx]
        # Insert periodic maintenance if due before starting the next change‑over
        def insert_periodic_tasks() -> None:
            nonlocal current_time, next_daily, next_weekly, next_fortnightly
            # Daily routine sanitisation【771340528377493†screenshot】
            while current_time >= next_daily:
                timeline.append({'task': 'Routine sanitisation (rinse x3 / knife & seal cleaning)',
                                 'start': current_time, 'end': current_time + 1.0})
                current_time += 1.0
                next_daily += 24.0
            # Weekly nitrogen bath cleaning and AM/PM check【771340528377493†screenshot】
            while current_time >= next_weekly:
                timeline.append({'task': 'Nitrogen bath cleaning', 'start': current_time,
                                 'end': current_time + 4.0})
                current_time += 4.0
                timeline.append({'task': 'AM/PM cleaning', 'start': current_time,
                                 'end': current_time + 6.0})
                current_time += 6.0
                next_weekly += 168.0
            # Fortnightly tunnel/nitrogen/sauce washing【771340528377493†screenshot】
            while current_time >= next_fortnightly:
                timeline.append({'task': 'Tunnel & nitrogen & sauce tank washing',
                                 'start': current_time, 'end': current_time + 14.0})
                current_time += 14.0
                next_fortnightly += 336.0

        # Insert periodic tasks before processing next SKU
        insert_periodic_tasks()

        if last_sku is not None:
            # Packaging change, including wrapper change, if needed
            pack_time = packaging_change[last_sku][sku]
            if pack_time > 0:
                timeline.append({
                    'task': f'Wrapper and volume change {last_sku} → {sku}',
                    'start': current_time,
                    'end': current_time + pack_time
                })
                current_time += pack_time
                # Check periodic tasks again after packaging change
                insert_periodic_tasks()
            # Extra hour for star codes
            code = tasks[last_sku][sku]
            star_bonus = 1.0 if code.endswith('*') else 0.0
            base_time = base_times[last_sku][sku]
            # Cleaning tasks
            if base_time + star_bonus > 0:
                timeline.append({
                    'task': f'Change‑over cleaning {last_sku} → {sku} (code {code.strip("*")}{" + extra" if star_bonus else ""})',
                    'start': current_time,
                    'end': current_time + base_time + star_bonus
                })
                current_time += base_time + star_bonus
                insert_periodic_tasks()
        # Production run for this SKU
        run_time = processing_time[sku]
        timeline.append({
            'task': f'Produce {sku}',
            'start': current_time,
            'end': current_time + run_time
        })
        current_time += run_time
        last_sku = sku

    # End of production tasks【771340528377493†screenshot】
    timeline.append({'task': 'Planned sanitisation after end of production',
                     'start': current_time, 'end': current_time + 6.0})
    current_time += 6.0
    timeline.append({'task': 'End of production procedures', 'start': current_time,
                     'end': current_time + 0.5})
    current_time += 0.5
    return timeline


# ---------------------------------------------------------------------------
# Resource‑aware simulation
#
# The functions below extend the base schedule simulation to consider
# additional constraints on the utilisation of process resources such as
# chocolate tubs, buffers and coating systems.  Each resource has a
# maximum continuous run time (``max_run_time``) before it must be
# cleaned, a cleaning duration (``clean_time``) and an optional maximum
# storage time when idle (``max_storage_time``) before it must be
# rinsed or flushed prior to reuse.  These values are taken from the
# constraints provided by the user (see the prompt for details).  If a
# SKU requires a resource and the resource has exceeded its run or
# storage limit, the schedule will insert the appropriate cleaning
# operation before the production run.

def simulate_schedule_duration_with_resources(
    order: List[int],
    sku_order: List[str],
    scale: float,
    demand: Dict[str, float],
    rate: float,
    base_times: Dict[str, Dict[str, float]],
    tasks: Dict[str, Dict[str, str]],
    packaging_change: Dict[str, Dict[str, float]],
    resource_map: Dict[str, Iterable[str]],
    resources: Dict[str, Resource],
    include_pre_post: bool = True
) -> float:
    """Compute the total duration of a schedule, including resource cleanings.

    This function is analogous to ``simulate_schedule_duration`` but
    additionally tracks the utilisation of each production resource.

    Parameters
    ----------
    order : list of int
        Indices representing the order of SKUs as returned by the solver.
    sku_order : list of str
        List mapping indices to SKU codes.
    scale : float
        Scaling factor applied to all demands (0 ≤ scale ≤ 1).
    demand : dict
        Base demand (in units) for each SKU.
    rate : float
        Production rate in units per hour.
    base_times : dict
        Base cleaning times between SKUs.
    tasks : dict
        Cleaning activity codes.
    packaging_change : dict
        Additional time required when switching between wrapper sizes.
    resource_map : dict
        Mapping from SKU codes to the set of resources it uses during
        production.  For example, ``{'B3': ['chocolate_tub', 'chocolate_buffer']}``.
    resources : dict
        Dictionary of ``Resource`` objects keyed by resource name.
    include_pre_post : bool, optional
        Whether to include pre‑production and post‑production sanitisation.

    Returns
    -------
    float
        Total time in hours required to run the schedule with the
        specified scale and resource constraints.
    """
    current_time = 0.0
    # Pre‑production sanitisation and production start
    if include_pre_post:
        current_time += 5.0  # planned sanitisation before start
        current_time += 1.0  # production start

    # Track periodic maintenance thresholds (daily, weekly, fortnightly)
    next_daily = 24.0
    next_weekly = 168.0
    next_fortnightly = 336.0

    # Track resource utilisation: run time since last cleaning,
    # last time the resource was active and last cleaning time.
    res_run_time: Dict[str, float] = {r: 0.0 for r in resources}
    res_last_active: Dict[str, float] = {r: 0.0 for r in resources}
    res_last_clean: Dict[str, float] = {r: 0.0 for r in resources}

    def check_periodic() -> None:
        """Insert routine maintenance based on accumulated time."""
        nonlocal current_time, next_daily, next_weekly, next_fortnightly
        # Daily routine sanitisation
        while current_time >= next_daily:
            current_time += 1.0
            next_daily += 24.0
        # Weekly nitrogen bath cleaning and AM/PM checks
        while current_time >= next_weekly:
            current_time += 4.0
            current_time += 6.0
            next_weekly += 168.0
        # Fortnightly tunnel/nitrogen/sauce washing
        while current_time >= next_fortnightly:
            current_time += 14.0
            next_fortnightly += 336.0

    last_sku: Optional[str] = None
    for idx in order:
        sku = sku_order[idx]
        # Before any change‑over, apply periodic tasks
        check_periodic()

        if last_sku is not None:
            # Packaging change
            pack = packaging_change[last_sku][sku]
            if pack > 0:
                current_time += pack
                check_periodic()
            # Change‑over cleaning: base time plus star bonus
            code = tasks[last_sku][sku]
            extra = 1.0 if code.endswith('*') else 0.0
            current_time += base_times[last_sku][sku] + extra
            check_periodic()

        # Production run with resource constraints
        run_time = (demand[sku] * scale) / rate
        # Determine which resources the SKU uses
        used_resources = list(resource_map.get(sku, []))
        # Before starting production, handle resource cleaning if needed
        for res_name in used_resources:
            res = resources[res_name]
            # Check idle duration
            idle_time = current_time - res_last_active[res_name]
            if idle_time > res.max_storage_time:
                # Storage time exceeded; rinse or clean resource
                current_time += res.clean_time
                res_run_time[res_name] = 0.0
                res_last_clean[res_name] = current_time
                check_periodic()
            # Check continuous run limit
            # If current run would exceed max run time, clean first
            if res_run_time[res_name] + run_time > res.max_run_time:
                current_time += res.clean_time
                res_run_time[res_name] = 0.0
                res_last_clean[res_name] = current_time
                check_periodic()
        # Now produce the SKU
        current_time += run_time
        # Update resource usage metrics
        for res_name in used_resources:
            res_run_time[res_name] += run_time
            res_last_active[res_name] = current_time
        last_sku = sku

    # Post‑production sanitisation and shutdown
    if include_pre_post:
        current_time += 6.0  # planned sanitisation after production
        current_time += 0.5  # end of production procedures
    return current_time


def find_max_scale_with_resources(
    time_horizon: float,
    order: List[int],
    sku_order: List[str],
    demand: Dict[str, float],
    rate: float,
    base_times: Dict[str, Dict[str, float]],
    tasks: Dict[str, Dict[str, str]],
    packaging_change: Dict[str, Dict[str, float]],
    resource_map: Dict[str, Iterable[str]],
    resources: Dict[str, Resource]
) -> Tuple[float, float]:
    """Binary search for the maximum production scale respecting resources.

    This function extends ``find_max_scale`` to include resource
    constraints.  Given a fixed production order and a time horizon
    (e.g. six weeks), it finds the largest fraction of the nominal
    demand that can be produced within that horizon while satisfying
    change‑over, packaging, sanitisation and resource cleaning rules.

    Parameters
    ----------
    time_horizon : float
        Total available time in hours.
    order : list of int
        Order returned by the solver.
    sku_order : list of str
        Mapping of indices to SKU codes.
    demand : dict
        Nominal demand for each SKU.
    rate : float
        Production rate (units per hour).
    base_times, tasks, packaging_change : dict
        Change‑over times, cleaning codes and packaging change times.
    resource_map : dict
        Mapping from SKU to resources used.
    resources : dict
        Dictionary of ``Resource`` objects.

    Returns
    -------
    (scale, total_bars)
        Fraction of demand that fits and the total number of bars produced.
    """
    low, high = 0.0, 1.0
    best_scale = 0.0
    for _ in range(30):  # search precision
        mid = (low + high) / 2
        total_time = simulate_schedule_duration_with_resources(
            order, sku_order, mid, demand, rate,
            base_times, tasks, packaging_change,
            resource_map, resources, include_pre_post=True
        )
        if total_time <= time_horizon + 1e-6:
            best_scale = mid
            low = mid
        else:
            high = mid
    total_bars = best_scale * sum(demand.values())
    return best_scale, total_bars


def max_bars_across_orders(
    time_horizon: float,
    possible_orders: List[List[int]],
    sku_order: List[str],
    demand: Dict[str, float],
    rate: float,
    base_times: Dict[str, Dict[str, float]],
    tasks: Dict[str, Dict[str, str]],
    packaging_change: Dict[str, Dict[str, float]],
    resource_map: Dict[str, Iterable[str]],
    resources: Dict[str, Resource]
) -> Tuple[List[int], float, float]:
    """Evaluate multiple production orders to maximise bars in a time horizon.

    This helper iterates through a list of candidate SKU orderings and
    computes the maximum number of bars that can be produced for each
    order within the specified time horizon using
    ``find_max_scale_with_resources``.  It returns the best order along
    with the corresponding production scale and bar count.

    Parameters
    ----------
    time_horizon : float
        Available production time (hours).
    possible_orders : list of list of int
        List of candidate production sequences, each expressed as a list
        of indices into ``sku_order``.  Typically this contains the
        optimal order from the solver and perhaps a few heuristically
        generated alternatives.  Enumerating all permutations is
        impractical for large numbers of SKUs.
    sku_order : list of str
        Mapping from index to SKU code.
    demand, rate, base_times, tasks, packaging_change : see above.
    resource_map, resources : see above.

    Returns
    -------
    (best_order, best_scale, max_bars)
        ``best_order`` is the sequence yielding the maximum number of
        bars, ``best_scale`` is the fraction of the nominal demand
        produced and ``max_bars`` the corresponding count of bars.
    """
    best_bars = -1.0
    best_order = possible_orders[0] if possible_orders else []
    best_scale = 0.0
    for order in possible_orders:
        scale, bars = find_max_scale_with_resources(
            time_horizon, order, sku_order, demand, rate,
            base_times, tasks, packaging_change, resource_map, resources
        )
        if bars > best_bars:
            best_bars = bars
            best_order = order
            best_scale = scale
    return best_order, best_scale, best_bars


def estimate_time_for_equal_bars(
    equal_bars: float,
    order: List[int],
    sku_order: List[str],
    rate: float,
    base_times: Dict[str, Dict[str, float]],
    tasks: Dict[str, Dict[str, str]],
    packaging_change: Dict[str, Dict[str, float]],
    resource_map: Dict[str, Iterable[str]],
    resources: Dict[str, Resource],
    include_pre_post: bool = True
) -> float:
    """Estimate the schedule duration to produce the same number of bars per category.

    This helper sets the base demand for each SKU to 1 unit and then
    scales it by ``equal_bars``.  The effect is that each SKU will
    produce ``equal_bars`` units.  The function returns the total time
    (in hours) required to complete the schedule under the resource
    constraints.

    Parameters
    ----------
    equal_bars : float
        Number of bars to produce for each SKU (category).
    order : list of int
        Production order indices.
    sku_order : list of str
        Mapping indices to SKU codes.
    rate, base_times, tasks, packaging_change : see other functions.
    resource_map, resources : see other functions.
    include_pre_post : bool, optional
        Whether to include pre‑production and post‑production sanitisation.

    Returns
    -------
    float
        Total time required (hours) to produce ``equal_bars`` bars per
        category.
    """
    # Create a base demand of 1 for each SKU so that scale corresponds to bars
    base_demand = {sku: 1.0 for sku in sku_order}
    return simulate_schedule_duration_with_resources(
        order, sku_order, equal_bars, base_demand, rate,
        base_times, tasks, packaging_change, resource_map, resources,
        include_pre_post=include_pre_post
    )


def max_equal_bars_for_time(
    time_horizon: float,
    order: List[int],
    sku_order: List[str],
    rate: float,
    base_times: Dict[str, Dict[str, float]],
    tasks: Dict[str, Dict[str, str]],
    packaging_change: Dict[str, Dict[str, float]],
    resource_map: Dict[str, Iterable[str]],
    resources: Dict[str, Resource]
) -> Tuple[float, float]:
    """Compute the maximum equal bars per category producible in a time horizon.

    This function uses ``find_max_scale_with_resources`` with a base demand
    of 1 for each SKU so that the scale returned equals the number of bars
    per category.  It returns both the bars per category and the total
    number of bars across all SKUs (bars_per_category × number_of_skus).

    Parameters
    ----------
    time_horizon : float
        Available time in hours.
    order : list of int
        Production order indices.
    sku_order : list of str
        Mapping indices to SKU codes.
    rate, base_times, tasks, packaging_change : see other functions.
    resource_map, resources : see other functions.

    Returns
    -------
    (bars_per_category, total_bars)
        Maximum bars per category (possibly fractional) that fit in
        ``time_horizon`` and the corresponding total bars across
        categories.
    """
    base_demand = {sku: 1.0 for sku in sku_order}
    scale, _ = find_max_scale_with_resources(
        time_horizon, order, sku_order, base_demand, rate,
        base_times, tasks, packaging_change, resource_map, resources
    )
    bars_per_category = scale
    total_bars = bars_per_category * len(sku_order)
    return bars_per_category, total_bars


def simulate_schedule_duration(order: List[int], sku_order: List[str],
                               scale: float,
                               demand: Dict[str, float],
                               rate: float,
                               base_times: Dict[str, Dict[str, float]],
                               tasks: Dict[str, Dict[str, str]],
                               packaging_change: Dict[str, Dict[str, float]],
                               include_pre_post: bool = True) -> float:
    """Compute the total duration of a schedule for a scaled demand.

    Parameters
    ----------
    order : list of int
        Indices representing the order of SKUs as returned by the solver.
    sku_order : list of str
        List mapping indices to SKU codes.
    scale : float
        Scaling factor applied to all demands (0 ≤ scale ≤ 1).  If
        ``scale=1``, the full demand is produced; if ``scale=0.5``, half of
        each demand is produced, etc.
    demand : dict
        The base demand (in units) for each SKU.
    rate : float
        Production rate in units per hour.
    base_times : dict
        Base cleaning times between SKUs.
    tasks : dict
        Cleaning activity codes between SKUs.
    packaging_change : dict
        Additional time required when switching between wrapper sizes.
    include_pre_post : bool
        Whether to include pre‑production and post‑production sanitisation.

    Returns
    -------
    float
        Total time in hours required to run the schedule with the
        specified scale.  This function accounts for routine daily
        sanitisation (every 24 h), weekly tasks (every 168 h) and
        fortnightly tasks (every 336 h) in the same way as
        ``generate_timeline``.  It does not produce the event list; it
        simply accumulates time.
    """
    current_time = 0.0
    # Pre‑production sanitisation and production start
    if include_pre_post:
        current_time += 5.0  # planned sanitisation before start【771340528377493†screenshot】
        current_time += 1.0  # production start【771340528377493†screenshot】

    # Periodic task thresholds
    next_daily = 24.0
    next_weekly = 168.0
    next_fortnightly = 336.0

    last_sku: Optional[str] = None
    for idx in order:
        sku = sku_order[idx]
        # Insert periodic tasks before change‑over or production
        def check_periodic() -> None:
            nonlocal current_time, next_daily, next_weekly, next_fortnightly
            # daily
            while current_time >= next_daily:
                current_time += 1.0
                next_daily += 24.0
            # weekly
            while current_time >= next_weekly:
                current_time += 4.0  # nitrogen bath cleaning【771340528377493†screenshot】
                current_time += 6.0  # AM/PM cleaning【771340528377493†screenshot】
                next_weekly += 168.0
            # fortnightly
            while current_time >= next_fortnightly:
                current_time += 14.0  # tunnel & nitrogen & sauce tank washing【771340528377493†screenshot】
                next_fortnightly += 336.0

        check_periodic()
        if last_sku is not None:
            # packaging change
            pack = packaging_change[last_sku][sku]
            if pack > 0:
                current_time += pack
                check_periodic()
            # star bonus
            code = tasks[last_sku][sku]
            extra = 1.0 if code.endswith('*') else 0.0
            current_time += base_times[last_sku][sku] + extra
            check_periodic()
        # production time
        run_time = (demand[sku] * scale) / rate
        current_time += run_time
        last_sku = sku
    # Post‑production sanitisation and shutdown
    if include_pre_post:
        current_time += 6.0  # planned sanitisation after production【771340528377493†screenshot】
        current_time += 0.5  # end of production procedures【771340528377493†screenshot】
    return current_time

import csv, random
from typing import List, Dict, Any, Iterable

def sample_random_orders(order_indices: List[int], k: int, seed: int = 42) -> List[List[int]]:
    """Return up to k unique random permutations of order_indices (identity first)."""
    rng = random.Random(seed)
    seen = set()
    orders = []

    ident = tuple(order_indices)
    seen.add(ident)
    orders.append(list(ident))

    attempts, cap_attempts = 0, 50 * k
    while len(orders) < k and attempts < cap_attempts:
        perm = order_indices[:]
        rng.shuffle(perm)
        key = tuple(perm)
        if key not in seen:
            seen.add(key)
            orders.append(perm)
        attempts += 1
    return orders


def _build_timeline_for_equal_bars(
    order: List[int],
    sku_order: List[str],
    equal_bars: float,
    rate: float,
    base_times: Dict[str, Dict[str, float]],
    tasks: Dict[str, Dict[str, str]],
    packaging_change: Dict[str, Dict[str, float]],
    resource_map: Dict[str, Iterable[str]],
    resources: Dict[str, Any],
    include_pre_post: bool = True,
) -> List[Dict[str, Any]]:
    """
    Produce a detailed timeline for equal bars per SKU.
    Uses resource-aware timeline if available; otherwise falls back to generate_timeline.
    """
    # Prefer resource-aware if you have it
    if 'generate_resource_aware_timeline' in globals():
        return generate_resource_aware_timeline(
            order=order,
            sku_order=sku_order,
            bars_per_category=equal_bars,
            rate=rate,
            base_times=base_times,
            tasks=tasks,
            packaging_change=packaging_change,
            resource_map=resource_map,
            resources=resources,
            include_pre_post=include_pre_post
        )
    # Fallback to simple timeline
    demand = {sku: equal_bars for sku in sku_order}
    processing_time = compute_processing_times(demand, rate)
    return generate_timeline(order, sku_order, processing_time, base_times, tasks, packaging_change)


def _transition_durations_after_each_product(
    order: List[int],
    sku_order: List[str],
    timeline: List[Dict[str, Any]],
) -> Dict[str, float]:
    """
    For each adjacent pair in the order, compute time from end of 'Produce SKU_i'
    to start of 'Produce SKU_{i+1}' by scanning the timeline.
    Returns a dict of columns like 'after_<SKUi>_to_<SKUj>_transition_h' -> hours.
    """
    # Index first 'Produce X' event start/end by SKU in displayed order (first occurrence)
    produce_spans: Dict[str, List[Dict[str, float]]] = {}
    for ev in timeline:
        if isinstance(ev.get('task'), str) and ev['task'].startswith('Produce '):
            sku = ev['task'].split(' ', 1)[1]
            produce_spans.setdefault(sku, []).append({'start': ev['start'], 'end': ev['end']})

    cols: Dict[str, float] = {}
    for i in range(len(order) - 1):
        sku_i = sku_order[order[i]]
        sku_j = sku_order[order[i+1]]
        # take the first production event of i and j in the timeline
        if not produce_spans.get(sku_i) or not produce_spans.get(sku_j):
            cols[f'after_{sku_i}_to_{sku_j}_transition_h'] = 0.0
            continue
        end_i = produce_spans[sku_i][0]['end']
        start_j = produce_spans[sku_j][0]['start']
        cols[f'after_{sku_i}_to_{sku_j}_transition_h'] = max(0.0, float(start_j) - float(end_i))
    return cols


def _summarize_states_or_zero(timeline: List[Dict[str, Any]]) -> Dict[str, float]:
    """Use summarize_timeline if present, else return zeros for known buckets."""
    if 'summarize_timeline' in globals():
        return summarize_timeline(timeline)
    return {k: 0.0 for k in [
        "PreProduction","Production","Changeover","Packaging",
        "RoutineSan","WeeklySan","FortnightlySan","ResourceCleaning",
        "PostProduction","Other"
    ]}


def grid_search_equal_bars_to_csv(
    bars_grid: List[float],
    order_candidates: List[List[int]],
    sku_order: List[str],
    production_rate: float,
    base_times: Dict[str, Dict[str, float]],
    tasks: Dict[str, Dict[str, str]],
    packaging_change: Dict[str, Dict[str, float]],
    resource_map: Dict[str, Iterable[str]],
    resources: Dict[str, Any],
    summary_csv_path: str = "grid_equal_bars_summary.csv",
    events_csv_path: str = "grid_equal_bars_events.csv",
) -> None:
    """For each bars value and candidate order, compute total time + timeline; write two CSVs."""
    summary_headers = [
        "bars_per_category",
        "schedule_id",
        "sequence",
        "total_hours",
        "PreProduction","Production","Changeover","Packaging",
        "RoutineSan","WeeklySan","FortnightlySan","ResourceCleaning",
        "PostProduction","Other"
    ]
    events_headers = ["bars_per_category","schedule_id","step_index","task","start_h","end_h","duration_h"]

    with open(summary_csv_path, "w", newline="") as fsum, open(events_csv_path, "w", newline="") as fevt:
        sw = csv.writer(fsum)
        ew = csv.writer(fevt)
        sw.writerow(summary_headers)
        ew.writerow(events_headers)

        for bars in bars_grid:
            for sid, order in enumerate(order_candidates, start=1):
                # total time (fast path)
                total_h = estimate_time_for_equal_bars(
                    equal_bars=bars,
                    order=order,
                    sku_order=sku_order,
                    rate=production_rate,
                    base_times=base_times,
                    tasks=tasks,
                    packaging_change=packaging_change,
                    resource_map=resource_map,
                    resources=resources,
                    include_pre_post=True
                )

                # full timeline (step-by-step)
                timeline = _build_timeline_for_equal_bars(
                    order, sku_order, bars, production_rate,
                    base_times, tasks, packaging_change, resource_map, resources, include_pre_post=True
                )
                states = _summarize_states_or_zero(timeline)
                # transition columns after every product (pairwise)
                trans_cols = _transition_durations_after_each_product(order, sku_order, timeline)

                # summary row (fixed columns)
                row = [
                    bars,
                    sid,
                    " -> ".join(sku_order[i] for i in order),
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
                # append the per-transition columns **after** the fixed product info
                # (order-specific column names)
                for i in range(len(order) - 1):
                    sku_i = sku_order[order[i]]
                    sku_j = sku_order[order[i+1]]
                    colname = f'after_{sku_i}_to_{sku_j}_transition_h'
                    # ensure a consistent column presence by writing value in same order
                    row.append(round(trans_cols.get(colname, 0.0), 3))

                sw.writerow(row)

                # events CSV
                for j, ev in enumerate(timeline):
                    ew.writerow([
                        bars, sid, j, ev.get("task",""),
                        round(float(ev["start"]), 3),
                        round(float(ev["end"]), 3),
                        round(float(ev["end"]) - float(ev["start"]), 3),
                    ])


def find_max_scale(time_horizon: float, order: List[int], sku_order: List[str],
                   demand: Dict[str, float], rate: float,
                   base_times: Dict[str, Dict[str, float]], tasks: Dict[str, Dict[str, str]],
                   packaging_change: Dict[str, Dict[str, float]]) -> Tuple[float, float]:
    """Determine the largest scaling factor to fit within a time horizon.

    Parameters
    ----------
    time_horizon : float
        The total time available (in hours).
    order : list of int
        The SKU sequence returned by the solver.
    sku_order : list of str
        Mapping from indices to SKU codes.
    demand : dict
        Base demand for each SKU.
    rate : float
        Production rate in units per hour.
    base_times : dict
        Base change‑over times.
    tasks : dict
        Cleaning activity codes.
    packaging_change : dict
        Packaging change times.

    Returns
    -------
    (scale, total_bars)
        scale is the fraction (0–1) of the demand that can be produced
        within ``time_horizon``.  total_bars is the maximum number of
        bars produced, i.e. ``scale * sum(demand.values())``.
    """
    # Binary search on scaling factor
    low = 0.0
    high = 1.0
    best_scale = 0.0
    for _ in range(30):  # 2^-30 precision
        mid = (low + high) / 2
        total_time = simulate_schedule_duration(order, sku_order, mid, demand, rate,
                                                base_times, tasks, packaging_change,
                                                include_pre_post=True)
        if total_time <= time_horizon + 1e-6:
            best_scale = mid
            low = mid
        else:
            high = mid
    total_bars = best_scale * sum(demand.values())
    return best_scale, total_bars


def main() -> None:
    """Demonstration of the extended scheduling problem."""
    skus = build_skus()
    base_times, task_codes = build_changeover_and_tasks()
    tasks_description = build_tasks_description()
    packaging_change = build_packaging_change_matrix(skus)

    # Example demand and production rate.  Adjust these values to reflect
    # the quantities you wish to produce and the machine throughput.
    demand = {
        'B3': 2000, 'B4': 2500, 'Bm': 3000,
        'CAB': 1500, 'S3': 2000, 'S4': 2000, 'Sm': 2500,
        'C': 2200, 'Ch3': 1800, 'Ch4': 1900, 'Hm': 1300
    }
    production_rate = 1200.0  # units per hour

    # Compute processing times.
    processing_time = compute_processing_times(demand, production_rate)

    # Determine the order of SKUs using OR‑Tools.
    sku_order = list(skus.keys())  # preserve definition order
    cost_matrix = build_cost_matrix(sku_order, base_times, task_codes,
                                    packaging_change, processing_time)
    order_indices = solve_schedule(cost_matrix)
    if order_indices is None:
        print('No feasible order found')
        return
    ordered_skus = [sku_order[i] for i in order_indices]
    print('Optimal production order:')
    print(' -> '.join(ordered_skus))

    # Generate detailed timeline without resource constraints.
    timeline = generate_timeline(order_indices, sku_order, processing_time,
                                 base_times, task_codes, packaging_change)
    print('\nDetailed schedule (without resource constraints):')
    for event in timeline:
        start = event['start']
        end = event['end']
        task = event['task']
        # Use \u00a0 (non‑breaking space) in the format to preserve spacing in output
        print(f"{start:7.2f}\u00a0h to {end:7.2f}\u00a0h : {task}")
    # List heavy allergens to emphasise they should run last
    heavy_allergens = ['Z', 'D', 'Se']
    print('\nHeavy allergen categories:', ', '.join(heavy_allergens))

    # Print cleaning descriptions legend
    print('\nCleaning activities legend:')
    for code, desc in tasks_description.items():
        print(f"{code}: {desc}")

    # ------------------------------------------------------------------
    # Resource constraint demonstration
    # Define illustrative resources.  Adjust run times and cleaning durations
    # to reflect your equipment.  Times are in hours.  The max_run_time
    # parameter enforces a mandatory cleaning after a resource has been in
    # continuous use for that many hours.  max_storage_time is the maximum
    # idle time before a rinse is required.
    resources: Dict[str, Resource] = {
        'chocolate_tub': Resource('chocolate_tub', max_run_time=72.0, clean_time=1.0, max_storage_time=21 * 24),
        'chocolate_buffer': Resource('chocolate_buffer', max_run_time=72.0, clean_time=1.0, max_storage_time=21 * 24),
        'coating_tub': Resource('coating_tub', max_run_time=72.0, clean_time=1.0, max_storage_time=21 * 24),
        'coating_buffer': Resource('coating_buffer', max_run_time=72.0, clean_time=1.0, max_storage_time=21 * 24),
        'caramel_tub': Resource('caramel_tub', max_run_time=14 * 24, clean_time=3.0, max_storage_time=14 * 24),
        'caramel_buffer': Resource('caramel_buffer', max_run_time=10 * 24, clean_time=3.0, max_storage_time=10 * 24),
        'chocolate_tank': Resource('chocolate_tank', max_run_time=0.0, clean_time=1.0, max_storage_time=10 * 24),
    }
    # Assign resources to each SKU.  All SKUs use chocolate and coating
    # systems, while caramel recipes also use caramel resources.
    resource_map: Dict[str, Iterable[str]] = {}
    for sku_code in skus:
        default_res = ['chocolate_tub', 'chocolate_buffer', 'coating_tub', 'coating_buffer']
        if sku_code in ['C', 'CAB']:
            resource_map[sku_code] = default_res + ['caramel_tub', 'caramel_buffer']
        else:
            resource_map[sku_code] = default_res

    # Choose a planning horizon (e.g. six weeks = 1008h)
    time_horizon_hours = 1008.0
    # Determine how much of the nominal demand fits within the horizon
    best_scale, max_bars = find_max_scale_with_resources(
        time_horizon_hours, order_indices, sku_order, demand, production_rate,
        base_times, task_codes, packaging_change, resource_map, resources
    )
    print(f"\nWith resource constraints: fraction of demand producible in {time_horizon_hours} h = {best_scale:.3f}")
    print(f"Total bars producible within time horizon: {max_bars:.0f}")

    # Optionally, test a few alternative sequences (e.g. reverse order)
    alt_order = list(reversed(order_indices))
    best_order, best_order_scale, best_order_bars = max_bars_across_orders(
        time_horizon_hours, [order_indices, alt_order], sku_order, demand, production_rate,
        base_times, task_codes, packaging_change, resource_map, resources
    )
    chosen = 'optimal order' if best_order == order_indices else 'reverse order'
    print(f"\nBest sequence for {time_horizon_hours} horizon (tested orders): {chosen}")
    print(f"Fraction of demand produced: {best_order_scale:.3f}; bars produced: {best_order_bars:.0f}")

    # ------------------------------------------------------------------
    # Demonstrate equal bars per category calculations
    # Suppose we want to know how long it would take to produce 500 bars of
    # each SKU for the optimal sequence.
    equal_bars = 500
    time_needed = estimate_time_for_equal_bars(
        equal_bars, order_indices, sku_order, production_rate,
        base_times, task_codes, packaging_change, resource_map, resources
    )
    print(f"\nTime required to produce {equal_bars} bars per category (optimal order): {time_needed:.2f} h")
    # Generate a few random shuffled permutations
    num_samples = 100000
    random_orders = []

    for _ in range(num_samples):
        shuffled = order_indices.copy()
        random.shuffle(shuffled)
        random_orders.append(shuffled)

    print("Total alternative random orders to evaluate:", len(random_orders))

    # Now compute the time needed for each random order
    for idx, alt_order in enumerate(random_orders):
        time_needed_alt = estimate_time_for_equal_bars(
            equal_bars, list(alt_order), sku_order, production_rate,
            base_times, task_codes, packaging_change, resource_map, resources
        )
        print(f"Time required to produce {equal_bars} bars per category "
              f"(order {idx + 1}) for order {alt_order}: {time_needed_alt:.2f} h")

    # Now compute the maximum bars per category achievable in a 1008h horizon
    bars_per_category, total_bars = max_equal_bars_for_time(
        time_horizon_hours, order_indices, sku_order, production_rate,
        base_times, task_codes, packaging_change, resource_map, resources
    )
    print(f"Maximum bars per category within {time_horizon_hours}: {bars_per_category:.1f} (total bars {total_bars:.0f})")

    # --- evaluate the optimal order quickly for one bars target
    equal_bars = 500
    time_needed = estimate_time_for_equal_bars(
        equal_bars, order_indices, sku_order, production_rate,
        base_times, task_codes, packaging_change, resource_map, resources
    )
    print(f"\nTime required to produce {equal_bars} bars per category (optimal order): {time_needed:.2f} h")

    # --- grid over different bars targets + sample ~10 unique random schedules (identity included)
    bars_grid = [200, 300, 400, 500, 600]  # edit to taste
    order_candidates = sample_random_orders(order_indices, k=10, seed=123)

    # Write two CSVs: summary (incl. per-transition cols) and full step-by-step events
    grid_search_equal_bars_to_csv(
        bars_grid=bars_grid,
        order_candidates=order_candidates,
        sku_order=sku_order,
        production_rate=production_rate,
        base_times=base_times,
        tasks=task_codes,
        packaging_change=packaging_change,
        resource_map=resource_map,
        resources=resources,
        summary_csv_path="grid_equal_bars_summary.csv",
        events_csv_path="grid_equal_bars_events.csv",
    )
    print("CSV written: grid_equal_bars_summary.csv, grid_equal_bars_events.csv")

def grid_search_equal_bars_to_csv(
    bars_grid: List[float],
    order_candidates: List[List[int]],
    sku_order: List[str],
    production_rate: float,
    base_times: Dict[str, Dict[str, float]],
    tasks: Dict[str, Dict[str, str]],
    packaging_change: Dict[str, Dict[str, float]],
    resource_map: Dict[str, Iterable[str]],
    resources: Dict[str, Any],
    summary_csv_path: str = "grid_equal_bars_summary.csv",
    events_csv_path: str = "grid_equal_bars_events.csv",
) -> None:
    """For each bars value and candidate order, compute total time + timeline; write two CSVs."""
    summary_headers = [
        "bars_per_category",
        "schedule_id",
        "sequence",
        "total_hours",
        "PreProduction","Production","Changeover","Packaging",
        "RoutineSan","WeeklySan","FortnightlySan","ResourceCleaning",
        "PostProduction","Other"
    ]
    events_headers = ["bars_per_category","schedule_id","step_index","task","start_h","end_h","duration_h"]

    with open(summary_csv_path, "w", newline="") as fsum, open(events_csv_path, "w", newline="") as fevt:
        sw = csv.writer(fsum)
        ew = csv.writer(fevt)
        sw.writerow(summary_headers)
        ew.writerow(events_headers)

        for bars in bars_grid:
            for sid, order in enumerate(order_candidates, start=1):
                # total time (fast path)
                total_h = estimate_time_for_equal_bars(
                    equal_bars=bars,
                    order=order,
                    sku_order=sku_order,
                    rate=production_rate,
                    base_times=base_times,
                    tasks=tasks,
                    packaging_change=packaging_change,
                    resource_map=resource_map,
                    resources=resources,
                    include_pre_post=True
                )

                # full timeline (step-by-step)
                timeline = _build_timeline_for_equal_bars(
                    order, sku_order, bars, production_rate,
                    base_times, tasks, packaging_change, resource_map, resources, include_pre_post=True
                )
                states = _summarize_states_or_zero(timeline)
                # transition columns after every product (pairwise)
                trans_cols = _transition_durations_after_each_product(order, sku_order, timeline)

                # summary row (fixed columns)
                row = [
                    bars,
                    sid,
                    " -> ".join(sku_order[i] for i in order),
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
                # append the per-transition columns **after** the fixed product info
                # (order-specific column names)
                for i in range(len(order) - 1):
                    sku_i = sku_order[order[i]]
                    sku_j = sku_order[order[i+1]]
                    colname = f'after_{sku_i}_to_{sku_j}_transition_h'
                    # ensure a consistent column presence by writing value in same order
                    row.append(round(trans_cols.get(colname, 0.0), 3))

                sw.writerow(row)

                # events CSV
                for j, ev in enumerate(timeline):
                    ew.writerow([
                        bars, sid, j, ev.get("task",""),
                        round(float(ev["start"]), 3),
                        round(float(ev["end"]), 3),
                        round(float(ev["end"]) - float(ev["start"]), 3),
                    ])

if __name__ == '__main__':
    main()