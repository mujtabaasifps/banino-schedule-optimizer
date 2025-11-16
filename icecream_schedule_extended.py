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

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any

from ortools.constraint_solver import pywrapcp, routing_enums_pb2


@dataclass
class SKU:
    """Representation of a stock‑keeping unit (SKU)."""
    code: str
    name: str
    allergens: List[str]
    volume_ml: int  # nominal volume of the bar (e.g. 85 or 55)


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
    production_rate = 120.0  # units per hour

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

    # Generate detailed timeline.
    timeline = generate_timeline(order_indices, sku_order, processing_time,
                                 base_times, task_codes, packaging_change)
    print('\nDetailed schedule:')
    for event in timeline:
        start = event['start']
        end = event['end']
        task = event['task']
        print(f"{start:7.2f} h to {end:7.2f} h : {task}")
    # List heavy allergens to emphasise they should run last
    heavy_allergens = ['Z', 'D', 'Se']
    print('\nHeavy allergen categories:', ', '.join(heavy_allergens))

    # Optionally print cleaning descriptions for each encountered code
    print('\nCleaning activities legend:')
    for code, desc in tasks_description.items():
        print(f"{code}: {desc}")


if __name__ == '__main__':
    main()