#!/usr/bin/env python3

"""
icecream_scheduler.py
======================

This script builds and solves a single‑machine scheduling problem for
ice‑cream production. Each SKU (stock keeping unit) must be produced on
a single production line, and switching from one SKU to another
requires a changeover. Changeover times are sequence‑dependent and are
measured in hours. The goal is to find an order of production that
minimizes the total time spent producing all SKUs (including both
processing and changeover times).

The changeover data comes from the Mars changeover matrix provided
in the challenge image【396900509045137†screenshot】. For example, transitioning from
“Magnum Billionaire Standard 3MP” (short code ``B3``) to
“Magnum Billionaire Standard 4MP” (``B4``) takes 2 hours, whereas
transitioning from ``B3`` to a peanut‑containing product like
“Magnum Double Caramel Almond & Billionaire Mini 6MP” (``CAB``) takes
10 hours【396900509045137†screenshot】. Heavy allergens such as peanuts or nuts
require the most intensive cleaning and should therefore be processed
at the end of the schedule.

This script demonstrates how to:

* Build a changeover matrix from a tabulated source.
* Compute processing times from demand and production rates.
* Formulate a Travelling Salesman problem (TSP) using OR‑Tools to
  determine the optimal production sequence.
* Assemble a human‑readable timeline that includes changeovers,
  production runs and periodic maintenance breaks.

It is intended as a starting point. Users should adjust the demand
figures, production rate, maintenance interval and duration as needed.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

# The OR‑Tools routing library is used to solve the sequencing problem.
from ortools.constraint_solver import pywrapcp, routing_enums_pb2


def build_changeover_matrix() -> Tuple[List[str], Dict[str, Dict[str, int]]]:
    """Construct the list of SKUs and a dictionary of changeover times.

    Returns
    -------
    skus : list of str
        Short codes for each SKU in the order used by the solver.
    changeover : dict of dict
        changeover[a][b] gives the hours required to clean the line when
        switching production from ``a`` to ``b``.

    Notes
    -----
    The numerical values here are direct transcriptions of the Mars
    changeover matrix provided in the problem description【396900509045137†screenshot】.
    """

    # Short codes for each SKU, in a consistent order.
    skus = [
        'B3',   # Magnum Billionaire Standard 3MP
        'B4',   # Magnum Billionaire Standard 4MP
        'Bm',   # Magnum Billionaire Mini 6MP
        'CAB',  # Magnum Double Caramel Almond & Billionaire Mini 6MP
        'S3',   # Magnum Double Starchaser 3MP
        'S4',   # Magnum Double Starchaser 4MP
        'Sm',   # Magnum Double Starchaser Mini 6MP
        'C',    # Magnum Double Caramel 4MP
        'Ch3',  # Magnum Double Cherry 3MP
        'Ch4',  # Magnum Double Cherry 4MP
        'Hm'    # Magnum Double Hazelnut Mini 6MP
    ]

    # Initialise the changeover dictionary.
    changeover: Dict[str, Dict[str, int]] = {sku: {} for sku in skus}

    # Billionaire Standard 3MP (B3)
    changeover['B3'].update({
        'B3': 1,  'B4': 2,  'Bm': 2,  'CAB': 10,
        'S3': 13, 'S4': 13, 'Sm': 13,
        'C': 15,  'Ch3': 14, 'Ch4': 14, 'Hm': 14
    })

    # Billionaire Standard 4MP (B4)
    changeover['B4'].update({
        'B3': 2,  'B4': 1,  'Bm': 2,  'CAB': 10,
        'S3': 13, 'S4': 13, 'Sm': 13,
        'C': 15,  'Ch3': 14, 'Ch4': 14, 'Hm': 14
    })

    # Billionaire Mini 6MP (Bm)
    changeover['Bm'].update({
        'B3': 6,  'B4': 6,  'Bm': 1,  'CAB': 10,
        'S3': 13, 'S4': 13, 'Sm': 13,
        'C': 15,  'Ch3': 14, 'Ch4': 14, 'Hm': 14
    })

    # Double Caramel Almond & Billionaire Mini 6MP (CAB)
    changeover['CAB'].update({
        'B3': 14, 'B4': 14, 'Bm': 14, 'CAB': 1,
        'S3': 14, 'S4': 14, 'Sm': 14,
        'C': 16,  'Ch3': 15, 'Ch4': 15, 'Hm': 16
    })

    # Double Starchaser 3MP (S3)
    changeover['S3'].update({
        'B3': 10, 'B4': 10, 'Bm': 10, 'CAB': 10,
        'S3': 1,  'S4': 2,  'Sm': 2,
        'C': 13, 'Ch3': 14, 'Ch4': 14, 'Hm': 15
    })

    # Double Starchaser 4MP (S4)
    changeover['S4'].update({
        'B3': 10, 'B4': 10, 'Bm': 10, 'CAB': 10,
        'S3': 2,  'S4': 1,  'Sm': 2,
        'C': 13, 'Ch3': 14, 'Ch4': 14, 'Hm': 15
    })

    # Double Starchaser Mini 6MP (Sm)
    changeover['Sm'].update({
        'B3': 10, 'B4': 10, 'Bm': 10, 'CAB': 10,
        'S3': 6,  'S4': 6,  'Sm': 1,
        'C': 13, 'Ch3': 14, 'Ch4': 14, 'Hm': 15
    })

    # Double Caramel 4MP (C)
    changeover['C'].update({
        'B3': 11, 'B4': 11, 'Bm': 11, 'CAB': 11,
        'S3': 8,  'S4': 8,  'Sm': 8,
        'C': 1,  'Ch3': 16, 'Ch4': 16, 'Hm': 17
    })

    # Double Cherry 3MP (Ch3)
    changeover['Ch3'].update({
        'B3': 14, 'B4': 14, 'Bm': 14, 'CAB': 14,
        'S3': 14, 'S4': 14, 'Sm': 14,
        'C': 16, 'Ch3': 1,  'Ch4': 2,  'Hm': 15
    })

    # Double Cherry 4MP (Ch4)
    changeover['Ch4'].update({
        'B3': 14, 'B4': 14, 'Bm': 14, 'CAB': 14,
        'S3': 14, 'S4': 14, 'Sm': 14,
        'C': 16, 'Ch3': 2,  'Ch4': 1,  'Hm': 15
    })

    # Double Hazelnut Mini 6MP (Hm)
    changeover['Hm'].update({
        'B3': 16, 'B4': 16, 'Bm': 16, 'CAB': 17,
        'S3': 17, 'S4': 17, 'Sm': 17,
        'C': 18, 'Ch3': 17, 'Ch4': 17, 'Hm': 1
    })

    return skus, changeover


def compute_processing_times(demand: Dict[str, float], rate: float) -> Dict[str, float]:
    """Compute processing times from demand and a production rate."""
    return {sku: demand[sku] / rate for sku in demand}


def build_cost_matrix(skus: List[str], changeover: Dict[str, Dict[str, int]],
                      processing_time: Dict[str, float]) -> List[List[float]]:
    """Build a cost matrix combining changeover and processing times."""
    n = len(skus)
    cost: List[List[float]] = [[0.0] * n for _ in range(n)]
    for i, from_sku in enumerate(skus):
        for j, to_sku in enumerate(skus):
            cost[i][j] = changeover[from_sku][to_sku] + processing_time[to_sku]
    return cost


def solve_schedule(cost_matrix: List[List[float]]) -> List[int] | None:
    """Solve the sequencing problem as a TSP via OR‑Tools."""
    n = len(cost_matrix)
    manager = pywrapcp.RoutingIndexManager(n, 1, 0)
    routing = pywrapcp.RoutingModel(manager)

    def transit_callback(from_index: int, to_index: int) -> int:
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return int(cost_matrix[from_node][to_node] * 1000)

    transit_callback_index = routing.RegisterTransitCallback(transit_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    search_params = pywrapcp.DefaultRoutingSearchParameters()
    search_params.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    search_params.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
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


def compute_timeline(order: List[int], skus: List[str], changeover: Dict[str, Dict[str, int]],
                      processing_time: Dict[str, float], maintenance_interval: float = 24.0,
                      maintenance_duration: float = 2.0) -> List[Dict[str, float | str]]:
    """Generate a chronological schedule including changeovers and maintenance."""
    timeline: List[Dict[str, float | str]] = []
    current_time = 0.0
    time_until_maintenance = maintenance_interval
    last_sku: str | None = None

    for idx in order:
        sku = skus[idx]

        # Insert maintenance if the cumulative time has reached the interval.
        if current_time >= time_until_maintenance:
            timeline.append({
                'task': 'Maintenance',
                'start': current_time,
                'end': current_time + maintenance_duration
            })
            current_time += maintenance_duration
            time_until_maintenance += maintenance_interval

        # Add changeover from the previous SKU to this one.
        if last_sku is not None:
            co = changeover[last_sku][sku]
            timeline.append({
                'task': f'Changeover {last_sku} -> {sku}',
                'start': current_time,
                'end': current_time + co
            })
            current_time += co

        # Add the production run for this SKU.
        p = processing_time[sku]
        timeline.append({
            'task': f'Produce {sku}',
            'start': current_time,
            'end': current_time + p
        })
        current_time += p
        last_sku = sku

    return timeline


def main() -> None:
    """Run a demonstration with example demand and production rate."""
    # Example demand (units to produce) for each SKU. Adjust these values
    # to match your production requirements.
    demand = {
        'B3': 2000,
        'B4': 2500,
        'Bm': 3000,
        'CAB': 1500,
        'S3': 2000,
        'S4': 2000,
        'Sm': 2500,
        'C': 2200,
        'Ch3': 1800,
        'Ch4': 1900,
        'Hm': 1300
    }
    # Production rate in units per hour. Modify according to your machine's throughput.
    production_rate = 120.0

    skus, changeover = build_changeover_matrix()
    processing_time = compute_processing_times(demand, production_rate)
    cost_matrix = build_cost_matrix(skus, changeover, processing_time)

    order = solve_schedule(cost_matrix)
    if order is None:
        print('No feasible schedule found.')
        return

    print('Optimal production order:')
    print(' -> '.join(skus[i] for i in order))

    timeline = compute_timeline(order, skus, changeover, processing_time)
    print('\nDetailed schedule:')
    for event in timeline:
        start = event['start']
        end = event['end']
        task = event['task']
        print(f"{start:.2f} h to {end:.2f} h : {task}")

    # Emphasise the heavy‑allergen SKUs which should appear towards the end of the sequence.
    heavy_allergens = {'CAB', 'Hm'}
    print('\nHeavy allergen SKUs (process last when possible):', ', '.join(sorted(heavy_allergens)))


if __name__ == '__main__':
    main()