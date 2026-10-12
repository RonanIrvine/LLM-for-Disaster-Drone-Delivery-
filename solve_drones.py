"""
Standalone drone-delivery planner.

Solves the same problem your generated_solver.py (Z3-based) targets, but with
plain Python search instead of Z3 -- so it needs no API key and no z3-solver
install. Reads a JSON file in the *native* schema used by input.json:

{
  "nodes": [...], "edges": [{"from","to","weight"}, ...],
  "resources_at": {node: qty}, "resources_required": {node: qty},
  "drones": {name: {"capacity","energy"}},
  "timing": {"takeoff","landing","travel","service","t_max"},
  "max_trips": int
}

Model:
- 'weight' on an edge = energy cost to traverse it (one-way).
- source node = whichever node has resources_at > 0.
- A drone's round trip on a path costs 2x the one-way weight sum (there and
  back to source), and that cost is charged every trip regardless of how
  much cargo is carried (matches generated_solver.py's energy accounting).
- Greedily use the cheapest available source->delivery-node path, and load
  the highest-capacity drone first, to minimize total number of trips
  (mirrors the objective in drone_planner.py: minimize energy, then trips).
"""
import json
import sys
from itertools import count


def load_problem(path):
    with open(path) as f:
        data = json.load(f)
    node_set = set(data.get("nodes", []))
    for e in data["edges"]:
        node_set.add(e["from"])
        node_set.add(e["to"])
    data["nodes"] = sorted(node_set)
    return data


def build_graph(edges):
    graph = {}
    for e in edges:
        graph.setdefault(e["from"], {})[e["to"]] = e["weight"]
    return graph


def all_simple_paths(graph, start, goal):
    """DFS over all simple paths start -> goal (fine for small graphs)."""
    paths = []
    stack = [(start, [start], 0)]
    while stack:
        node, path, dist = stack.pop()
        if node == goal:
            paths.append((path, dist))
            continue
        for nxt, w in graph.get(node, {}).items():
            if nxt not in path:
                stack.append((nxt, path + [nxt], dist + w))
    return paths


def solve(data):
    nodes = data["nodes"]
    graph = build_graph(data["edges"])
    resources_at = data["resources_at"]
    resources_required = {k: v for k, v in data["resources_required"].items() if v > 0}
    drones = data["drones"]
    max_trips = data["max_trips"]

    source_candidates = [n for n, q in resources_at.items() if q > 0]
    if len(source_candidates) != 1:
        return {"status": "Unsatisfiable",
                "reason": f"Expected exactly one source node with resources, found {source_candidates}"}
    source = source_candidates[0]

    plan = {k: [] for k in drones}          # k -> list of trip dicts
    energy_used = {k: 0 for k in drones}
    trips_used = {k: 0 for k in drones}
    delivered_total = {n: 0 for n in resources_required}

    # cheapest path per delivery node
    best_path = {}
    for dest, need in resources_required.items():
        paths = all_simple_paths(graph, source, dest)
        if not paths:
            return {"status": "Unsatisfiable",
                    "reason": f"No path from source '{source}' to delivery node '{dest}'"}
        path, dist = min(paths, key=lambda pd: pd[1])
        best_path[dest] = (path, dist)

    # order drones by capacity descending -> minimizes total trip count
    drone_order = sorted(drones.keys(), key=lambda k: -drones[k]["capacity"])

    for dest, need in resources_required.items():
        remaining = need
        path, one_way = best_path[dest]
        round_trip_cost = 2 * one_way
        for k in drone_order:
            if remaining <= 0:
                break
            cap = drones[k]["capacity"]
            energy_budget = drones[k]["energy"]
            while remaining > 0:
                if trips_used[k] >= max_trips:
                    break
                if energy_used[k] + round_trip_cost > energy_budget:
                    break
                deliver = min(cap, remaining)
                trips_used[k] += 1
                energy_used[k] += round_trip_cost
                remaining -= deliver
                delivered_total[dest] += deliver
                plan[k].append({
                    "trip": trips_used[k],
                    "dest": dest,
                    "path": path,
                    "delivered": deliver,
                    "round_trip_energy": round_trip_cost,
                    "energy_after": energy_used[k],
                })
        if remaining > 0:
            return {"status": "Unsatisfiable",
                    "reason": f"Could not deliver required {need} units to '{dest}' "
                              f"({remaining} short) within drone energy/max_trips limits "
                              f"using best path {path} (round-trip cost {round_trip_cost})"}

    return {
        "status": "Satisfiable",
        "source": source,
        "plan": plan,
        "energy_used": energy_used,
        "trips_used": trips_used,
        "delivered_total": delivered_total,
        "resources_required": resources_required,
    }


def format_result(res):
    if res["status"] != "Satisfiable":
        return f"Unsatisfiable\nReason: {res['reason']}"

    lines = ["Satisfiable", f"Source node: {res['source']}", "Best Solution:"]
    for k, trips in res["plan"].items():
        lines.append(f"\nDrone {k}:")
        if not trips:
            lines.append("  No delivery trips")
            continue
        lines.append(f"  Energy used: {res['energy_used'][k]}")
        lines.append(f"  Number of delivery trips: {res['trips_used'][k]}")
        lines.append("")
        lines.append("  Trip  Path                     Delivered  Round-trip Energy  Cumulative Energy")
        lines.append("  " + "-" * 78)
        for t in trips:
            path_disp = "->".join(t["path"])
            lines.append(f"  {t['trip']:<5} {path_disp:<24} {t['delivered']:<10} {t['round_trip_energy']:<18} {t['energy_after']}")

    lines.append("\nOverall summary:")
    lines.append(f"  Total delivered: {res['delivered_total']}")
    lines.append(f"  Total energy used: {res['energy_used']}")
    lines.append(f"  Total trips used: {res['trips_used']}")
    lines.append(f"  Resources required: {res['resources_required']}")
    return "\n".join(lines)


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "input.json"
    data = load_problem(path)
    result = solve(data)
    print(format_result(result))
