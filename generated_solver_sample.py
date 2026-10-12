def run_z3_solver(params):
    from z3 import Optimize, Int, Bool, Sum, If, Or, And, Not, IntVal, sat, ForAll, Implies, Abs
    import time
    from itertools import combinations

    t0 = time.time()
    try:
        # Extract and normalize inputs
        nodes = list(params.get('nodes', []))
        if not nodes:
            return "Unsatisfiable\nReason: No nodes provided\nSolve time: {:.2f} seconds".format(time.time() - t0)
        initial_node = params.get('initial_node', nodes[0])
        if initial_node not in nodes:
            nodes = [initial_node] + [n for n in nodes if n != initial_node]
        node_index = {n: i for i, n in enumerate(nodes)}
        n_nodes = len(nodes)

        resources_at = params.get('resources_at', {})
        resources_required = dict(params.get('resources_required', {}))
        delivery_nodes = list(params.get('delivery_nodes', []))
        # Ensure delivery_nodes are in nodes and deduplicate
        delivery_nodes = [n for n in delivery_nodes if n in nodes]
        if not delivery_nodes:
            # If none provided, infer from resources_required keys
            delivery_nodes = [n for n in resources_required.keys() if n in nodes]

        # Edges: support both tuple keys and "a->b" string keys
        edges_param = dict(params.get('edges', {}))
        edges = {}
        for k, v in edges_param.items():
            if isinstance(k, tuple) and len(k) == 2:
                i, j = k
            elif isinstance(k, str) and '->' in k:
                i, j = k.split('->')
            else:
                continue
            if i in nodes and j in nodes:
                edges[(i, j)] = {'distance': int(v.get('distance', 1)), 'time': int(v.get('time', 1))}
        if not edges:
            return "Unsatisfiable\nReason: No valid edges provided\nSolve time: {:.2f} seconds".format(time.time() - t0)

        # Drones
        drones = dict(params.get('drones', {}))
        if not drones:
            return "Unsatisfiable\nReason: No drones provided\nSolve time: {:.2f} seconds".format(time.time() - t0)

        # Optional parameters
        opt_params = params.get('optional', {}) or {}
        max_trips = int(opt_params.get('max_trips', 6))
        t_max = int(opt_params.get('t_max', 500))
        # Separation/service time used in time propagation and non-overlap
        t_s = int(opt_params.get('t_s', 5))
        # w not used directly; included for completeness
        w = int(opt_params.get('w', 0))

        # Helpers: adjacency lists for faster constraint building
        out_edges = {i: [] for i in nodes}
        in_edges = {i: [] for i in nodes}
        for (i, j), attr in edges.items():
            out_edges[i].append(j)
            in_edges[j].append(i)

        # Model
        opt = Optimize()

        # Variables containers
        X_delivery = {}            # (k, t, i, j) -> Bool
        start_time = {}            # (k, t) -> Int
        arrival_time = {}          # (k, t, i) -> Int
        delivery_time = {}         # (k, t, i) -> Int [delivery nodes only]
        resources_delivered = {}   # (k, t, i) -> Int [delivery nodes only]
        load = {}                  # (k, i) -> Int
        energy_used = {}           # (k) -> Int
        trip_active = {}           # (k, t) -> Bool
        visited_nodes = {}         # (k, t, i) -> Bool
        successor = {}             # (k, t, i) -> Int
        drone_used = {}            # (k) -> Bool

        # Create variables
        for k in sorted(drones.keys()):
            energy_used[k] = Int(f"energy_used[{k}]")
            opt.add(energy_used[k] >= 0)

            drone_used[k] = Bool(f"drone_used[{k}]")

            # Load per node (bounded -100..100)
            for i in nodes:
                load[(k, i)] = Int(f"load[{k},{i}]")
                opt.add(load[(k, i)] >= -100, load[(k, i)] <= 100)

            for t in range(max_trips):
                trip_active[(k, t)] = Bool(f"trip_active[{k},{t}]")
                start_time[(k, t)] = Int(f"start_time[{k},{t}]")
                opt.add(start_time[(k, t)] >= 0)

                for i in nodes:
                    arrival_time[(k, t, i)] = Int(f"arrival_time[{k},{t},{i}]")
                    opt.add(arrival_time[(k, t, i)] >= 0)
                    visited_nodes[(k, t, i)] = Bool(f"visited_nodes[{k},{t},{i}]")
                    successor[(k, t, i)] = Int(f"successor[{k},{t},{i}]")
                    opt.add(successor[(k, t, i)] >= -1, successor[(k, t, i)] <= n_nodes - 1)

                for i in delivery_nodes:
                    delivery_time[(k, t, i)] = Int(f"delivery_time[{k},{t},{i}]")
                    opt.add(delivery_time[(k, t, i)] >= 0)
                    resources_delivered[(k, t, i)] = Int(f"resources_delivered[{k},{t},{i}]")
                    # per-trip delivery to a single node bounded by capacity and non-negative
                    opt.add(resources_delivered[(k, t, i)] >= 0)
                    opt.add(resources_delivered[(k, t, i)] <= int(drones[k].get('capacity', 0)))

                # Edge usage variables
                for (i, j) in edges.keys():
                    X_delivery[(k, t, i, j)] = Bool(f"X_delivery[{k},{t},{i},{j}]")

        # Drone used definition
        for k in sorted(drones.keys()):
            opt.add(drone_used[k] == Or([trip_active[(k, t)] for t in range(max_trips)]))

        # Helper sums
        def sum_bools(bools):
            return Sum([If(b, 1, 0) for b in bools]) if bools else IntVal(0)

        # Flow conservation and visit definitions
        for k in sorted(drones.keys()):
            for t in range(max_trips):
                # Visit definition: visited iff any incident edge used
                for i in nodes:
                    outgoing = [X_delivery[(k, t, i, j)] for j in out_edges[i] if (i, j) in edges]
                    incoming = [X_delivery[(k, t, j, i)] for j in in_edges[i] if (j, i) in edges]
                    opt.add(Implies(visited_nodes[(k, t, i)], Or(outgoing + incoming) if (outgoing + incoming) else False))
                    # If not visited, no incident edges used
                    if outgoing:
                        opt.add(Implies(Not(visited_nodes[(k, t, i)]), sum_bools(outgoing) == 0))
                    if incoming:
                        opt.add(Implies(Not(visited_nodes[(k, t, i)]), sum_bools(incoming) == 0))

                    # Successor linkage
                    succ_opts = []
                    for j in out_edges[i]:
                        if (i, j) in edges:
                            succ_opts.append(And(X_delivery[(k, t, i, j)], successor[(k, t, i)] == IntVal(node_index[j])))
                    if succ_opts:
                        opt.add(Implies(visited_nodes[(k, t, i)], Or(succ_opts)))
                    opt.add(Implies(Not(visited_nodes[(k, t, i)]), successor[(k, t, i)] == -1))

                    # Arrival time bound if visited
                    opt.add(Implies(visited_nodes[(k, t, i)], arrival_time[(k, t, i)] <= IntVal(t_max)))

                # Flow conservation: sum out == sum in for all nodes (including source)
                for i in nodes:
                    sum_out = sum_bools([X_delivery[(k, t, i, j)] for j in out_edges[i] if (i, j) in edges])
                    sum_in = sum_bools([X_delivery[(k, t, j, i)] for j in in_edges[i] if (j, i) in edges])
                    opt.add(sum_out == sum_in)

                # Trip must start and end at source if active
                source = initial_node
                out_from_source = sum_bools([X_delivery[(k, t, source, j)] for j in out_edges[source] if (source, j) in edges])
                into_source = sum_bools([X_delivery[(k, t, j, source)] for j in in_edges[source] if (j, source) in edges])
                opt.add(Implies(trip_active[(k, t)], out_from_source == 1))
                opt.add(Implies(trip_active[(k, t)], into_source == 1))

                # If trip not active then no edges and no visits or deliveries
                all_edges_trip = [X_delivery[(k, t, i, j)] for (i, j) in edges.keys()]
                all_visits_trip = [visited_nodes[(k, t, i)] for i in nodes]
                all_delivs_trip = [resources_delivered[(k, t, i)] for i in delivery_nodes]
                if all_edges_trip:
                    opt.add(Implies(Not(trip_active[(k, t)]), sum_bools(all_edges_trip) == 0))
                if all_visits_trip:
                    opt.add(Implies(Not(trip_active[(k, t)]), sum_bools(all_visits_trip) == 0))
                if all_delivs_trip:
                    opt.add(Implies(Not(trip_active[(k, t)]), Sum(all_delivs_trip) == 0))

                # Each visited non-source node has exactly one outgoing and one incoming edge
                for i in nodes:
                    if i == source:
                        continue
                    sum_out_i = sum_bools([X_delivery[(k, t, i, j)] for j in out_edges[i] if (i, j) in edges])
                    sum_in_i = sum_bools([X_delivery[(k, t, j, i)] for j in in_edges[i] if (j, i) in edges])
                    opt.add(Implies(visited_nodes[(k, t, i)], sum_out_i == 1))
                    opt.add(Implies(visited_nodes[(k, t, i)], sum_in_i == 1))

                # At least one delivery node visited if trip active
                if delivery_nodes:
                    opt.add(Implies(trip_active[(k, t)], sum_bools([visited_nodes[(k, t, i)] for i in delivery_nodes]) >= 1))

                # Start time monotonic and linkage to end of previous trip
                if t + 1 < max_trips:
                    opt.add(start_time[(k, t + 1)] >= start_time[(k, t)] + IntVal(t_s))
                    opt.add(start_time[(k, t + 1)] >= arrival_time[(k, t, source)])

                # Delivery-resource relationship implies visit, and delivery time definition
                for i in delivery_nodes:
                    rd = resources_delivered[(k, t, i)]
                    opt.add(Implies(rd > 0, visited_nodes[(k, t, i)]))
                    opt.add(Implies(rd > 0, delivery_time[(k, t, i)] == arrival_time[(k, t, i)]))
                    opt.add(Implies(rd == 0, delivery_time[(k, t, i)] == 0))

                # Capacity per trip
                opt.add(Sum([resources_delivered[(k, t, i)] for i in delivery_nodes]) <= int(drones[k].get('capacity', 0)))

                # If a trip is active, it must deliver something; otherwise nothing and no edges (already ensured)
                opt.add(Implies(trip_active[(k, t)], Sum([resources_delivered[(k, t, i)] for i in delivery_nodes]) > 0))

                # Equality of visited count and edge count if trip active (as per given constraint)
                all_edges_cnt = sum_bools([X_delivery[(k, t, i, j)] for (i, j) in edges.keys()])
                all_visits_cnt = sum_bools([visited_nodes[(k, t, i)] for i in nodes])
                opt.add(Implies(trip_active[(k, t)], all_visits_cnt == all_edges_cnt))

                # Exactly one return edge into source if active
                opt.add(Implies(trip_active[(k, t)], sum_bools([X_delivery[(k, t, j, source)] for j in in_edges[source] if (j, source) in edges]) == 1))

                # Time propagation:
                # From source to next node: arrival[j] = start + t_e
                for j in out_edges[source]:
                    if (source, j) in edges:
                        opt.add(Implies(X_delivery[(k, t, source, j)],
                                        arrival_time[(k, t, j)] ==
                                        start_time[(k, t)] + IntVal(edges[(source, j)]['time'])))

                # General propagation for used edges
                for (i, j), attr in edges.items():
                    # Skip source->j as handled; but also allow general handling to avoid conflicts. Use piecewise:
                    add_time = attr['time']
                    # Add service time if departing from a delivery node
                    extra = t_s if i in delivery_nodes else 0
                    base_expr = arrival_time[(k, t, i)] + IntVal(add_time + extra)
                    # For i == source, we already set arrival[j] from start_time. Also allow alternative inequality to keep consistent
                    if i == source:
                        # Arrival from source shouldn't include service time
                        opt.add(Implies(X_delivery[(k, t, i, j)],
                                        arrival_time[(k, t, j)] >= start_time[(k, t)] + IntVal(add_time)))
                    else:
                        opt.add(Implies(X_delivery[(k, t, i, j)], arrival_time[(k, t, j)] == base_expr))

        # Avoid multiple visits to same node across trips for a drone (except source)
        for k in sorted(drones.keys()):
            for i in nodes:
                if i == initial_node:
                    continue
                opt.add(Sum([If(visited_nodes[(k, t, i)], 1, 0) for t in range(max_trips)]) <= 1)

        # Inter-drone non-overlapping starts for same t (only if both trips active)
        for (k1, k2) in combinations(sorted(drones.keys()), 2):
            for t in range(max_trips):
                opt.add(Implies(And(trip_active[(k1, t)], trip_active[(k2, t)]),
                                Abs(start_time[(k1, t)] - start_time[(k2, t)]) >= IntVal(t_s)))

        # Energy accounting
        for k in sorted(drones.keys()):
            total_energy_terms = []
            for t in range(max_trips):
                for (i, j), attr in edges.items():
                    total_energy_terms.append(If(X_delivery[(k, t, i, j)], IntVal(attr['distance']), IntVal(0)))
            opt.add(energy_used[k] == Sum(total_energy_terms) if total_energy_terms else IntVal(0))
            # Drone energy limit
            opt.add(energy_used[k] <= int(drones[k].get('energy', 0)))

        # Resource satisfaction across drones and trips
        for i in delivery_nodes:
            tot = []
            for k in sorted(drones.keys()):
                for t in range(max_trips):
                    tot.append(resources_delivered[(k, t, i)])
            req_i = int(resources_required.get(i, 0))
            opt.add(Sum(tot) == IntVal(req_i))

        # Load equations and resource availability at source
        for k in sorted(drones.keys()):
            # source load equals total delivered by this drone
            sum_delivered_k = []
            for t in range(max_trips):
                for i in delivery_nodes:
                    sum_delivered_k.append(resources_delivered[(k, t, i)])
            opt.add(load[(k, initial_node)] == (Sum(sum_delivered_k) if sum_delivered_k else IntVal(0)))
            # delivery node loads
            for i in nodes:
                if i in delivery_nodes:
                    sum_i = [resources_delivered[(k, t, i)] for t in range(max_trips)]
                    opt.add(load[(k, i)] == -Sum(sum_i) if sum_i else load[(k, i)] == 0)
                elif i != initial_node:
                    opt.add(load[(k, i)] == 0)

        # Source resource availability cap
        if initial_node in resources_at:
            cap = int(resources_at.get(initial_node, 0))
            opt.add(Sum([load[(k, initial_node)] for k in sorted(drones.keys())]) <= IntVal(cap))

        # At least one trip if there are requirements
        has_req = False
        for v in resources_required.values():
            if int(v) > 0:
                has_req = True
                break
        if has_req:
            opt.add(Or([trip_active[(k, t)] for k in sorted(drones.keys()) for t in range(max_trips)]))

        # Objective: Minimize total energy, then number of active trips
        total_energy = Sum([energy_used[k] for k in sorted(drones.keys())]) if drones else IntVal(0)
        total_trips = Sum([If(trip_active[(k, t)], 1, 0) for k in sorted(drones.keys()) for t in range(max_trips)])
        opt.minimize(total_energy)
        opt.minimize(total_trips)

        # Attempt to map provided textual constraints (pattern matching)
        # These are mostly already encoded; the following ensures we apply key items if present.
        provided_constraints = params.get('constraints', []) or []
        for c in provided_constraints:
            s = str(c).lower()
            # We already implemented non-neg times, flow, capacity, energy, etc.
            # Recognize some patterns to assert consistency (duplicates are fine).
            if "start_time" in s and ">=" in s and "0" in s:
                for k in sorted(drones.keys()):
                    for t in range(max_trips):
                        opt.add(start_time[(k, t)] >= 0)
            if "arrival_time" in s and ">=" in s and "0" in s:
                for k in sorted(drones.keys()):
                    for t in range(max_trips):
                        for i in nodes:
                            opt.add(arrival_time[(k, t, i)] >= 0)
            if "x_delivery[k,t,i,i] = 0" in s:
                # We do not create self-loops; no-op
                pass
            if "sum_{j} x_delivery[k,t,source,j] = 1" in s:
                for k in sorted(drones.keys()):
                    for t in range(max_trips):
                        opt.add(Implies(trip_active[(k, t)],
                                        sum_bools([X_delivery[(k, t, initial_node, j)] for j in out_edges[initial_node] if (initial_node, j) in edges]) == 1))
            if "|start_time" in s and ">=" in s and ("4" in s or str(t_s) in s):
                for (k1, k2) in combinations(sorted(drones.keys()), 2):
                    for t in range(max_trips):
                        opt.add(Implies(And(trip_active[(k1, t)], trip_active[(k2, t)]),
                                        Abs(start_time[(k1, t)] - start_time[(k2, t)]) >= IntVal(t_s)))
            if "energy_used[k]" in s and "<=" in s:
                for k in sorted(drones.keys()):
                    opt.add(energy_used[k] <= int(drones[k].get('energy', 0)))
            # Additional patterns are already deeply covered.

        # Solve
        res = opt.check()
        solve_time = time.time() - t0
        if res != sat:
            return "Unsatisfiable\nReason: {}\nSolve time: {:.2f} seconds".format(str(res), solve_time)

        m = opt.model()

        def eval_int(x):
            return int(str(m.evaluate(x, model_completion=True)))

        def eval_bool(x):
            return str(m.evaluate(x, model_completion=True)) == 'True'

        # Path extraction
        def find_path(drone, trip):
            if not eval_bool(trip_active[(drone, trip)]):
                return [], []
            path = []
            edges_used = []
            current = initial_node
            visited_steps = 0
            max_steps = len(nodes) * 2
            # Build a quick lookup of chosen edges in this trip
            chosen_out = {}
            for (i, j) in edges.keys():
                if eval_bool(X_delivery[(drone, trip, i, j)]):
                    chosen_out.setdefault(i, []).append(j)
            # Follow a single successor per node (arbitrary if multiple)
            while visited_steps < max_steps:
                visited_steps += 1
                nexts = chosen_out.get(current, [])
                if not nexts:
                    break
                j = nexts[0]
                path.append(f"{current}->{j}")
                edges_used.append((current, j))
                current = j
                if current == initial_node:
                    break
            return path, edges_used

        # Build formatted output
        lines = []
        lines.append("Satisfiable")
        lines.append("Solve time: {:.2f} seconds".format(solve_time))
        lines.append("Best Solution:")

        # Precompute totals
        total_resources_delivered = {i: 0 for i in delivery_nodes}
        for i in delivery_nodes:
            ssum = 0
            for k in sorted(drones.keys()):
                for t in range(max_trips):
                    ssum += eval_int(resources_delivered[(k, t, i)])
            total_resources_delivered[i] = ssum

        total_energy_used = sum(eval_int(energy_used[k]) for k in sorted(drones.keys()))
        total_trips_val = 0

        # For "Required Amount" column tracking remaining after printed trips
        remaining_req = {i: int(resources_required.get(i, 0)) for i in delivery_nodes}

        for k in sorted(drones.keys()):
            lines.append(f"Drone {k}:")
            # Count active trips
            active_trips = [t for t in range(max_trips) if eval_bool(trip_active[(k, t)])]
            if not active_trips:
                lines.append("No delivery trips")
                continue

            lines.append(f"Energy used: {eval_int(energy_used[k])}")
            lines.append(f"Number of delivery trips: {len(active_trips)}")
            lines.append("")
            lines.append("Trip Summary:")
            lines.append("Trip  Path                           Start  End    Delivery Times       EN Used Current EN  Resources Held Required Amount")
            lines.append("---------------------------------------------------------------------------------------------------------------------------------------")

            # Remaining energy tracker for the drone
            drone_energy_budget = int(drones[k].get('energy', 0))
            used_energy_so_far = 0

            for idx, t in enumerate(active_trips, start=1):
                path_strs, edges_used_path = find_path(k, t)
                path_disp = ", ".join(path_strs) if path_strs else "-"
                st = eval_int(start_time[(k, t)])
                endt = eval_int(arrival_time[(k, t, initial_node)])
                # Delivery times display
                deliv_entries = []
                trip_resources_held = 0
                for i in delivery_nodes:
                    amt = eval_int(resources_delivered[(k, t, i)])
                    if amt > 0:
                        dt = eval_int(delivery_time[(k, t, i)])
                        deliv_entries.append(f"{i}@{dt}")
                        trip_resources_held += amt
                deliv_disp = ", ".join(deliv_entries) if deliv_entries else "-"

                # Trip energy used: sum over X in this trip
                trip_energy_terms = 0
                for (i, j), attr in edges.items():
                    if eval_bool(X_delivery[(k, t, i, j)]):
                        trip_energy_terms += int(attr['distance'])
                used_energy_so_far += trip_energy_terms
                current_en = max(0, drone_energy_budget - used_energy_so_far)

                # Required amount remaining after this trip (update remaining_req)
                rem_before = dict(remaining_req)
                for i in delivery_nodes:
                    amt = eval_int(resources_delivered[(k, t, i)])
                    if amt > 0:
                        remaining_req[i] = max(0, remaining_req[i] - amt)
                required_amount_disp = ", ".join([f"{i}({remaining_req[i]})" for i in delivery_nodes])

                # Format fields with basic alignment
                lines.append(f"{idx:<5} {path_disp:<30} {st:<6} {endt:<6} {deliv_disp:<22} {trip_energy_terms:<7} {current_en:<11} {trip_resources_held:<15} {required_amount_disp}")

            total_trips_val += len(active_trips)
            lines.append("")

        # Overall summary
        lines.append("Overall summary:")
        lines.append(f"  Total resources delivered: {total_resources_delivered}")
        lines.append(f"  Total energy used: {total_energy_used}")
        lines.append(f"  Total trips: {total_trips_val}")
        lines.append(f"  Resources required: {resources_required}")

        return "\n".join(lines)

    except Exception as e:
        return "Unsatisfiable\nReason: Solver crashed: {}\nSolve time: {:.2f} seconds".format(str(e), time.time() - t0)
