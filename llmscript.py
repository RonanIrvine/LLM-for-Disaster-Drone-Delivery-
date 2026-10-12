import re
import os
import json
from ctransformers import AutoModelForCausalLM
from collections import deque

try:
    llm = AutoModelForCausalLM.from_pretrained(
        "TheBloke/CodeLlama-7B-Python-GGML",
        model_file="codellama-7b-python.ggmlv3.q4_0.bin",
        model_type="llama",
        gpu_layers=0
    )
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

def parse_conversational_input_with_llm():
    print("Describe your drone delivery problem in natural language, e.g.:")
    print("- Nodes: 'I have 5 nodes, a, b, c, d, e'")
    print("- Connections: 'a->b:5, b->d:6, d->e:4, a->c:3, c->e:5'")
    print("- Resources: 'node a has 20 resources, node e needs 15 resources'")
    print("- Drones: 'drone1 has 3 resource and 200 energy capacity, drone2 has 2 resource and 150 energy capacity'")
    print("- Timing: 'all timings are 5' or 'takeoff time and landing time both 5, travel time is 15, service time is 7, t_max is 1000'")
    print("- Max trips: 'max trips is 20'")
    print("Press Enter twice to finish input.\n")
    lines = []
    while True:
        line = input()
        if line == "":
            if lines and input() == "":
                break
        lines.append(line)
    user_input = " ".join(lines)
    print(f"DEBUG: Raw input: {user_input}")

    prompt = f"""
You are an expert in parsing natural language inputs for a drone delivery optimization problem. Given the user input below, extract the parameters into a Python dictionary with the following structure:
{
  "nodes": ["a", "b", "c", "d", "e"],
  "edges": [{"from": "a", "to": "b", "weight": 5.0}, {"from": "b", "to": "d", "weight": 6.0}, ...],
  "resources_at": {"a": 20, "b": 0, "c": 0, "d": 0, "e": 0},
  "resources_required": {"e": 15},
  "drones": {"drone1": {"capacity": 3, "energy": 200}, "drone2": {"capacity": 2, "energy": 150}},
  "timing": {"takeoff": 5, "landing": 5, "travel": 5, "service": 5, "t_max": 1000},
  "max_trips": 20
}
Rules:
- Extract nodes as a list of strings (e.g., ["a", "b", "c", "d", "e"]).
- Extract edges as a list of dictionaries with "from", "to", and "weight" (float). Include bidirectional edges if not specified (e.g., a->b:5 implies b->a:5 unless contradicted).
- Set resources_at to 0 for nodes not mentioned. Resources are integers.
- resources_required should have only the destination node (e.g., "e") with an integer amount.
- Drones should have a name (e.g., "drone1"), with integer "capacity" and "energy".
- Timing parameters (takeoff, landing, travel, service, t_max) are integers. If the input says "all timings are X," set takeoff, landing, travel, and service to X, and set t_max to 1000 (unless specified otherwise, as t_max is typically large).
- max_trips is an integer.
- Handle variations like "a->b:5" or "a-b:5" or "a to b with weight 5", "drone1" or "first drone(drone1)", "all timings are X", etc.
- If parameters are missing, omit them from the dictionary but ensure valid Python syntax.
- Output only the Python dictionary as a string, without additional text.

Example input: "I have 5 nodes, a,b,c,d,e. With node 'a' having 20 resources and node 'e' requiring 15 resources. Connections are a->b:5, b->d:6, d->e:4, a->c:3, c->e:5, e->d:2, d->b:4, b->a:1, e->c:8, c->a:4. The takeoff time and landing time both 5 unit time. Service time is 7, travel time is 15. t_max is 1000 and max trips is 20. There are 2 drones, first drone(drone1) has 3 resource and 200 energy capacity. drone2 has 2 resource and 150 energy capacity. Both drones also start from node 'a'."
Example output: {
  "nodes": ["a", "b", "c", "d", "e"],
  "edges": [{"from": "a", "to": "b", "weight": 5.0}, {"from": "b", "to": "d", "weight": 6.0}, {"from": "d", "to": "e", "weight": 4.0}, {"from": "a", "to": "c", "weight": 3.0}, {"from": "c", "to": "e", "weight": 5.0}, {"from": "e", "to": "d", "weight": 2.0}, {"from": "d", "to": "b", "weight": 4.0}, {"from": "b", "to": "a", "weight": 1.0}, {"from": "e", "to": "c", "weight": 8.0}, {"from": "c", "to": "a", "weight": 4.0}],
  "resources_at": {"a": 20, "b": 0, "c": 0, "d": 0, "e": 0},
  "resources_required": {"e": 15},
  "drones": {"drone1": {"capacity": 3, "energy": 200}, "drone2": {"capacity": 2, "energy": 150}},
  "timing": {"takeoff": 5, "landing": 5, "travel": 15, "service": 7, "t_max": 1000},
  "max_trips": 20
}

Example input: "I have 4 nodes, a, b, c, d. Node a has 3 resources and node d needs 3 resources. Connections are a->b:8, a->c:4, b->d:3, c->d:10. All timings are 5. Max trips is 20. There are 2 drones, drone1 with resource capacity 2 and energy capacity 20, drone2 with resource capacity 1 and energy capacity 25."
Example output: {
  "nodes": ["a", "b", "c", "d"],
  "edges": [{"from": "a", "to": "b", "weight": 8.0}, {"from": "b", "to": "a", "weight": 8.0}, {"from": "a", "to": "c", "weight": 4.0}, {"from": "c", "to": "a", "weight": 4.0}, {"from": "b", "to": "d", "weight": 3.0}, {"from": "d", "to": "b", "weight": 3.0}, {"from": "c", "to": "d", "weight": 10.0}, {"from": "d", "to": "c", "weight": 10.0}],
  "resources_at": {"a": 3, "b": 0, "c": 0, "d": 0},
  "resources_required": {"d": 3},
  "drones": {"drone1": {"capacity": 2, "energy": 20}, "drone2": {"capacity": 1, "energy": 25}},
  "timing": {"takeoff": 5, "landing": 5, "travel": 5, "service": 5, "t_max": 1000},
  "max_trips": 20
}

User input: {user_input}
"""

    try:
        import ast
        params_str = llm(prompt, max_new_tokens=1500, temperature=0.6, top_p=0.95)
        params = ast.literal_eval(params_str)
        print(f"DEBUG: LLM-parsed parameters: {params}")
        if not isinstance(params, dict):
            raise ValueError("LLM did not return a valid dictionary")
        
        # Ensure all timing parameters are present
        if "timing" in params and not all(k in params["timing"] for k in ["takeoff", "landing", "travel", "service", "t_max"]):
            print("Warning: Incomplete timing parameters. Please specify missing timings.")
            timing_input = input("Enter missing timing parameters (e.g., takeoff:5,landing:5,travel:5,service:5,t_max:1000): ").strip()
            for item in timing_input.split(","):
                match = re.match(r"(\w+):(\d+\.?\d*)", item.strip())
                if match:
                    key, value = match.groups()
                    params["timing"][key] = int(float(value.rstrip('.')))
        
        # Validate t_max for feasibility
        if "timing" in params and params["timing"].get("t_max", 0) < 100:
            print(f"Warning: t_max={params['timing']['t_max']} may be too small. Setting t_max=1000 for feasibility.")
            params["timing"]["t_max"] = 1000

    except Exception as e:
        print(f"Error parsing input with LLM: {e}")
        print("Falling back to manual input for missing parameters")
        params = {
            "nodes": input("Enter nodes (e.g., a,b,c,d,e): ").strip().split(","),
            "edges": [],
            "resources_at": {},
            "resources_required": {},
            "drones": {},
            "timing": {},
            "max_trips": int(input("Enter max trips: ").strip())
        }
        edges_input = input("Enter edges (e.g., a->b:5,b->d:6,...): ").strip()
        for edge in edges_input.split(","):
            match = re.match(r"(\w+)->(\w+):(\d+\.?\d*)", edge.strip())
            if match:
                start, end, weight = match.groups()
                params["edges"].append({"from": start, "to": end, "weight": float(weight.rstrip('.'))})
        res_at_input = input("Enter resources at nodes (e.g., a:20,b:0,c:0,d:0,e:0): ").strip()
        for item in res_at_input.split(","):
            match = re.match(r"(\w+):(\d+\.?\d*)", item.strip())
            if match:
                node, amount = match.groups()
                params["resources_at"][node] = int(float(amount.rstrip('.')))
        req_input = input("Enter required resources (e.g., e:15): ").strip()
        match = re.match(r"(\w+):(\d+\.?\d*)", req_input.strip())
        if match:
            node, amount = match.groups()
            params["resources_required"][node] = int(float(amount.rstrip('.')))
        drones_input = input("Enter drones (e.g., drone1:3:200,drone2:2:150): ").strip()
        for drone in drones_input.split(","):
            match = re.match(r"(\w+):(\d+\.?\d*)\s*[,;]\s*(\d+\.?\d*)", drone.strip())
            if match:
                name, capacity, energy = match.groups()
                params["drones"][name] = {
                    "capacity": int(float(capacity.rstrip('.'))),
                    "energy": int(float(energy.rstrip('.')))
                }
        timing_input = input("Enter timing (e.g., takeoff:5,landing:5,travel:5,service:5,t_max:1000): ").strip()
        for item in timing_input.split(","):
            match = re.match(r"(\w+):(\d+\.?\d*)", item.strip())
            if match:
                key, value = match.groups()
                params["timing"][key] = int(float(value.rstrip('.')))
    
    return params

def parse_conversational_input_regex():
    # Fallback regex-based parser (same as original, for brevity)
    user_input = input("Enter input again for fallback parsing: ").lower()
    params = {
        "nodes": [],
        "edges": [],
        "resources_at": {},
        "resources_required": {},
        "drones": {},
        "timing": {},
        "max_trips": None
    }

    # nodes
    node_match = re.search(r"(?:\d+\s*nodes?)\s*[,:=]?\s*([a-z\s,]+)", user_input, re.IGNORECASE)
    if node_match:
        nodes_str = node_match.group(1).replace("and", ",").replace(" ", "")
        params["nodes"] = [n.strip() for n in nodes_str.split(",") if n.strip()]
    if not params["nodes"]:
        nodes_input = input("Enter nodes (comma-separated, e.g., a,b,c,d): ").strip()
        params["nodes"] = [n.strip() for n in nodes_input.split(",")]

    # edges
    edge_pattern = r"(\w+)\s*(?:to|->)\s*(\w+)\s*(?::|with\s*weight)\s*(\d+\.?\d*)"
    edges = re.findall(edge_pattern, user_input, re.IGNORECASE)
    for start, end, weight in edges:
        params["edges"].append({"from": start.strip(), "to": end.strip(), "weight": float(weight.rstrip('.'))})
    for start, end, weight in edges:
        if not any(ed["from"] == end and ed["to"] == start for ed in params["edges"]):
            params["edges"].append({"from": end.strip(), "to": start.strip(), "weight": float(weight.rstrip('.'))})
    if not params["edges"]:
        edges_input = input("Enter edges (e.g., a->b:8,a->c:4,...): ").strip()
        edge_pattern = r"(\w+)->(\w+):(\d+\.?\d*)"
        for edge in edges_input.split(","):
            match = re.match(edge_pattern, edge.strip())
            if match:
                start, end, weight = match.groups()
                params["edges"].append({"from": start, "to": end, "weight": float(weight.rstrip('.'))})

    # resources_at
    resource_at_pattern = r"(?:node\s+)?['\"]?(\w+)['\"]?\s*ha(?:s|ving)\s*(\d+\.?\d*)\s*(?:unit|resource|resources?)"
    resources = re.findall(resource_at_pattern, user_input, re.IGNORECASE)
    for node, amount in resources:
        if node in params["nodes"]:
            params["resources_at"][node.strip()] = int(float(amount.rstrip('.')))
    for node in params["nodes"]:
        if node not in params["resources_at"]:
            params["resources_at"][node] = 0
    if not params["resources_at"]:
        res_input = input("Enter resources at nodes (e.g., a:15,b:0,c:0,d:0): ").strip()
        for item in res_input.split(","):
            match = re.match(r"(\w+):(\d+\.?\d*)", item.strip())
            if match:
                node, amount = match.groups()
                params["resources_at"][node] = int(float(amount.rstrip('.')))

    # resources_required
    req_pattern = r"(?:node\s+)?['\"]?(\w+)['\"]?\s*(?:needs|requiring)\s*(\d+\.?\d*)\s*(?:unit|resource|resources?)"
    req = re.findall(req_pattern, user_input, re.IGNORECASE)
    for node, amount in req:
        params["resources_required"][node.strip()] = int(float(amount.rstrip('.')))
    if not params["resources_required"]:
        req_input = input("Enter required resources (e.g., d:15): ").strip()
        match = re.match(r"(\w+):(\d+\.?\d*)", req_input.strip())
        if match:
            node, amount = match.groups()
            params["resources_required"][node] = int(float(amount.rstrip('.')))

    # drones
    drone_pattern = r"(?:the\s+)?(?:first|second)?\s*drone\s*(?:\(([^)]+)\)|(\w+))?\s*(?:has|with)?\s*(?:resource\s*)?capacity\s*(\d+\.?\d*)\s*(?:resource|resources?|unit)?\s*(?:and)?\s*energy\s*(?:capacity)?\s*(\d+\.?\d*)"
    drones = re.findall(drone_pattern, user_input, re.IGNORECASE)
    for idx, (paren_name, simple_name, capacity, energy) in enumerate(drones, 1):
        name = paren_name or simple_name or f"drone{idx}"
        params["drones"][name.strip()] = {
            "capacity": int(float(capacity.rstrip('.'))),
            "energy": int(float(energy.rstrip('.')))
        }
    if not params["drones"]:
        drones_input = input("Enter drones (e.g., drone1:2:200,drone2:1:300): ").strip()
        drone_pattern = r"(\w+):(\d+\.?\d*)\s*[,;]\s*(\d+\.?\d*)"
        for drone in drones_input.split(","):
            match = re.match(drone_pattern, drone.strip())
            if match:
                name, capacity, energy = match.groups()
                params["drones"][name] = {
                    "capacity": int(float(capacity.rstrip('.'))),
                    "energy": int(float(energy.rstrip('.')))
                }

    # timing
    timing_pattern = r"(takeoff|landing|travel|service|t_max)\s*(?:time)?\s*(?:are\s*(?:both\s*)?|is\s*)[:=]?\s*(\d+\.?\d*)"
    timing = re.findall(timing_pattern, user_input, re.IGNORECASE)
    timing_map = {"flight": "travel"}
    for key, value in timing:
        key = timing_map.get(key.lower(), key.lower())
        params["timing"][key] = int(float(value.rstrip('.')))
    both_match = re.search(r"(?:the\s+)?takeoff\s*(?:time)?\s*(?:and\s*landing\s*(?:time)?)?\s*(?:are\s*both)\s*(\d+\.?\d*)", user_input, re.IGNORECASE)
    if both_match:
        value = int(float(both_match.group(1).rstrip('.')))
        params["timing"]["takeoff"] = value
        params["timing"]["landing"] = value
    if not all(key in params["timing"] for key in {"takeoff", "landing", "travel", "service", "t_max"}):
        timing_input = input("Enter timing (e.g., takeoff:3,landing:5,travel:10,service:0,t_max:1000): ").strip()
        for item in timing_input.split(","):
            match = re.match(r"(\w+):(\d+\.?\d*)", item.strip())
            if match:
                key, value = match.groups()
                params["timing"][key] = int(float(value.rstrip('.')))

    # max_trips
    trips_match = re.search(r"(?:max\s*trips?|trips?)\s*(?:per\s*drone)?\s*(?:is\s*)[:=]?\s*(\d+\.?\d*)", user_input, re.IGNORECASE)
    if trips_match:
        params["max_trips"] = int(float(trips_match.group(1).rstrip('.')))
    if not params["max_trips"]:
        params["max_trips"] = int(input("Enter max trips: ").strip())

    return params

def validate_nodes(nodes):
    if not nodes or not all(isinstance(n, str) and n.strip() for n in nodes):
        raise ValueError("Nodes must be a non-empty list of non-empty strings.")
    if len(nodes) != len(set(nodes)):
        raise ValueError("Duplicate node names are not allowed.")
    if 'a' not in nodes:
        raise ValueError("Nodes must include 'a' (source).")
    return nodes

def validate_edges(edges, nodes):
    edges_dict = {}
    nodes_set = set(nodes)
    for edge in edges:
        if not isinstance(edge, dict) or not all(k in edge for k in ['from', 'to', 'weight']):
            raise ValueError("Each edge must be a dict with 'from', 'to', and 'weight' keys.")
        start, end, weight = edge['from'], edge['to'], edge['weight']
        if start not in nodes_set or end not in nodes_set:
            raise ValueError(f"Edge {start}->{end} references invalid node(s). Nodes: {nodes}")
        try:
            weight = float(weight)
            if weight <= 0:
                raise ValueError(f"Edge weight for {start}->{end} must be positive.")
        except (TypeError, ValueError):
            raise ValueError(f"Edge weight for {start}->{end} must be a positive number.")
        edges_dict[(start, end)] = weight
    return edges_dict

def validate_resources_at(resources, nodes, initial_node='a'):
    if not isinstance(resources, dict):
        raise ValueError("resources_at must be a dictionary.")
    if initial_node not in resources:
        raise ValueError(f"resources_at must include source node '{initial_node}'.")
    resources_dict = {}
    nodes_set = set(nodes)
    for node, amount in resources.items():
        if node not in nodes_set:
            raise ValueError(f"resources_at node {node} is not in nodes: {nodes}")
        try:
            amount = int(float(str(amount).rstrip('.')))
            if amount < 0:
                raise ValueError(f"resources_at amount for {node} must be non-negative.")
        except (TypeError, ValueError):
            raise ValueError(f"resources_at amount for {node} must be a non-negative integer.")
        resources_dict[node] = amount
    return resources_dict

def validate_resources_required(resources, nodes, final_node):
    if not isinstance(resources, dict) or list(resources.keys()) != [final_node]:
        raise ValueError(f"resources_required must be a dictionary with only node '{final_node}'.")
    if final_node not in nodes:
        raise ValueError(f"Destination node '{final_node}' must be in nodes.")
    try:
        amount = int(float(str(resources[final_node]).rstrip('.')))
        if amount < 0:
            raise ValueError(f"resources_required amount for '{final_node}' must be non-negative.")
    except (TypeError, ValueError):
        raise ValueError(f"resources_required amount for '{final_node}' must be a non-negative integer.")
    return {final_node: amount}

def validate_drones(drones, nodes):
    if not drones:
        raise ValueError("At least one drone is required.")
    drones_dict = {}
    for name, props in drones.items():
        if not isinstance(props, dict) or not all(k in props for k in ['capacity', 'energy']):
            raise ValueError(f"Drone {name} must have 'capacity' and 'energy' properties.")
        try:
            capacity = int(float(str(props['capacity']).rstrip('.')))
            energy = int(float(str(props['energy']).rstrip('.')))
            if capacity <= 0 or energy <= 0:
                raise ValueError(f"Drone {name} must have positive capacity and energy.")
        except (TypeError, ValueError):
            raise ValueError(f"Drone {name} capacity and energy must be positive integers.")
        drones_dict[name] = {'capacity': capacity, 'energy': energy}
    return drones_dict

def validate_timing_params(timing):
    expected_keys = {'takeoff', 'landing', 'travel', 'service', 't_max'}
    if not isinstance(timing, dict) or not all(k in timing for k in expected_keys):
        raise ValueError(f"Timing parameters must include: {expected_keys}")
    for key, value in timing.items():
        try:
            value = int(float(str(value).rstrip('.')))
            if value < 0:
                raise ValueError(f"Timing parameter {key} must be non-negative.")
        except (TypeError, ValueError):
            raise ValueError(f"Timing parameter {key} must be a non-negative integer.")
    return timing['takeoff'], timing['landing'], timing['travel'], timing['service'], timing['t_max']

def validate_max_trips(max_trips):
    try:
        max_trips = int(float(str(max_trips).rstrip('.')))
        if max_trips <= 0:
            raise ValueError("max_trips must be positive.")
    except (TypeError, ValueError):
        raise ValueError("max_trips must be a positive integer.")
    return max_trips

def find_path_in_graph(nodes, edges, start, end):
    graph = {node: [] for node in nodes}
    for edge in edges:
        graph[edge['from']].append(edge['to'])
    visited = set()
    queue = deque([start])
    visited.add(start)
    while queue:
        current = queue.popleft()
        if current == end:
            return True
        for neighbor in graph[current]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    return False

def validate_inputs(params, initial_node='a', final_node='d'):
    nodes = validate_nodes(params['nodes'])
    edges = validate_edges(params['edges'], nodes)
    resources_at = validate_resources_at(params['resources_at'], nodes, initial_node)
    resources_required = validate_resources_required(params['resources_required'], nodes, final_node)
    drones = validate_drones(params['drones'], nodes)
    takeoff_time, landing_time, travel_time, service_time, t_max = validate_timing_params(params['timing'])
    max_trips = validate_max_trips(params['max_trips'])

    if not find_path_in_graph(nodes, edges, initial_node, final_node):
        raise ValueError(f"No path from source '{initial_node}' to destination '{final_node}'.")
    if not find_path_in_graph(nodes, edges, final_node, initial_node):
        raise ValueError(f"No path from destination '{final_node}' back to source '{initial_node}'.")

    total_available = resources_at.get(initial_node, 0)
    total_required = resources_required[final_node]
    if total_required > total_available:
        raise ValueError(f"Required resources ({total_required}) exceed available resources at '{initial_node}' ({total_available}).")

    return {
        'nodes': nodes,
        'edges': edges,
        'resources_at': resources_at,
        'resources_required': resources_required,
        'drones': drones,
        'takeoff_time': takeoff_time,
        'landing_time': landing_time,
        'travel_time': travel_time,
        'service_time': service_time,
        't_max': t_max,
        'max_trips': max_trips,
        'initial_node': initial_node,
        'final_node': final_node
    }

def save_json(params):
    with open("generated_inputs.json", "w") as f:
        json.dump(params, f, indent=2)
    print("Generated JSON saved as 'generated_inputs.json'.")

def generate_solver_code_with_llm(params):
    nodes_str = str(params['nodes'])
    edges_str = '{\n' + ',\n'.join(f"    ('{edge['from']}', '{edge['to']}'): {edge['weight']}" for edge in params['edges']) + '\n}'
    resources_at_str = '{\n' + ',\n'.join(f"    '{k}': {v}" for k, v in params['resources_at'].items()) + '\n}'
    resources_required_str = "{" + f"'{params['final_node']}': {params['resources_required'][params['final_node']]}" + '}'
    drones_str = '{\n' + ',\n'.join(f"    '{k}': {{'capacity': {v['capacity']}, 'energy': {v['energy']}}}" 
                                    for k, v in params['drones'].items()) + '\n}'
    initial_node = params['initial_node']
    final_node = params['final_node']
    takeoff_time = params['takeoff_time']
    landing_time = params['landing_time']
    travel_time = params['travel_time']
    service_time = params['service_time']
    t_max = params['t_max']
    max_trips = params['max_trips']

    prompt = f"""
You are an expert Python developer. Below are parameters for a drone delivery optimization problem using the Z3 solver. Generate a Python script that implements a Z3 solver for a drone delivery problem with the following constraints:
1. **Parameters**: Use the exact parameters provided below, with '{initial_node}' as the source and '{final_node}' as the destination.
2. **Constraints**:
   - (2) Non-negative mission start times (s_k >= 0) for each drone k.
   - (3) Non-negative delivery times (y^k_i >= 0) for each node i and drone k.
   - (4) Distinct takeoff times for drones (s_k != s_q for k != q).
   - (5) Each drone starts from '{initial_node}' (sum of outgoing edges from '{initial_node}' = 1).
   - (6) No simultaneous deliveries to '{final_node}' by multiple drones.
   - (7) If drone k travels from '{initial_node}' to node j, delivery time y^k_j = s_k + t_{{1,j}} + landing_time.
   - (8) If drone k travels from node i to j (i,j != '{initial_node}'), y^k_j = y^k_i + t_{{i,j}} + service_time + landing_time.
   - (9) Flow conservation: incoming edges = outgoing edges at each node.
   - (10) Delivery times bounded by t_max.
   - (11) No self-loops (x^k_{{i,i}} = 0).
   - Drones alternate between delivery (to '{final_node}') and pickup (to '{initial_node}') trips.
   - Drones respect capacity and energy constraints.
3. **Objective**: Minimize total energy used.
4. **Output**: A table with columns: Trip, Type, Start Node, End Node, Path, Start, End, EN Used, Current EN, Resources Held, Required Amount.
5. Include a `find_path` function to reconstruct delivery and pickup paths.
6. Ensure drones start at '{initial_node}', deliver to '{final_node}', and return to '{initial_node}' for pickup.

Parameters:
initial_node = '{initial_node}'
final_node = '{final_node}'
nodes = {nodes_str}
edges = {edges_str}
resources_at = {resources_at_str}
resources_required = {resources_required_str}
drones = {drones_str}
max_trips = {max_trips}
t_max = {t_max}
takeoff_time = {takeoff_time}
landing_time = {landing_time}
travel_time = {travel_time}
service_time = {service_time}

Output only the Python script, without explanations or markdown formatting.
"""

    try:
        generated_code = llm(prompt, max_new_tokens=4000, temperature=0.7, top_p=0.9)
        if "def" not in generated_code or "z3" not in generated_code.lower():
            raise ValueError("Invalid code generated by LLM")
    except Exception as e:
        print(f"Error generating code with LLM: {e}")
        print("Using template-based fallback.")
        generated_code = f"""from z3 import *
import time

s = Optimize()

# Parameters
initial_node = '{initial_node}'
final_node = '{final_node}'
nodes = {nodes_str}
edges = {edges_str}
resources_at = {resources_at_str}
resources_required = {resources_required_str}
drones = {drones_str}
max_trips = {max_trips}
t_max = {t_max}
takeoff_time = {takeoff_time}
landing_time = {landing_time}
travel_time = {travel_time}
service_time = {service_time}

# Variables
X_delivery = {{}}
X_pickup = {{}}
load = {{}}
energy_used = {{}}
resources_delivered = {{}}
delivery_time = {{}}
pickup_time = {{}}
start_time = {{}}
arrival_time = {{}}

for k in drones:
    energy_used[k] = Int(f'energy_used_{{k}}')
    s.add(energy_used[k] >= 0)
    for node in nodes:
        load[k, node] = Int(f'load_{{k}}_{{node}}')
        s.add(load[k, node] >= -100, load[k, node] <= 100)
    for t in range(1, max_trips + 1):
        for (i, j) in edges:
            X_delivery[k, t, i, j] = Bool(f'X_delivery_{{k}}_{{t}}_{{i}}_{{j}}')
            X_pickup[k, t, i, j] = Bool(f'X_pickup_{{k}}_{{t}}_{{i}}_{{j}}')
        resources_delivered[k, t] = Int(f'resources_delivered_{{k}}_{{t}}')
        s.add(resources_delivered[k, t] >= 0, resources_delivered[k, t] <= drones[k]['capacity'])
        delivery_time[k, t] = Int(f'delivery_time_{{k}}_{{t}}_{{final_node}}')
        s.add(delivery_time[k, t] >= 0)
        pickup_time[k, t] = Int(f'pickup_time_{{k}}_{{t}}_{{initial_node}}')
        s.add(pickup_time[k, t] >= 0)
        start_time[k, t] = Int(f'start_time_{{k}}_{{t}}')
        s.add(start_time[k, t] >= 0)
        for node in nodes:
            if node not in [initial_node, final_node]:
                arrival_time[k, t, node] = Int(f'arrival_time_{{k}}_{{t}}_{{node}}')
                s.add(arrival_time[k, t, node] >= 0)

# Constraint (2, 3): Non-negative start and delivery times
for k in drones:
    for t in range(1, max_trips + 1):
        s.add(start_time[k, t] >= 0)
        s.add(delivery_time[k, t] >= 0)
        s.add(pickup_time[k, t] >= 0)
        for node in nodes:
            if node not in [initial_node, final_node]:
                s.add(arrival_time[k, t, node] >= 0)

# Constraint (4): Distinct takeoff times
for k1 in drones:
    for k2 in drones:
        if k1 < k2:
            s.add(start_time[k1, 1] != start_time[k2, 1])

# Constraint (5): Start from initial_node
for k in drones:
    for t in range(1, max_trips + 1):
        is_delivery = Or([X_delivery[k, t, i, final_node] for i in nodes if (i, final_node) in edges])
        s.add(Implies(is_delivery, Sum([If(X_delivery[k, t, initial_node, j], 1, 0) for j in nodes if (initial_node, j) in edges]) == 1))

# Constraint (6): No simultaneous deliveries
for k1 in drones:
    for k2 in drones:
        if k1 < k2:
            for t1 in range(1, max_trips + 1):
                for t2 in range(1, max_trips + 1):
                    s.add(Implies(
                        And(
                            Or([X_delivery[k1, t1, i, final_node] for i in nodes if (i, final_node) in edges]),
                            Or([X_delivery[k2, t2, i, final_node] for i in nodes if (i, final_node) in edges])
                        ),
                        delivery_time[k1, t1] != delivery_time[k2, t2]
                    ))

# Constraints (7, 8): Timing constraints
for k in drones:
    for t in range(1, max_trips + 1):
        is_delivery = Or([X_delivery[k, t, i, final_node] for i in nodes if (i, final_node) in edges])
        is_pickup = Or([X_pickup[k, t, i, initial_node] for i in nodes if (i, initial_node) in edges])
        for i in nodes:
            for j in nodes:
                if (i, j) in edges:
                    if i == initial_node and j != initial_node:
                        if j == final_node:
                            s.add(Implies(
                                X_delivery[k, t, i, j],
                                delivery_time[k, t] == start_time[k, t] + takeoff_time + edges[(i, j)] + landing_time
                            ))
                        else:
                            s.add(Implies(
                                X_delivery[k, t, i, j],
                                arrival_time[k, t, j] == start_time[k, t] + takeoff_time + edges[(i, j)] + landing_time
                            ))
                    elif i != initial_node and j != initial_node and i != final_node and j != final_node:
                        s.add(Implies(
                            And(X_delivery[k, t, i, j], Or([X_delivery[k, t, n, i] for n in nodes if (n, i) in edges])),
                            arrival_time[k, t, j] == arrival_time[k, t, i] + travel_time + edges[(i, j)] + service_time + landing_time
                        ))
                    elif i != initial_node and j == final_node:
                        s.add(Implies(
                            X_delivery[k, t, i, j],
                            delivery_time[k, t] == arrival_time[k, t, i] + travel_time + edges[(i, j)] + service_time + landing_time
                        ))
                    if i == final_node and j != final_node:
                        if j == initial_node:
                            s.add(Implies(
                                X_pickup[k, t, i, j],
                                pickup_time[k, t] == start_time[k, t] + takeoff_time + edges[(i, j)] + landing_time
                            ))
                        else:
                            s.add(Implies(
                                X_pickup[k, t, i, j],
                                arrival_time[k, t, j] == start_time[k, t] + takeoff_time + edges[(i, j)] + landing_time
                            ))
                    elif i != final_node and j != final_node and i != initial_node and j != initial_node:
                        s.add(Implies(
                            And(X_pickup[k, t, i, j], Or([X_pickup[k, t, n, i] for n in nodes if (n, i) in edges])),
                            arrival_time[k, t, j] == arrival_time[k, t, i] + travel_time + edges[(i, j)] + service_time + landing_time
                        ))
                    elif i != final_node and j == initial_node:
                        s.add(Implies(
                            X_pickup[k, t, i, j],
                            pickup_time[k, t] == arrival_time[k, t, i] + travel_time + edges[(i, j)] + service_time + landing_time
                        ))

# Constraint (9): Flow conservation
for k in drones:
    for t in range(1, max_trips + 1):
        is_delivery = Or([X_delivery[k, t, i, final_node] for i in nodes if (i, final_node) in edges])
        is_pickup = Or([X_pickup[k, t, i, initial_node] for i in nodes if (i, initial_node) in edges])
        for node in nodes:
            inbound_delivery = Sum([If(X_delivery[k, t, i, node], 1, 0) for (i, j) in edges if j == node])
            outbound_delivery = Sum([If(X_delivery[k, t, node, j], 1, 0) for (i, j) in edges if i == node])
            inbound_pickup = Sum([If(X_pickup[k, t, i, node], 1, 0) for (i, j) in edges if j == node])
            outbound_pickup = Sum([If(X_pickup[k, t, node, j], 1, 0) for (i, j) in edges if i == node])
            if node == initial_node:
                s.add(Implies(is_delivery, And(outbound_delivery == 1, inbound_delivery == 0)))
                s.add(Implies(is_pickup, And(inbound_pickup == 1, outbound_pickup == 0)))
            elif node == final_node:
                s.add(Implies(is_delivery, And(inbound_delivery == 1, outbound_delivery == 0)))
                s.add(Implies(is_pickup, And(outbound_pickup == 1, inbound_pickup == 0)))
            else:
                s.add(Implies(is_delivery, inbound_delivery == outbound_delivery))
                s.add(Implies(is_pickup, inbound_pickup == outbound_pickup))
            s.add(Implies(Not(is_delivery), inbound_delivery == 0))
            s.add(Implies(Not(is_pickup), inbound_pickup == 0))

# Constraint (10): Delivery and pickup times bounded by t_max
for k in drones:
    for t in range(1, max_trips + 1):
        s.add(Implies(Or([X_delivery[k, t, i, final_node] for i in nodes if (i, final_node) in edges]), delivery_time[k, t] <= t_max))
        s.add(Implies(Or([X_pickup[k, t, i, initial_node] for i in nodes if (i, initial_node) in edges]), pickup_time[k, t] <= t_max))

# Constraint (11): No self-loops
for k in drones:
    for t in range(1, max_trips + 1):
        for i in nodes:
            if (i, i) in edges:
                s.add(X_delivery[k, t, i, i] == False)
                s.add(X_pickup[k, t, i, i] == False)

# Energy and resource constraints
for k in drones:
    s.add(energy_used[k] == Sum([If(X_delivery[k, t, i, j], edges[(i, j)], 0) + If(X_pickup[k, t, i, j], edges[(i, j)], 0)
                                 for (i, j) in edges for t in range(1, max_trips + 1)]))
    s.add(energy_used[k] <= drones[k]['energy'])
    s.add(-load[k, final_node] == Sum([resources_delivered[k, t] for t in range(1, max_trips + 1)]))
    s.add(load[k, initial_node] == Sum([resources_delivered[k, t] for t in range(1, max_trips + 1)]))

# Resource constraints
total_picked = Sum([load[k, initial_node] for k in drones])
total_delivered = Sum([resources_delivered[k, t] for k in drones for t in range(1, max_trips + 1)])
s.add(total_picked <= resources_at[initial_node])
s.add(total_delivered == resources_required[final_node])

# Alternation and position continuity
for k in drones:
    s.add(Or([X_delivery[k, 1, initial_node, j] for j in nodes if (initial_node, j) in edges]))
    for t in range(1, max_trips):
        prev_is_delivery = Or([X_delivery[k, t, i, final_node] for i in nodes if (i, final_node) in edges])
        prev_is_pickup = Or([X_pickup[k, t, i, initial_node] for i in nodes if (i, initial_node) in edges])
        curr_is_delivery = Or([X_delivery[k, t+1, i, final_node] for i in nodes if (i, final_node) in edges])
        curr_is_pickup = Or([X_pickup[k, t+1, i, initial_node] for i in nodes if (i, initial_node) in edges])
        future_resources = Sum([resources_delivered[k, t2] for t2 in range(t + 1, max_trips + 1)])
        s.add(Implies(And(prev_is_delivery, future_resources > 0), curr_is_pickup))
        s.add(Implies(And(prev_is_pickup, future_resources > 0), curr_is_delivery))
        s.add(Implies(future_resources == 0, And(Not(curr_is_delivery), Not(curr_is_pickup))))
        s.add(start_time[k, t + 1] >= start_time[k, t] + 1)
        s.add(Implies(prev_is_delivery, start_time[k, t + 1] >= delivery_time[k, t]))
        s.add(Implies(prev_is_pickup, start_time[k, t + 1] >= pickup_time[k, t]))

# Objective: Minimize total energy
s.minimize(Sum([energy_used[k] for k in drones]))

# Solve and display results
start = time.time()
if s.check() == sat:
    m = s.model()
    print(f"Solve time: {{time.time() - start:.2f}} seconds\\n")
    print("Solution:\\n")
    total_resources_delivered = 0
    initial_resources_required = resources_required[final_node]

    def find_path(start, end, trip_type, k, t):
        path = []
        energy = 0
        current = start
        visited = set([current])
        max_steps = len(nodes)
        steps = 0
        while current != end and steps < max_steps:
            found_next = False
            for next_node in nodes:
                if (current, next_node) in edges and next_node not in visited:
                    edge_val = m.evaluate(X_delivery[k, t, current, next_node]) if trip_type == 'delivery' else \
                               m.evaluate(X_pickup[k, t, current, next_node])
                    if edge_val:
                        path.append((current, next_node))
                        energy += edges[(current, next_node)]
                        current = next_node
                        visited.add(current)
                        found_next = True
                        break
            if not found_next:
                return [], 0
            steps += 1
        if current != end:
            return [], 0
        return path, energy

    for k in drones:
        print(f"Drone {{k}}:")
        table_data = []
        trip_counter = 1
        cumulative_energy = 0
        for t in range(1, max_trips + 1):
            is_delivery = m.evaluate(Or([X_delivery[k, t, i, final_node] for i in nodes if (i, final_node) in edges]))
            is_pickup = m.evaluate(Or([X_pickup[k, t, i, initial_node] for i in nodes if (i, initial_node) in edges]))
            start_node = initial_node if t == 1 or (t > 1 and m.evaluate(Or([X_pickup[k, t-1, i, initial_node] for i in nodes if (i, initial_node) in edges]))) else final_node
            if is_delivery:
                path, energy = find_path(initial_node, final_node, 'delivery', k, t)
                if path:
                    resources_held = drones[k]['capacity']
                    delivered_amount = m.evaluate(resources_delivered[k, t]).as_long()
                    total_resources_delivered += delivered_amount
                    remaining_resources = initial_resources_required - total_resources_delivered
                    cumulative_energy += energy
                    current_energy_capacity = drones[k]['energy'] - cumulative_energy
                    table_data.append({{
                        'Trip': trip_counter,
                        'Type': 'Delivery',
                        'Start Node': start_node,
                        'End Node': final_node,
                        'Path': ', '.join(f'{{i}}->{{j}}' for i, j in path),
                        'Start': m.evaluate(start_time[k, t]).as_long(),
                        'End': f"{{m.evaluate(delivery_time[k, t]).as_long()}} (delivery)",
                        'EN Used': energy,
                        'Current EN': current_energy_capacity,
                        'Resources Held': resources_held,
                        'Required Amount': f"{{final_node}}({{remaining_resources}})"
                    }})
                    trip_counter += 1
            if is_pickup:
                path, energy = find_path(final_node, initial_node, 'pickup', k, t)
                if path:
                    resources_held = 0
                    cumulative_energy += energy
                    current_energy_capacity = drones[k]['energy'] - cumulative_energy
                    remaining_resources = initial_resources_required - total_resources_delivered
                    table_data.append({{
                        'Trip': trip_counter,
                        'Type': 'Pickup',
                        'Start Node': start_node,
                        'End Node': initial_node,
                        'Path': ', '.join(f'{{i}}->{{j}}' for i, j in path),
                        'Start': m.evaluate(start_time[k, t]).as_long(),
                        'End': f"{{m.evaluate(pickup_time[k, t]).as_long()}} (pickup)",
                        'EN Used': energy,
                        'Current EN': current_energy_capacity,
                        'Resources Held': resources_held,
                        'Required Amount': f"{{final_node}}({{remaining_resources}})"
                    }})
                    trip_counter += 1
        print(f"  Picks up {{m.evaluate(load[k, initial_node]).as_long()}} resources at {{initial_node}}")
        print(f"  Delivers {{-m.evaluate(load[k, final_node]).as_long()}} resources at {{final_node}}")
        print(f"  Energy used: {{m.evaluate(energy_used[k]).as_long()}}")
        print(f"  Number of delivery trips: {{m.evaluate(Sum([If(Or([X_delivery[k, t, i, final_node] for i in nodes if (i, final_node) in edges]), 1, 0) for t in range(1, max_trips + 1)])).as_long()}}")
        print("\\n  Trip Summary:")
        print(f"  {{'Trip':<5}} {{'Type':<10}} {{'Start Node':<10}} {{'End Node':<9}} {{'Path':<22}} {{'Start':<6}} {{'End':<16}} {{'EN Used':<7}} {{'Current EN':<11}} {{'Resources Held':<14}} {{'Required Amount':<15}}")
        print("  " + "-" * 120)
        for row in table_data:
            print(f"  {{row['Trip']:<5}} {{row['Type']:<10}} {{row['Start Node']:<10}} {{row['End Node']:<9}} {{row['Path']:<22}} {{row['Start']:<6}} {{row['End']:<16}} {{row['EN Used']:<7}} {{row['Current EN']:<11}} {{row['Resources Held']:<14}} {{row['Required Amount']:<15}}")
        print()
    print(f"Total delivered: {{m.evaluate(total_delivered).as_long()}}")
    print(f"Total energy used: {{sum(m.evaluate(energy_used[k]).as_long() for k in drones)}}")
else:
    print("No solution found.")
"""
    with open("custom_drone_solver.py", "w") as f:
        f.write(generated_code)
    print("Generated solver code saved as 'custom_drone_solver.py'.")
    return generated_code

def main():
    try:
        params = parse_conversational_input_with_llm()
        final_node = list(params['resources_required'].keys())[0] if params['resources_required'] else 'd'
        validated_params = validate_inputs(params, initial_node='a', final_node=final_node)
        print("\nParsed Parameters:")
        print(json.dumps(validated_params, indent=2))
        confirm = input("\nAre these parameters correct? (yes/no): ").strip().lower()
        if confirm != 'yes':
            print("Please rerun the script and provide corrected input.")
            return
        save_json(validated_params)
        solver_code = generate_solver_code_with_llm(validated_params)
        if not solver_code:
            print("Failed to generate solver code.")
            return
        output_file = "custom_drone_solver.py"
        with open(output_file, 'w') as f:
            f.write(solver_code)
        print(f"\nCustom solver code generated at '{output_file}'.")
        print("Run it with: python custom_drone_solver.py")
    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()