import os
import re
import ast
import textwrap
import time
from datetime import datetime
import uuid
from openai import OpenAI
from z3 import z3
import sys
from io import StringIO
import builtins
from itertools import combinations
import json
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional
import token as tk
from io import BytesIO
import tokenize

api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI()
MAX_RETRIES = 5
GENERATED_CODE_FILE = "generated_solver.py"
JSON_ARTIFACT_DIR = "artifacts"  

safe_builtins = {
    "Exception": builtins.Exception,
    "int": builtins.int,
    "str": builtins.str,
    "float": builtins.float,
    "bool": builtins.bool,
    "list": builtins.list,
    "dict": builtins.dict,
    "set": builtins.set,
    "tuple": builtins.tuple,
    "range": builtins.range,
    "len": builtins.len,
    "min": builtins.min,
    "max": builtins.max,
    "sum": builtins.sum,
    "abs": builtins.abs,
    "isinstance": builtins.isinstance,
    "enumerate": builtins.enumerate,
    "print": builtins.print,
    "__import__": builtins.__import__,
    "sorted": builtins.sorted,
    "type": builtins.type,
    "hasattr": builtins.hasattr,
    "getattr": builtins.getattr,
    "dict": builtins.dict,
    "list": builtins.list,
    "object": builtins.object,
    "property": builtins.property,
    "True": True,
    "False": False,
    "None": None,
}

class Edge(BaseModel):
    from_node: str = Field(alias="from")
    to_node: str = Field(alias="to")
    distance: int
    time: int

class Drone(BaseModel):
    name: str
    capacity: int
    energy: int

class OptionalParams(BaseModel):
    max_trips: int = 6
    t_max: int = 500
    w: int = 0
    t_s: int = 5

class ProblemParams(BaseModel):
    nodes: List[str]
    source_node: str
    resources_available: Dict[str, int]
    resources_required: Dict[str, int]
    edges: List[Edge]
    drones: List[Drone]
    optional: OptionalParams
    missing_parameters: Optional[List[str]] = None
    constraints: List[str]

    @validator('nodes')
    def nodes_not_empty(cls, v):
        if not v:
            raise ValueError('Nodes cannot be empty')
        return v
    
    @validator('drones')
    def drones_not_empty(cls, v):
        if not v:
            raise ValueError('Drones list cannot be empty')
        return v

    @validator('edges')
    def edges_not_empty(cls, v):
        if not v:
            raise ValueError('Edges list cannot be empty')
        return v

    @validator('source_node')
    def source_in_nodes(cls, v, values):
        if 'nodes' in values and v not in values['nodes']:
            raise ValueError('Source node must be in nodes list')
        return v

    @validator('resources_required')
    def req_in_nodes(cls, v, values):
        nodes = values.get('nodes', [])
        for node in v.keys():
            if node not in nodes:
                raise ValueError(f'Resource required for unknown node {node}')
        return v
        
INPUT_PROMPT = """
Your role is to assist a disaster manager who must distribute lifesaving resources (food and water) within a disaster-affected region using a fleet of UAVs (which we also call drones). 
Drones have capacities, energy limits, and must follow routes (edges exp: a->b:2 from node 'a' to 'b') with distances/times, while respecting constraints like no zero-delivery trips, energy consumption (equals edge distance), and returning to the depot after each trip.

You will assist by requesting the following information in 1, 2, 3  and 4 below from the disater manager, and generate a JSON formatted file as an artefect when all information is confirmed.

1. a. Information about the region. You may assume the region is represented by a weighted, bi-directional GRAPH G, (weighted by physical distance in meters).
b. Ask followup details from the disaster manager using basic graph terminology such as nodes, edges, weights.
2. a. The number of drones (k) currently available,
   b. Their source location (source must be the same for all drones)
   c. Each drone's battery power, and capacity for carrying resources
3. Amounts and locations of food and water resource demands throughout the region. All resources must be at the source node, and cannot reside on edges.
4. Additional information such as 
    a. Maximum time limit t_max,
    b. Maximum Number of trips a drone can make max_trips. One trip is considered when the drone (k) from the source to other locations back to the source location,
    c. Landing, takeoff, or service time required at delivery nodes (w).
    d. Minimum time gap between drone start times for the same trip (t_s)

Ensure to extract the below information from the user: 
- Nodes: List of location names (exp: ["a", "b", "c"]).
- Source Node: Node with resources (exp: "c").
- Resources Available: Dict of resources at source (exp: {"c": 15}).
- Resources Required: Dict of resources needed at nodes (exp: {"b": 4, "a": 5}).
- Edges: Routes with distances and times (exp: [{"from": "a", "to": "b", "distance": 3, "time": 3} or {"a"->"c":5" means a to c takes 5 distance and 5 time} ]). Assume times=distances. Include reverse edges for bidirectional routes with same distance/time unless stated otherwise. Autocorrect minor typos in edge notation (example: 'a-c>:7' to 'a->c:7')
- Drones: List of dicts with name, capacity, energy (exp: [{"name": "drone1", "capacity": 1, "energy": 100}]). if no drone names are mentioned, use "drone1", "drone2", ...
- Optional Parameters: Dict with max_trips (per drone), t_max, w (landing/takeoff/service time), t_s (start time gap).
If parameters are missing, use defaults (max_trips=6 per drone, t_max=500, w=0, t_s=5) and list ambiguities in "missing_parameters" only if critical (exp: nodes, source, edges, drones).

Decision Variables:
- X_delivery[k,t,i,j]: Bool, if drone k uses edge (node i, node j) in trip t.
- start_time[k,t]: Int (>=0), start time of trip t for drone k. In constraints, denoted as s^k or start_time[k,t].
- arrival_time[k,t,i]: Int (>=0), arrival time at node i for drone k in trip t. In constraints, denoted as y_i^k or arrival_time[k,t,i].
- delivery_time[k,t,i]: Int (>=0), delivery time at node i for drone k in trip t (delivery nodes only). In constraints, denoted as y_i^k when referring to delivery times.
- resources_delivered[k,t,i]: Int (>=0, <= capacity), resources delivered to node i by drone k in trip t (delivery nodes only).
- load[k,i]: Int, resources carried by drone k at node i (positive at source, negative at delivery nodes, zero elsewhere).
- energy_used[k]: Int, total energy used by drone k.
- trip_active[k,t]: Bool, whether trip t for drone k is active.
- visited_nodes[k,t,i]: Bool, whether node i is visited by drone k in trip t.
- successor[k,t,i]: Int (-1 to len(nodes)-1), index of next node after i for drone k in trip t (-1 if not visited).
- drone_used[k]: Bool, whether drone k performs at least one trip.

The AI agent must be preprogrammed with the following base constraints. Make sure to encode these constraints in the output JSON under the "constraints" key using the suitable varaibles mentioned above.
__CONSTRAINTS_BLOCK__

For each constraint above, translate it into a Z3-readable logical expression string using decision variables and Z3 operators (exp: ForAll, Implies, And, Or, Sum, If). 
Examples:
- 'Each drone trip must start at or after time zero.' to 'ForAll([k,t], start_time[k,t] >= 0)'
- 'The starting times of two drones on the same trip must differ with a start time gap t_s.' to 'ForAll([k1,k2,t], Implies(k1 != k2, Abs(start_time[k1,t] - start_time[k2,t]) >= {t_s}))'
- 'Each trip must begin with exactly one outgoing route from the source location.' to 'ForAll([k,t], Sum([If(X_delivery[k,t,source,j], 1, 0) for j in nodes]) == 1)'
-  Do not change the values for w or t_s or t_max in the constraints. Keep them as variables.
Use 'source' for source_node, 'nodes' for all nodes, 'delivery_nodes' for nodes with resources_required > 0.

Output a JSON object matching this schema:
{
  "nodes": [str],
  "source_node": str,
  "resources_available": {str: int},
  "resources_required": {str: int},
  "edges": [{"from": str, "to": str, "distance": int, "time": int}],
  "drones": [{"name": str, "capacity": int, "energy": int}],
  "optional": {"max_trips": int, "t_max": int, "w": int, "t_s": int},
  "missing_parameters": [str] or null,
  "constraints": [str]
}
"""

# --- Constraint-prompt variants -------------------------------------------
# The constraints section of INPUT_PROMPT comes in two forms, used to compare
# how reliably the LLM translates them into Z3:
#   "pseudo" -> prompts/constraints_pseudocode.txt        (math / logic notation)
#   "nl"     -> prompts/constraints_natural_language.txt  (plain English)
CONSTRAINT_FILES = {
    "pseudo": "constraints_pseudocode.txt",
    "nl": "constraints_natural_language.txt",
}
PROMPTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts")

def build_input_prompt(mode: str) -> str:
    """Return INPUT_PROMPT with the chosen constraints block filled in."""
    if mode not in CONSTRAINT_FILES:
        raise ValueError(f"Unknown constraints mode {mode!r}; choose from {list(CONSTRAINT_FILES)}")
    path = os.path.join(PROMPTS_DIR, CONSTRAINT_FILES[mode])
    with open(path, encoding="utf-8") as fh:
        block = fh.read().rstrip("\n")
    return INPUT_PROMPT.replace("__CONSTRAINTS_BLOCK__", block)

CODE_GEN_PROMPT = """
You are an expert Python developer and Z3 modeler. Generate a Python function "def run_z3_solver(params):" to solve a drone routing and delivery optimization problem using Z3. 
The function models the UAV routing problem by directly encoding all constraints in Z3, without parsing a constraints list. It returns a string with the solution or unsatisfiability reason.

Problem Description:
- Drones deliver resources from a source node to delivery nodes in a weighted, bidirectional graph.
- Each drone has capacity and energy limits, follows edges with distance/time costs, and must return to the source after each trip.
- Constraints ensure valid paths, resource delivery, energy limits, and timing (exp, service time 'w', start time gap 't_s').

Inputs (params dict):
- 'nodes': list[str] (exp, ["a", "b", "c"])
- 'initial_node': str (exp, "c")
- 'resources_at': dict[str,int] (exp, {"c": 15})
- 'resources_required': dict[str,int] (exp, {"a": 3, "b": 2})
- 'edges': dict[(str,str)]->dict (exp, {("a","b"): {"distance": 3, "time": 3}})
- 'drones': dict[str]->dict (exp, {"drone1": {"capacity": 1, "energy": 100}})
- 'optional': dict with 'max_trips' (default 6), 't_max' (default 500), 'w' (default 4), 't_s' (default 5)
- 'delivery_nodes': list[str] (nodes with resources_required > 0)
(Use default values if optional params are missing from JSON file.)
Decision Variables:
- X_delivery[k,t,i,j]: Bool, drone k uses edge (i,j) in trip t.
- start_time[k,t]: Int (>=0), start time of trip t for drone k.
- arrival_time[k,t,i]: Int (>=0), arrival time at node i for drone k in trip t.
- delivery_time[k,t,i]: Int (>=0), delivery time at node i for drone k in trip t (delivery nodes only).
- resources_delivered[k,t,i]: Int (>=0, <= capacity), resources delivered to node i by drone k in trip t.
- load[k,i]: Int, resources carried by drone k at node i.
- energy_used[k]: Int, total energy used by drone k.
- trip_active[k,t]: Bool, whether trip t for drone k is active.
- visited_nodes[k,t,i]: Bool, whether node i is visited by drone k in trip t.
- successor[k,t,i]: Int (-1 to len(nodes)-1), index of next node after i for drone k in trip t.
- drone_used[k]: Bool, whether drone k performs any trips.

Constraints: 
You must formally model the Constraint Satisfaction Problem by encoding the constraints from params['constraints']. For each constraint string:
1. - Identify iteration variables and domains:
     - 'k': 'drones.keys()'
     - 't': 'range(max_trips)'
     - 'i', 'j': 'nodes' or 'delivery_nodes' (if "i in delivery_nodes")
   - Substitute placeholders:
     - 'source' -> 'params['initial_node']'
     - 'nodes' -> 'params['nodes']'
     - 'delivery_nodes' -> 'params['delivery_nodes']'
     - 'drones' -> 'drones.keys()'
     - 'max_trips', 'w', 't_s', 't_max' -> 'params['optional']' values
     - 'edges[(i,j)]['time']' -> 'edges[(i,j)]['time']'
     - 'node_indices[j]' -> 'node_indices[j]' (precompute 'node_indices = {n: idx for idx, n in enumerate(nodes)}')
2. - Implement into Z3:
   - Use evaluate():
   - Translate each constraint into a Z3 expression using 'ForAll', 'Implies', 'And', 'Or', 'Sum', 'If', etc.
   - Handle complex constraints (exp, 'ForAll([k,t,i], Implies(And(i != source, visited_nodes[k,t,i]), Or([And(X_delivery[k,t,i,j], successor[k,t,i] == node_indices[j]) for j in nodes if (i,j) in edges])))') by iterating over appropriate domains and constructing 'Or' conditions.
3. - Ensure Completeness:
   - Process each constraint in params['constraints']. If a pattern from the constraint is unrecognized, log a warning and continue with the rest. Ensure core constraints are not skipped.
   - Ensure decision variables are logically linked through the constraints to reflect route consistency (exp, 'visited_nodes[k,t,i]' is true if any edge '(i,j)' or '(j,i)' is used).

Path Finding
- Implement 'find_path(k,t)' to construct a path starting at 'initial_node', following 'X_delivery[k,t,i,j]', and returning to 'initial_node' if 'trip_active[k,t]'. Limit to 'len(nodes)*2' steps to prevent infinite loops.
- Validate that active trips have non-empty paths and visit all nodes where 'resources_delivered[k,t,i] > 0'.
- Return list of edges '[(i,j), ...]' or '[]' if no valid path or trip inactive.

Output Format:
If satisfiable:
- "Satisfiable\nSolve time: {time:.2f} seconds\nBest Solution:"
- For each drone k (sorted):
  - If no trips: "Drone {k}:\n  No delivery trips"
  - Else:
   Format the solver output exactly like this example:
    Drone drone3:
    Energy used: 38
    Number of delivery trips: 3

    Trip Summary:
    Trip  Path                           Start  End    Delivery Times       EN Used Current EN  Resources Held Required Amount
    ---------------------------------------------------------------------------------------------------------------------------------------
    1     c->b, b->c                     0      13     b@4                  8       42          2              b(0), a(3), d(2), e(1)
    2     c->b, b->a, a->c               25     49     a@37                 14      28          3              b(0), a(0), d(2), e(1)
    3     c->e, e->d, d->c               53     79     d@69, e@61           16      12          3              b(0), a(0), d(0), e(0)

    Then print:
    Overall summary:
  Total resources delivered: {'b': 2, 'a': 3, 'd': 2, 'e': 1}
  Total energy used: 38
  Total trips: 3
  Resources required: {'b': 2, 'a': 3, 'd': 2, 'e': 1}
If unsatisfiable:
- "Unsatisfiable\nReason: {core or error}\nSolve time: {time:.2f} seconds"

Implementation
- Treat the problem as a CSP where all constraints in 'params['constraints']' must be satisfied. Translate each constraint into Z3 expressions accurately, preserving their logical structure.
- Use string parsing (exp, regex or tokenization) to interpret constraint strings, identifying variables, operators ('ForAll', 'Implies', etc.), and domains. Map these to Z3 constructs dynamically.
- Use int values, bound loads (-50..50), successor (-1..len(nodes)-1).
- Catch exceptions: Return "Unsatisfiable\nReason: Solver crashed: {e}\nSolve time: ...".
- No external services; use z3, standard library.
- Test with: 5 nodes (a,b,c,d,e), depot c, bidirectional edges, drones cap1-3/energy50-100, req a3/b2/d2/e1, max_trips=7/t_max=5000/w=5/t_s=4.
Return only the Python function def run_z3_solver(params): ...
"""


def get_gpt_response(system_prompt: str, user_message: str, model: str = "gpt-4o", json_mode: bool = False):
    kwargs = {}
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ],
        **kwargs
    )
    return resp.choices[0].message.content

def generate_with_gpt5(prompt_text: str, model: str = "gpt-5") -> str:
    return get_gpt_response(
        "You are an expert Python/Z3 developer. Output exactly one function def run_z3_solver(params): ...",
        prompt_text,
        model=model
    )

def clean_code(code: str) -> str:
    if code is None:
        return ""
    code = str(code)
    m = re.search(r"```(?:python)?\s*\n(.*?)\n```$", code, re.DOTALL | re.IGNORECASE)
    if not m:
        m = re.search(r"```(?:python)?\s*\n(.*)", code, re.DOTALL | re.IGNORECASE)
    if m:
        return textwrap.dedent(m.group(1)).rstrip() + "\n"
    m = re.search(r"'''(.*?)'''", code, re.DOTALL)
    if m:
        return textwrap.dedent(m.group(1)).rstrip() + "\n"
    m = re.search(r'"""(.*?)"""', code, re.DOTALL)
    if m:
        return textwrap.dedent(m.group(1)).rstrip() + "\n"
    lines = code.splitlines()
    if len(lines) > 0 and lines[0].strip().lower().startswith("python"):
        return textwrap.dedent("\n".join(lines[1:])).rstrip() + "\n"
    return textwrap.dedent(code).strip("\n") + "\n"

def check_brackets(code_str):

    stack = []
    brackets = {')': '(', ']': '[', '}': '{'}
    errors = []
    
    tokens = tokenize.tokenize(BytesIO(code_str.encode('utf-8')).readline)
    for token in tokens:
        t_type, t_string, t_start, t_end, t_line = token
        if t_type in (tk.STRING, tk.COMMENT):
            continue
        for char in t_string:
            if char in '([{':
                stack.append((char, t_start[0], t_start[1]))
            elif char in ')]}':
                if not stack:
                    errors.append({'type': char, 'line': t_start[0], 'col': t_start[1], 'msg': 'Unmatched closing'})
                    continue
                open_bracket, oline, ocol = stack.pop()
                if brackets[char] != open_bracket:
                    errors.append({'type': char, 'line': t_start[0], 'col': t_start[1], 'msg': f'Mismatch: expected {brackets[char]} got {char}'})
    
    for open_bracket, oline, ocol in stack[::-1]:
        errors.append({'type': open_bracket, 'line': oline, 'col': ocol, 'msg': 'Unmatched opening'})
    
    return errors

def parse_and_convert(json_str: str) -> dict:
    try:
        parsed_json = json.loads(json_str)
        params_obj = ProblemParams.model_validate(parsed_json)
        params = params_obj.model_dump()
        
        edges_dict = {(e['from_node'], e['to_node']): {'distance': e['distance'], 'time': e['time']} for e in params['edges']}
        params['edges'] = edges_dict          
        params['drones'] = {d['name']: {'capacity': d['capacity'], 'energy': d['energy']} for d in params['drones']}
        params['initial_node'] = params.pop('source_node')
        params['resources_at'] = params.pop('resources_available')
        delivery_nodes = [n for n in params['nodes'] if params['resources_required'].get(n, 0) > 0]
        params['delivery_nodes'] = delivery_nodes
        
        visited = set()
        stack = [params['initial_node']]
        while stack:
            node = stack.pop()
            if node not in visited:
                visited.add(node)
                stack.extend([to for (fr, to) in params['edges'] if fr == node])
        
        unreachable_delivery = [n for n in delivery_nodes if n not in visited]
        if unreachable_delivery:
            raise ValueError(f"Unreachable delivery nodes from source: {unreachable_delivery} (add connecting edges)")
        
        outgoing = [to for (fr, to) in params['edges'] if fr == params['initial_node']]
        if not outgoing:
            raise ValueError("Source node has no outgoing edges—violates start route constraint")
        
        if params.get('validation_issues'):
            raise ValueError(f"LLM-detected issues: {params['validation_issues']}")
        
        return params
    except Exception as e:
        raise ValueError(f"Validation failed: {e}")

def serialize_params(params: dict) -> str:
    serializable = params.copy()
    serializable['edges'] = {f"{i}->{j}": v for (i, j), v in params['edges'].items()}
    return json.dumps(serializable, indent=2)

def main(constraints_mode: str = "pseudo"):
    print("Welcome to the hybrid parser+code-generator (GPT-5 for everything) Drone Planner.")
    os.makedirs(JSON_ARTIFACT_DIR, exist_ok=True)
    input_prompt = build_input_prompt(constraints_mode)
    print(f"Constraint prompt variant: {constraints_mode}")
    
    while True:        
        session_id = str(uuid.uuid4())[:8]  
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        print("\nEnter the details of the disaster scenario (or 'exit' to quit):")
        user_input = input().strip()
        if user_input.lower() == 'exit':
            break

        # Save user interaction artifact
        user_input_filename = os.path.join(JSON_ARTIFACT_DIR, f"user_input_{session_id}_{timestamp}.txt")
        try:
            with open(user_input_filename, "w", encoding="utf-8") as fh:
                fh.write("=== User Input ===\n")
                fh.write(user_input)
                fh.write("\n\n=== System Prompt ===\n")
                fh.write(f"(constraints variant: {constraints_mode})\n")
                fh.write(input_prompt)
            print(f"User interaction artifact (with prompt) saved to {user_input_filename}")
        except Exception as e:
            print(f"Failed to save user interaction artifact: {e}")

        try:
            parsed_json_str = get_gpt_response(input_prompt, user_input, model="gpt-5", json_mode=True)
            print("\nParsed Parameters (JSON):")
            print(parsed_json_str)
            
            # Save JSON artifact
            json_filename = os.path.join(JSON_ARTIFACT_DIR, f"problem_{session_id}_{timestamp}.json")
            try:
                with open(json_filename, "w", encoding="utf-8") as fh:
                    fh.write(parsed_json_str)
                print(f"JSON artifact saved to {json_filename}")
            except Exception as e:
                print(f"Failed to save JSON artifact: {e}")
            
            params = parse_and_convert(parsed_json_str)  
            missing = params.get('missing_parameters')
            questions = params.get('questions_for_user')
            if missing or questions:
                print("\nIncomplete scenario detected:")
                if missing:
                    print(f"Missing: {', '.join(missing)}")
                if questions:
                    print(f"Suggested questions: {' '.join(questions)}")
                print("Please provide more details in your next input.")
                
                # Save incomplete feedback
                feedback = f"Incomplete scenario detected:\nMissing: {', '.join(missing) if missing else 'None'}\nQuestions: {' '.join(questions) if questions else 'None'}"
                feedback_filename = os.path.join(JSON_ARTIFACT_DIR, f"feedback_{session_id}_{timestamp}.txt")
                try:
                    with open(feedback_filename, "w", encoding="utf-8") as fh:
                        fh.write(feedback)
                    print(f"Feedback artifact saved to {feedback_filename}")
                except Exception as e:
                    print(f"Failed to save feedback artifact: {e}")
                continue  # Re-prompt   
        except ValueError as e:
            print(f"\nInvalid or incomplete input: {e}")
            print("Please provide a complete scenario (exp: edges for reachability, drones, bidirectional if needed). Try again.")
            
            # Save error
            error_feedback = f"Invalid or incomplete input: {e}"
            error_filename = os.path.join(JSON_ARTIFACT_DIR, f"error_{session_id}_{timestamp}.txt")
            try:
                with open(error_filename, "w", encoding="utf-8") as fh:
                    fh.write(error_feedback)
                print(f"Error feedback artifact saved to {error_filename}")
            except Exception as e:
                print(f"Failed to save error feedback artifact: {e}")
            continue   
        except Exception as e:
            print("Parsing failed:", e)
            error_feedback = f"Parsing failed: {e}"
            error_filename = os.path.join(JSON_ARTIFACT_DIR, f"error_{session_id}_{timestamp}.txt")
            try:
                with open(error_filename, "w", encoding="utf-8") as fh:
                    fh.write(error_feedback)
                print(f"Error feedback artifact saved to {error_filename}")
            except Exception as e:
                print(f"Failed to save error feedback artifact: {e}")
            continue

        if params.get('missing_parameters'):
            print("\nMissing or ambiguous parameters detected (will proceed with defaults).")

        prompt_for_codegen = CODE_GEN_PROMPT + "\n\n# Context (parsed params):\n" + serialize_params(params)

        generated_code_raw = None
        cleaned_code = None
        execution_result = None

        for attempt in range(1, MAX_RETRIES + 1):
            print(f"\nAttempt {attempt}/{MAX_RETRIES} to generate and run solver...")
            try:
                if attempt == 1:
                    generated_code_raw = generate_with_gpt5(prompt_for_codegen, model="gpt-5")
                else:
                    repair_prompt = (
                        f"The previous generated function failed with the output of:\n{execution_result}\n\n"
                        f"Here is the previous code (edit the code to fix the error, ensure the logic is preserved):\n{cleaned_code}\n\n"
                        "Only change lines related to the error. Do not change all the variables, constraints, and output format.\n"
                        "Return the full fixed function def run_z3_solver(params)"
                    )
                    generated_code_raw = generate_with_gpt5(repair_prompt, model="gpt-5")
            except Exception as e:
                print("Code generation failed:", e)
                error_feedback = f"Code generation failed (attempt {attempt}): {e}"
                error_filename = os.path.join(JSON_ARTIFACT_DIR, f"codegen_error_{session_id}_{timestamp}_attempt{attempt}.txt")
                try:
                    with open(error_filename, "w", encoding="utf-8") as fh:
                        fh.write(error_feedback)
                    print(f"Code error artifact saved to {error_filename}")
                except Exception as e:
                    print(f"Failed to save code error artifact: {e}")
                continue

            cleaned_code = clean_code(generated_code_raw)
            # Save CSP artifact
            csp_filename = os.path.join(JSON_ARTIFACT_DIR, f"csp_{session_id}_{timestamp}_attempt{attempt}.py")
            try:
                with open(csp_filename, "w", encoding="utf-8") as fh:
                    fh.write(cleaned_code)
                print(f"CSP artifact saved to {csp_filename}")
            except Exception as e:
                print(f"CSP artifact failed to save: {e}")

            bracket_errors = check_brackets(cleaned_code)
            if bracket_errors:
                print("Bracket errors detected:")
                for err in bracket_errors:
                    print(f"  Line {err['line']} Col {err['col']}: {err['msg']} ({err['type']})")
                execution_result = f"Bracket errors: {bracket_errors}"
                error_filename = os.path.join(JSON_ARTIFACT_DIR, f"bracket_error_{session_id}_{timestamp}_attempt{attempt}.txt")
                try:
                    with open(error_filename, "w", encoding="utf-8") as fh:
                        fh.write(str(execution_result))
                    print(f"Bracket error artifact saved to {error_filename}")
                except Exception as e:
                    print(f"Failed to save bracket error artifact: {e}")
                continue

            try:
                ast.parse(cleaned_code)
            except SyntaxError as e:
                execution_result = f"SyntaxError during ast.parse: {e}"
                print("Syntax error detected:", e)
                error_filename = os.path.join(JSON_ARTIFACT_DIR, f"syntax_error_{session_id}_{timestamp}_attempt{attempt}.txt")
                try:
                    with open(error_filename, "w", encoding="utf-8") as fh:
                        fh.write(str(execution_result))
                    print(f"Syntax error artifact saved to {error_filename}")
                except Exception as e:
                    print(f"Failed to save syntax error artifact: {e}")
                continue

            old_stdout = sys.stdout
            old_stderr = sys.stderr
            sys.stdout = mystdout = StringIO()
            sys.stderr = mystderr = StringIO()
            local_scope = {}
            try:
                exec(cleaned_code, {"__builtins__": safe_builtins, "z3": z3, "combinations": combinations, "time": time}, local_scope)
                if "run_z3_solver" not in local_scope:
                    raise RuntimeError("Generated code did not define run_z3_solver(params)")
                result = local_scope['run_z3_solver'](params)
                execution_result = str(result)
                if not re.match(r'^Satisfiable\n|^Unsatisfiable\n', execution_result):
                    raise RuntimeError("Output format mismatch")
            except Exception as e:
                execution_result = f"Error during exec or run: {e}"
            finally:
                sys.stdout = old_stdout
                sys.stderr = old_stderr
                captured_out = mystdout.getvalue().strip()
                captured_err = mystderr.getvalue().strip()
                # Save model artifact
                model_filename = os.path.join(JSON_ARTIFACT_DIR, f"model_{session_id}_{timestamp}_attempt{attempt}.txt")
                try:
                    with open(model_filename, "w", encoding="utf-8") as fh:
                        fh.write(f"Execution Result:\n{execution_result}\n\n")
                        if captured_out:
                            fh.write(f"Captured stdout:\n{captured_out}\n\n")
                        if captured_err:
                            fh.write(f"Captured stderr:\n{captured_err}")
                    print(f"Model artifact saved to {model_filename}")
                except Exception as e:
                    print(f"Failed to save model artifact: {e}")

            print("\nSolver Result:")
            print(execution_result)

            if isinstance(execution_result, str) and "Satisfiable" in execution_result:
                print("\nSuccess! The plan is ready.")
                break

            print("Attempt failed — requesting repair and retrying...\n")

        else:
            print("\nMax retries exceeded. Please inspect artifacts in", JSON_ARTIFACT_DIR)

        if "Unsatisfiable" in str(execution_result):
            adjustment_prompt = f"The Z3 solver found the problem unsatisfiable for this setup:\n{parsed_json_str}\nExplain why it might have failed and suggest adjustments."
            suggestion = get_gpt_response("You are an expert in optimization and drone routing.", adjustment_prompt)
            print("\nFailure Analysis and Suggestion:")
            print(suggestion)
            
            # Save failure of user interaction artifact
            analysis_filename = os.path.join(JSON_ARTIFACT_DIR, f"analysis_{session_id}_{timestamp}.txt")
            try:
                with open(analysis_filename, "w", encoding="utf-8") as fh:
                    fh.write(suggestion)
                print(f"Failure analysis artifact saved to {analysis_filename}")
            except Exception as e:
                print(f"Failed to save failure analysis artifact: {e}")

            print("\nDo you want to re-run with a new scenario? (y/n)")
            if input().strip().lower() != 'y':
                break

    print("Session ended.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="LLM + Z3 drone mission planner")
    parser.add_argument(
        "--constraints", choices=list(CONSTRAINT_FILES),
        default=os.getenv("DRONE_CONSTRAINTS", "pseudo"),
        help="constraint prompt variant: 'pseudo' (pseudocode) or 'nl' (natural language)",
    )
    args = parser.parse_args()
    main(args.constraints)
