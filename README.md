# LLMs for UAV Resource Distribution in Disasters

An LLM-powered agent that turns a natural-language disaster scenario into a formal
Constraint Satisfaction Problem, then uses the Z3 theorem prover to formally verify
that the generated drone delivery plan satisfies every constraint (capacity, energy,
timing, delivery demands).

Code for my Master's thesis at Auckland University of Technology:
*"LLMs for UAV Resource Distribution in Disasters: Enhancing Trustworthiness through
Formal Verification"* (`docs/thesis.pdf`).

## How it works

1. **Natural-language input** - you describe the scenario (nodes, distances, resources, drones).
2. **LLM parsing** (GPT) - extracts structured parameters, validated with Pydantic (`Edge`, `Drone`, `ProblemParams`).
3. **Code generation** (GPT) - writes a scenario-specific `run_z3_solver(params)` function.
   Syntax/bracket errors trigger an automatic repair loop (up to 5 retries).
4. **Z3 solving** - checks satisfiability and extracts routes (minimise energy, then trips).
5. **Logging** - every stage is saved to `artifacts/` for auditing.

## Run it

```bash
python -m venv venv && source venv/bin/activate    # Windows: venv\Scripts\activate
pip install -r requirements.txt
export OPENAI_API_KEY=...        # PowerShell: $env:OPENAI_API_KEY="..."  (never commit it)

python drone_planner.py                      # pseudocode constraints (default)
python drone_planner.py --constraints nl     # natural-language constraints
```

`DRONE_CONSTRAINTS=nl` works too. The script then asks you to type a scenario at the
prompt (see `examples/sample_scenario.txt`).

## The two constraint variants (the thesis experiment)

The same 34 constraints are given to the LLM in two forms, to compare how reliably it
translates each into Z3:

| Flag | File | Style |
|---|---|---|
| `--constraints pseudo` | `prompts/constraints_pseudocode.txt` | math / logic notation |
| `--constraints nl` | `prompts/constraints_natural_language.txt` | plain English |

In the thesis tests, natural language produced feasible solutions more often than
pseudocode (see sections 7.3-7.4). The chosen variant is recorded in each session's
`user_input_*.txt` artifact.

## Layout

```
drone_planner.py              main pipeline
prompts/                      the two constraint variants
examples/
  generated_solver_sample.py  a solver the pipeline GENERATED (sample output, not hand-written)
  sample_scenario.txt         example text to type at the prompt
  input.json, solve_drones.py a structured problem file + small standalone solver (no Z3/API needed);
                              NOT read by drone_planner.py
reference/dissertPythonTest.py  hand-written Z3 model with Auckland map (needs optional deps)
legacy/llmscript.py           early prototype using a local CodeLlama model
docs/thesis.pdf
```

## Limitations

See thesis section 7.4: bounded domains limit scalability, LLM mistranslation is still
possible, and there is a known issue with how the source node is treated in the
delivery-time-separation constraint.
