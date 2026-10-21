# Project Guide: GNN-based N-1 Contingency Screening

A beginner-friendly guide to this project. It is updated after every step.
Numbers in this guide come only from real runs. Each one says which run produced it.

---

## 1. The project in 5 sentences

1. A power grid is a network of buses (connection points) joined by branches (lines and transformers).
2. Grid operators must check that if any **one** branch fails, the grid still stays within its safe limits. This check is called **N-1 contingency screening**.
3. The exact check needs one power-flow calculation per possible failure, which is slow for big grids and many scenarios.
4. We generate thousands of synthetic scenarios on IEEE test grids with pandapower, label each (scenario, failure) pair as **secure** or **insecure**, and train graph neural networks (GNNs) to predict that label.
5. We compare GNNs against simple rules and classic ML, with leakage-safe splits and 5 seeds, and report honestly where they do and do not help.

**What this is NOT:** it is not dynamic or transient *stability* (how the grid swings in the seconds after a fault). It is **static security**: the steady state after the failure.

---

## 2. Glossary

Each term has a one-line analogy.

| Term | Meaning | Analogy |
|---|---|---|
| **Bus** | A node in the grid where lines, loads and generators connect. | A road junction. |
| **Branch** | Anything that connects two buses: a **line** or a **transformer**. | A road between two junctions. |
| **Transformer** | A branch that changes voltage level (e.g. 345 kV to 138 kV). | A ramp between a highway and a city road. |
| **Load** | Power consumed at a bus (MW = active, MVAr = reactive). | Cars leaving the road network at a junction. |
| **Generator** | Power injected at a bus. | Cars entering at a junction. |
| **Slack bus** | One generator that absorbs whatever imbalance is left, so supply = demand + losses. | The one parking garage that takes whatever overflow there is. |
| **Power flow (PF)** | Solving the physics equations to get every voltage and every branch flow for a given load/generation. | Working out how traffic spreads over all roads given where cars enter and leave. |
| **AC power flow** | The full, nonlinear version (voltages, reactive power, losses). Solved iteratively with Newton-Raphson. | Exact traffic simulation. |
| **DC power flow** | A linear approximation: ignores voltage magnitude and losses, only active power. Fast but cruder. | A quick rule of thumb for traffic. |
| **Newton-Raphson / convergence** | The iterative solver. "Converged" = it found a solution. "Non-converged" = it gave up. | A GPS that either finds a route or says "no route found". |
| **Per-unit (p.u.)** | Values divided by a reference value. 1.0 p.u. voltage = exactly nominal. 0.94 p.u. = 6% below nominal. | Saying "80% full" instead of "40 litres". |
| **Rating** | The maximum power a branch may carry continuously (MVA). | A road's capacity in cars/hour. |
| **Loading** | Flow / rating. Above 100% = overload. | Traffic as a % of road capacity. |
| **Contingency** | An unplanned outage of one element (here: one branch). | One road suddenly closed. |
| **N-1** | "The grid with N elements must survive losing any 1 of them." | The city must still work if any single road closes. |
| **Islanding** | An outage that splits the grid into two disconnected parts. | Closing the only bridge to an island. |
| **Operating point (OP)** | One snapshot of all loads and generator outputs. | Traffic at one moment of the day. |
| **Secure / insecure** | Our label. Defined exactly in Section 5, decision D1. | After the road closure, is traffic still within limits everywhere? |
| **Static security vs stability** | Security = is the *new steady state* within limits? Stability = does the grid *survive the transient swing* to get there? We only do security. | Security: "is the final traffic jam acceptable?" Stability: "did cars crash during the sudden rerouting?" |
| **IEEE test case** | Standard, public, made-up grids (14, 30, 39, 118 buses) used by researchers. Not real grids. | A standard driving-test course. |
| **MATPOWER** | A popular power-system tool whose case files pandapower loads. | The source of the test courses. |
| **pandapower** | Python library that runs power flow. | Our traffic simulator. |
| **Synthetic data** | Data we generate from a simulator instead of measuring it. | Practice exam questions you write yourself. |
| **Load factor** | How much all loads are scaled versus the case file (0.8 = 80% of nominal demand). | "Rush hour is 1.3, night is 0.7." |
| **Class imbalance** | One label is much rarer than the other (here insecure ≈ 12%). A model can score 88% accuracy by always saying "secure". | A smoke alarm that never rings is "right" most days. |
| **Reactive power (MVAr)** | Power that sloshes back and forth to keep voltages up. It does no useful work but still uses line capacity. | Foam on a beer: takes up glass space, isn't beer. |

More terms (GNN, message passing, PR-AUC, leakage, ...) are added on the day we use them.

---

## 3. What we did, day by day, and WHY

### Day 1: inspect the grids, lock the label

- **Set up a pinned environment.** Every library version is fixed in `requirements.txt`. *Why:* pandapower's case files and PyTorch Geometric change between versions, and results must be reproducible.
- **Inspected the four candidate grids** (`python -m pgs.inspect_cases`). We looked at size, whether the branch ratings in the file are real, which outages island the grid, and whether the default operating point is already within limits. *Why:* a label is only meaningful if the limits it checks are meaningful.
- **Found that case14 and case118 have no real ratings.** Every branch says 9900 MVA, which MATPOWER uses to mean "no limit". Without a fix, "overload" could never happen on them. We chose a planning-style synthetic rating (decision D3).
- **Found that case39 is unusable with our label.** One generator holds its voltage at 1.064 p.u., above our 1.06 limit, so every sample would be "insecure". We dropped it (D5).
- **Wrote the label code and 7 tests** (`pgs/grid.py`, `tests/test_grid.py`). The tests check, on a hand-built 3-bus grid, that overloads, low voltages and solver failures are all labelled insecure, and that islanding outages are detected. *Why:* if the label is wrong, every result after it is wrong.

### Day 2: generate the case30 dataset

- **Sampled random operating points** (`pgs/generate.py`). Each one scales all loads by a global factor drawn uniformly from 0.7 to 1.3, then by a per-bus random factor (about ±7%). Generators are redispatched in proportion, with ±10% noise. *Why:* real grids never sit at one fixed load. We need variety so the model learns patterns, not one snapshot.
- **Kept only operating points that are secure before any outage** (D7). Only 38% of random points passed, mostly the lower-load ones, because case30's default point already overloads a line.
- **For each kept point, ran all 38 non-islanding outages** with full AC power flow, labelling each one. That gives 300 × 38 = 11,400 samples. *Why:* this is exactly the N-1 check an operator runs.
- **Also stored a DC power-flow estimate** for every outage, used only by a rule baseline later, never as a model input. *Why:* "a cheap physics approximation" is the fair baseline a GNN must beat.
- **Stored inputs and labels in separate files.** Inputs (`arrays.npz`) hold only pre-outage quantities. Post-outage results (`samples.parquet`) hold the label and diagnostics. A test checks that the input file has no extra keys. *Why:* this is the main protection against data leakage (defined on Day 3).
- **Checked determinism.** A test runs generation twice with different worker counts and compares a hash of the labels. *Why:* "reproducible" must be checked, not just claimed.

---

## 4. Worked examples (real numbers)

Produced by `PYTHONPATH=. python scripts/guide_examples.py`.

### Example 1: one N-1 outage on a 4-bus ring

Four buses in a ring 0-1-2-3-0. Bus 0 is the slack (the supply). Loads: 20 MW on bus 1, 40 MW on bus 2, 20 MW on bus 3. Every line is rated 50 MVA.

All lines in service:

| Line | Flow | Loading |
|---|---|---|
| 0-1 | 41.4 MVA | 83% |
| 1-2 | 20.5 MVA | 41% |
| 2-3 | 20.5 MVA | 41% |
| 3-0 | 41.4 MVA | 83% |

Voltages are 1.000, 0.987, 0.981, 0.987 p.u., all inside [0.94, 1.06]. Result: **secure**.

Now line 3-0 trips (the N-1 contingency). All power must now go around the other side:

| Line | Flow | Loading |
|---|---|---|
| 0-1 | 85.4 MVA | **171%** |
| 1-2 | 62.6 MVA | **125%** |
| 2-3 | 20.5 MVA | 41% |
| 3-0 | out | out |

Voltages drop to 1.000, 0.972, 0.952, 0.946, still above 0.94. But two lines are overloaded, so the result is **insecure**.

**Lesson:** the outage of one line moved flow onto *other* lines. A model must understand *where* the outage is relative to everything else. That is why a graph model is a natural fit.

### Example 2: one GNN message-passing step (toy weights)

Each bus has two features: [voltage, load]. Bus 2 = [0.95, 0.4]. Its neighbours are bus 1 = [0.98, 0.2] and bus 3 = [0.97, 0.2].

1. **Aggregate:** mean of the neighbours = [0.975, 0.2].
2. **Combine:** new bus 2 = ReLU(1.0 × own + 0.5 × neighbour mean) = ReLU([0.95 + 0.4875, 0.4 + 0.1]) = **[1.438, 0.5]**.

In a real GNN, the weights 1.0 and 0.5 are learned matrices, not fixed numbers. After 2 such steps, each bus "knows" about buses 2 hops away.

---

## 5. Decisions log

| # | Decision | Why | Alternatives considered |
|---|---|---|---|
| D1 | **Label:** a (operating point, single-branch outage) pair is **insecure** if the post-outage AC power flow (with generator reactive limits enforced) has any bus voltage outside [0.94, 1.06] p.u., OR any branch loading above 100%, OR fails to converge after 3 attempts (warm start from the pre-outage solution, flat start, DC start). Otherwise **secure**. | Standard static-security criteria. Agreed with the user on Day 0. | Using each case file's own voltage limits (they differ per case, e.g. case30 uses 0.95-1.05); treating non-convergence as a separate class. |
| D2 | **Loading = apparent power (MVA) at the worse end of the branch / rating (MVA).** | MATPOWER ratings are in MVA. pandapower's own `loading_percent` is current-based, which reads higher when voltage is low (case30 line 5-7: 112% current-based vs 109% MVA-based at the default point). | pandapower's current-based loading. |
| D3 | **Synthetic ratings for branches without a real one:** rating = 1.1 × the largest flow that branch carries over the nominal case and every non-islanding N-1 outage. | Mimics planning: the nominal grid is N-1 secure by construction, and insecurity appears as load and generation move. A simpler rule (1.2-1.5 × base flow with a median floor) gave 36-61% insecure outages even at nominal load, which mostly reflected an arbitrary margin. | Fixed multiple of base flow; leaving ratings unlimited (then only voltage could fail). |
| D4 | **Grids:** case30 (real ratings, small, main debugging grid) and case118 (synthetic ratings, main large grid). case14 is for debugging only. | case30 is the only small case here with real ratings. case118 is where a GNN should matter. | IEEE-30 original (`case_ieee30`, no real ratings). |
| D5 | **Drop case39 and the unseen-grid test (S3).** | A generator setpoint of 1.064 p.u. breaks our 1.06 limit at every operating point. It was first on the cut list anyway. | Use case39's own limits, which would mean a different label per grid. |
| D6 | **Islanding outages are excluded** (3 in case30, 9 in case118) and counted. | After islanding, part of the grid has no slack bus. That is a different problem (loss of load), not a power-flow limit check. | Label them all insecure (trivially predictable). |
| D7 | **Only operating points that are secure before the outage are kept.** | The question is "does this outage break a currently secure grid?". | Keep all (then many labels are just "already broken"). |
| D8 | **Python dict configs instead of YAML files.** | Fewer moving parts for a 9-day project. | YAML files plus a loader. |
| D9 | **Scenario sampling:** global load factor ~ U(0.7, 1.3) × per-bus N(1, 0.07). Reactive load scales with active load. Generator P is set proportional to total load × U(0.9, 1.1), clipped to its limits, and the slack covers the rest. Seed 0. | Simple, standard and documented. It covers light to heavy load. | Real load profiles (rejected: mapping them onto IEEE buses is arbitrary); optimal power flow dispatch (slower, more complex). |
| D10 | **Sample counts:** case30 has 300 operating points × 38 outages. case118 has 120 × 177. | Enough samples for a stable 5-seed comparison, while generation stays at minutes. | More points with fewer outages per point. |
| D11 | **Node features (10 per bus):** load P/Q, generator P/Q, voltage setpoint, bus type (slack/PV/PQ), pre-outage voltage magnitude and angle. **Branch features:** r, x, b, rating, is-transformer (fixed), plus pre-outage loading and active flow. All in p.u. on 100 MVA. | Everything an operator knows *before* the outage. Pre-outage power-flow results are legitimate inputs because operators already have them. | Raw inputs only (no pre-outage PF), which would be harder for every model. |
| D12 | **Generator outages are not included.** | First on the cut list. Branch outages already give the core problem. | Add the 53 case118 generator outages. |

---

## 6. Results so far

### Day 1: grid inspection (container run, 21/10)

From `python -m pgs.inspect_cases`. Ratings use D3 where the file has none. Max load = worst branch loading at the default operating point.

| Case | Buses | Lines | Trafos | Gens | Real ratings | Islanding outages | Vmin | Vmax | Max load | Secure at default? |
|---|---|---|---|---|---|---|---|---|---|---|
| case14 | 14 | 15 | 5 | 5 | 0/20 | 1 | 1.010 | 1.090 | 0.720 | No (V > 1.06) |
| case30 | 30 | 41 | 0 | 6 | 41/41 | 3 | 0.961 | 1.000 | 1.088 | No (overload) |
| case39 | 39 | 35 | 11 | 10 | 46/46 | 11 | 0.982 | 1.064 | 0.764 | No (V > 1.06) |
| case118 | 118 | 173 | 13 | 54 | 0/186 | 9 | 0.943 | 1.050 | 0.909 | Yes |

"Gens" includes the slack. AC power-flow time per run was 43-75 ms here (no GPU, 4 CPUs).

**What this means:** case30's default point overloads a line, so its usable operating points will come from lower loads. That is fine, because we keep only secure points (D7).

### Day 2: case30 dataset (container run, 21/10, label hash `42676ea248`)

| Quantity | Value |
|---|---|
| Candidate operating points tried | 785 |
| Secure before outage (kept) | 38.2%, first 300 kept |
| Outages per point | 38 (3 islanding outages excluded) |
| Samples | 11,400 |
| **Insecure** | **12.1%** |
| ... because of overload | 6.4% |
| ... because of voltage outside [0.94, 1.06] | 8.0% (a sample can have both) |
| Non-converged | 0 |
| Load factor of kept points | 0.70 to 1.07 (median 0.82) |
| Insecure rate per operating point | 2.6% to 47.4% (median 10.5%) |
| Outages insecure at *every* point / at *no* point | 1 / 12 |

**What this means:**
- The data is imbalanced (12% insecure), so accuracy alone would mislead.
- 12 of 38 outages are never insecure and 1 is always insecure. A model can get part of the way just by memorising "which outage is dangerous". We will add a baseline that does exactly that, so a GNN is not credited for it.
- Non-convergence never happened on case30, so the "with/without non-converged" ablation means nothing here.
- The DC estimate of loading read about 0.70 where AC read 0.98 on one point. DC ignores reactive power, so a DC rule needs a threshold tuned on training data, not a fixed 1.0.

---

## 7. Limitations: what we can NOT claim

- **Not stability.** No dynamics, no transient or voltage-collapse simulation. Only steady-state limit checks.
- **Not real grids.** IEEE test cases are standard synthetic benchmarks.
- **case118 ratings are synthetic** (D3). The overload part of its label depends on that rule.
- **Single-branch outages only.** No generator outages and no N-2.
- **Non-convergence ≠ collapse.** The solver can fail for numerical reasons. We will report how often it happens.

---

## 8. "If an interviewer asks..."

1. **Is this stability prediction?** No. It is static security assessment, meaning the steady state after an outage. Stability would need time-domain simulation of generator dynamics. I renamed the project to say that.
2. **Why IEEE test grids and not real data?** Real grid models are confidential. IEEE cases are the standard public benchmarks, so results are comparable and reproducible.
3. **Where do your line ratings come from?** case30 has real ratings in its file. case118's file has none (9900 MVA placeholders), so I set each rating to 1.1 × the worst flow the branch sees across all N-1 outages at nominal load. That makes the nominal grid N-1 secure, as a planned grid would be.
4. **Why drop islanding outages?** After islanding, part of the grid has no generator to balance it, so power flow is not defined in the same way. That is a different problem. I count them and exclude them.
5. **Why is non-convergence labelled insecure?** In practice an operator cannot trust that state. But it can also be a numerical failure, so I retry from 3 starting points and report the rate. On case30 it never happened.
6. **How did you generate scenarios?** I scaled all loads by a random factor from 0.7 to 1.3 with per-bus noise, redispatched generators proportionally, kept only points that are secure before the outage, then ran every single-branch outage with AC power flow.
7. **What is your class balance?** On case30, 12.1% insecure out of 11,400 samples. So I report recall, F1 and PR-AUC on the insecure class, not accuracy.
8. **How do you know the pipeline is reproducible?** Seeds are fixed, versions are pinned, and a test runs generation twice with different numbers of parallel workers and checks that the label hash is identical.

---

## 9. Resume bullet (draft, rewritten at the end from real results)

> *Placeholder until real results exist. No numbers yet.*
> GNN-based N-1 contingency screening (static security assessment) on IEEE 30/118-bus grids: built a pandapower pipeline generating labelled (operating point, outage) samples; compared GCN/GraphSAGE/GAT against rule-based and classic ML baselines with leakage-safe splits.
