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
| **Class imbalance** | One label is much rarer than the other (here insecure ≈ 12%). A model can score 88% accuracy by always saying "secure". | A smoke alarm that never rings is "right" most steps. |
| **Reactive power (MVAr)** | Power that sloshes back and forth to keep voltages up. It does no useful work but still uses line capacity. | Foam on a beer: takes up glass space, isn't beer. |
| **Feature** | One input number the model sees (e.g. the load at bus 7). | One clue a detective gets. |
| **Label** | The answer the model must predict (secure = 0, insecure = 1). | The solution at the back of the book. |
| **Train / validation / test set** | Train: the model learns from it. Validation: used to pick settings (when to stop, which threshold). Test: touched once, at the end, to report results. | Homework, mock exam, final exam. |
| **Data leakage** | When information about the test answers sneaks into training or into the inputs, so scores look better than real-world performance. | Seeing the final exam questions while doing homework. |
| **Graph** | Nodes joined by edges. Here, nodes = buses and edges = in-service branches. | A metro map: stations and tracks. |
| **One-hot vector** | A list of zeros with a single 1 marking which item is meant (e.g. which branch is out). | A row of light switches with exactly one on. |
| **Standardising (scaling)** | Shifting and stretching each feature so it has mean 0 and spread 1, using training data only. | Converting all prices to the same currency before comparing. |
| **Baseline** | A simpler method a fancy model must beat to be worth using. | Before buying a GPS, check whether a paper map already gets you there. |
| **Rule-based baseline** | A hand-written rule with no learning, e.g. "insecure if DC loading > threshold". | A rule of thumb. |
| **Logistic regression (LR)** | A weighted sum of features pushed through an S-curve to give a probability. Straight-line decision boundary. | A checklist where each item has points and you add them up. |
| **Gradient boosting (LightGBM)** | Hundreds of small decision trees, each one fixing the mistakes of the ones before it. Strong on tables of numbers. | A committee where each new member focuses on the cases the others got wrong. |
| **MLP (multi-layer perceptron)** | A basic neural network: layers of weighted sums with nonlinear "bends" between them. | Many stacked checklists, where each feeds the next. |
| **Score and threshold** | Models output a score. We call a sample insecure if score ≥ threshold. | A fever thermometer and the "38 °C = fever" cut-off. |
| **Precision** | Of the samples we flagged insecure, the share that really were. | Of the alarms that rang, how many were real fires. |
| **Recall** | Of the truly insecure samples, the share we caught. **The most important number here: a missed insecure case is the dangerous error.** | Of the real fires, how many set off the alarm. |
| **F1** | A single number that balances precision and recall (their harmonic mean). | A grade that is only high if both scores are high. |
| **PR-AUC (average precision)** | Precision averaged over every possible threshold. 1.0 = perfect. **Random guessing scores the base rate** (≈ 0.09-0.12 here), not 0.5. | Grading the alarm at every sensitivity setting, not just one. |
| **Base rate** | The share of insecure samples in a set. | How common fires are in the first place. |
| **Shuffled-label check** | Train on randomly shuffled labels. A sound pipeline must then score at chance. If it scores well, something leaks. | Giving a student a random answer key. If they still "pass", the exam is broken. |
| **GNN (graph neural network)** | A neural network that works directly on a graph. Each node keeps a vector of numbers and updates it using its neighbours' vectors. | Each station on the metro map updates its "status" by listening to the stations next to it. |
| **Message passing** | One GNN layer: every node collects messages from its neighbours, combines them (sum, mean or weighted) and updates itself. *k* layers = information travels *k* hops. | Gossip spreading one street further each round. |
| **GCN** | Graph Convolutional Network: averages neighbours with fixed weights based on how many connections each node has. | Everyone's opinion counts equally, adjusted for how talkative they are. |
| **GraphSAGE** | Keeps a node's own vector separate from the mean of its neighbours, then combines the two. | "What I think" + "what my neighbours think on average". |
| **GAT / GATv2 (attention)** | Learns *how much* to listen to each neighbour. Our GATv2 also reads branch features (impedance, rating, pre-outage flow) when deciding. | Listening more to the neighbour on the busiest road. |
| **Pooling (readout)** | Turns all node vectors into one vector for the whole graph. We use mean + max. Max matters because one overloaded spot makes the whole sample insecure. | Summarising a city by its average traffic and its worst jam. |
| **Epoch** | One full pass over the training data. | Reading the whole textbook once. |
| **Early stopping** | Stop training when the validation score has not improved for 8 epochs, and keep the best version. Prevents memorising the training set (overfitting). | Stop studying when mock-exam scores stop rising. |
| **Seed** | The number that fixes all randomness (split, initial weights, batch order). Different seeds = different, equally valid runs. | Shuffling a deck with a recorded shuffle. |
| **Mean ± std over seeds** | Average over 5 seeds, ± their standard deviation. It shows how much a result depends on luck. | Five test attempts: your average and how much it varies. |

More terms are added at the step where we first use them.

---

## 3. What we did, step by step, and WHY

### Step 1: inspect the grids, lock the label

- **Set up a pinned environment.** Every library version is fixed in `requirements.txt`. *Why:* pandapower's case files and PyTorch Geometric change between versions, and results must be reproducible.
- **Inspected the four candidate grids** (`python -m pgs.inspect_grids`). We looked at size, whether the branch ratings in the file are real, which outages island the grid, and whether the default operating point is already within limits. *Why:* a label is only meaningful if the limits it checks are meaningful.
- **Found that case14 and case118 have no real ratings.** Every branch says 9900 MVA, which MATPOWER uses to mean "no limit". Without a fix, "overload" could never happen on them. We chose a planning-style synthetic rating (decision D3).
- **Found that case39 is unusable with our label.** One generator holds its voltage at 1.064 p.u., above our 1.06 limit, so every sample would be "insecure". We dropped it (D5).
- **Wrote the label code and 7 tests** (`pgs/power_flow.py`, `tests/test_power_flow.py`). The tests check, on a hand-built 3-bus grid, that overloads, low voltages and solver failures are all labelled insecure, and that islanding outages are detected. *Why:* if the label is wrong, every result after it is wrong.

### Step 2: generate the case30 dataset

- **Sampled random operating points** (`pgs/generate_dataset.py`). Each one scales all loads by a global factor drawn uniformly from 0.7 to 1.3, then by a per-bus random factor (about ±7%). Generators are redispatched in proportion, with ±10% noise. *Why:* real grids never sit at one fixed load. We need variety so the model learns patterns, not one snapshot.
- **Kept only operating points that are secure before any outage** (D7). Only 38% of random points passed, mostly the lower-load ones, because case30's default point already overloads a line.
- **For each kept point, ran all 38 non-islanding outages** with full AC power flow, labelling each one. That gives 300 × 38 = 11,400 samples. *Why:* this is exactly the N-1 check an operator runs.
- **Also stored a DC power-flow estimate** for every outage, used only by a rule baseline later, never as a model input. *Why:* "a cheap physics approximation" is the fair baseline a GNN must beat.
- **Stored inputs and labels in separate files.** Inputs (`arrays.npz`) hold only pre-outage quantities. Post-outage results (`samples.parquet`) hold the label and diagnostics. A test checks that the input file has no extra keys. *Why:* this is the main protection against data leakage (defined on Step 3).
- **Checked determinism.** A test runs generation twice with different worker counts and compares a hash of the labels. *Why:* "reproducible" must be checked, not just claimed.

### Step 3: case118 data, splits, graphs

- **Generated the case118 dataset:** 120 operating points × 177 outages = 21,240 samples. *Why:* case30 is small. A GNN's advantage, if any, should show on the bigger grid.
- **Built two leakage-safe splits** (`pgs/train_test_splits.py`). *Why:* a random split per sample would put the *same operating point* in both train and test, just with a different outage. The model could then recognise the operating point instead of learning physics.
  - **S1 "unseen operating points":** operating points are divided 70/15/15. No point appears in two sets, but every outage appears everywhere.
  - **S2 "unseen outages":** operating points are divided 70/15/15 *and* outages are dealt into 5 folds. Seed *s* tests on fold *s*, validates on fold *s*+1 and trains on the other 3 (so 60/20/20). Over the 5 seeds every outage is tested exactly once. Test samples combine new operating points with outages the model has *never* seen. This is the harder and more interesting test. (Changed on Step 5, see D20.)
- **Built the graph view of each sample** (`pgs/features.py`). Buses are nodes with 10 pre-outage features, plus a flag that is 1 on the two buses at the ends of the outaged branch (from Step 5 also the tripped branch's flow and loading on those two buses, D17). In-service branches are edges in both directions. The outaged branch is **removed**. *Why:* the GNN should "see" the hole in the grid.
- **Built the flat view for classic models.** All bus features, all branch features, a one-hot vector saying which branch is out, and (from Step 5) the outaged branch's pre-outage flow and loading, joined into one long list (case30: 425 numbers; case118: 1,740). *Why:* logistic regression, gradient boosting and MLPs need a fixed-length input.
- **Scaling uses training data only.** *Why:* computing the mean and spread over all data would leak a little information about the test set into training.
- **Tests (16)** check that splits never share an operating point (S1) or an outage (S2), that the outaged branch is really removed, that reversed edges flip the sign of power flow, and that the scaler only sees training rows.

### Step 4: baselines

- **Four rules with no real learning** (`pgs/train_and_evaluate.py`). *Why:* they show how much of the problem is trivial.
  - `majority`: always says secure.
  - `cont_prior`: "how often was this outage insecure in training?" It memorises dangerous outages and nothing else.
  - `base_loading`: the worst branch loading *before* the outage.
  - `dc_rule`: the worst branch loading *after* the outage, from a fast DC power flow.
- **Three classic ML models on the flat view:** logistic regression, LightGBM and an MLP. *Why:* a GNN must beat ordinary ML, not just rules.
- **Every model's threshold is tuned on validation** (maximise F1). The test set is used once, for reporting (D16).
- **Metrics focus on the insecure class:** PR-AUC, recall, precision and F1. *Why:* with 9-12% insecure, "always secure" gets 88-91% accuracy and catches nothing.
- **Leakage check on a suspicious number.** LightGBM scored 0.995 PR-AUC on case30 S1 (0.997 in the re-run after D17), which is too good to trust without checking. Two checks:
  1. **Shuffled labels:** LightGBM then scores 0.120 PR-AUC (0.106 in the re-run after D17), close to the 0.112 base rate. So the pipeline does not leak labels.
  2. **Feature importance:** the model leans mostly on *which outage* it is (4 outages carry 52% of importance), then on load at bus 7 and pre-outage voltages. That matches the physics. A few outages are dangerous, and how dangerous depends on how stressed the grid is.

  Conclusion: S1 is genuinely easy, not leaky. S2 is the real test.

### Step 5: graph neural networks

- **Fixed a design flaw before training any GNN.** Removing the tripped branch from the graph also removed its pre-outage flow, which is the power that must reroute. We now put that information on the branch's two end buses as an "equivalent injection" (+P at one end, −P at the other), using the compensation idea from power-system textbooks (D17). The flat models get the same numbers, and all baselines were re-run. *Why:* a fair comparison needs every model to have the same information.
- **Built three GNNs with one shared skeleton** (`pgs/models.py`, D19). GCN and GraphSAGE see bus features and topology. GATv2 additionally reads branch features (impedance, rating, pre-outage flow) when deciding how much to listen to each neighbour. *Why:* these are the three standard GNN families. Same skeleton = fair comparison between them.
- **Trained with class-weighted loss and early stopping on validation PR-AUC** (D18). *Why:* the weighting stops the model ignoring the rare insecure class. Early stopping stops it memorising the training set.
- **Seed 0 only, as a working check.** The 5-seed comparison is Step 6.

---

## 4. Worked examples (real numbers)

Produced by `PYTHONPATH=. python scripts/print_guide_examples.py`.

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

**The "equivalent injection" idea (D17), using the same numbers:** before the outage, line 3-0 carried 41.4 MVA. When it trips, that power has to find another path. Line 0-1 goes from 41.4 to 85.4 MVA, roughly its own old flow plus the tripped line's flow (41.4 + 41.4 = 82.8). The rest is extra losses and reactive power on the longer path. So "how much was the tripped line carrying, and where were its ends?" is the key information, and that is exactly what we give the model on the two end buses.

### Example 2: one GNN message-passing step (toy weights)

Each bus has two features: [voltage, load]. Bus 2 = [0.95, 0.4]. Its neighbours are bus 1 = [0.98, 0.2] and bus 3 = [0.97, 0.2].

1. **Aggregate:** mean of the neighbours = [0.975, 0.2].
2. **Combine:** new bus 2 = ReLU(1.0 × own + 0.5 × neighbour mean) = ReLU([0.95 + 0.4875, 0.4 + 0.1]) = **[1.438, 0.5]**.

In a real GNN, the weights 1.0 and 0.5 are learned matrices, not fixed numbers. After 2 such steps, each bus "knows" about buses 2 hops away.

### Example 3: why accuracy misleads, and what PR-AUC measures

Ten samples, 2 insecure (base rate 0.2). Model scores: 0.9, 0.8, 0.3, 0.2, 0.1, 0.1, 0.05, 0.05, 0, 0. The truly insecure samples are the 1st (0.9) and 3rd (0.3).

| Method | Precision | Recall | F1 | Accuracy |
|---|---|---|---|---|
| "Always secure" | (none flagged) | 0.00 | 0.00 | **0.80** |
| Model, threshold 0.5 | 0.50 | 0.50 | 0.50 | 0.80 |
| Model, threshold 0.25 | 0.67 | 1.00 | 0.80 | 0.90 |

- "Always secure" has 80% accuracy and catches **zero** insecure cases.
- Lowering the threshold from 0.5 to 0.25 catches both insecure cases at the cost of one false alarm. That is the right trade for safety.
- The model's **PR-AUC is 0.833**, scored over all thresholds. Random scores would give about **0.2** (the base rate), not 0.5.

---

## 5. Decisions log

| # | Decision | Why | Alternatives considered |
|---|---|---|---|
| D1 | **Label:** a (operating point, single-branch outage) pair is **insecure** if the post-outage AC power flow (with generator reactive limits enforced) has any bus voltage outside [0.94, 1.06] p.u., OR any branch loading above 100%, OR fails to converge after 3 attempts (warm start from the pre-outage solution, flat start, DC start). Otherwise **secure**. | Standard static-security criteria. Agreed with the user on Step 0. | Using each case file's own voltage limits (they differ per case, e.g. case30 uses 0.95-1.05); treating non-convergence as a separate class. |
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
| D13 | **Two splits.** S1: operating points 70/15/15. S2: operating points 70/15/15 × outages 60/20/20 (5 rotating outage folds since D20). The operating-point split is redrawn for each seed. | S1 checks "new conditions, known outages". S2 checks "new conditions, new outages", which is where a graph model should help. Redrawing per seed puts split luck into the ± numbers. | Random split per sample (leaks); one fixed split (hides variance); keeping operating points shared in S2 (larger test set, but the model would have seen the test operating point). |
| D14 | **Outage = remove the branch from the graph + flag its two end buses.** | The model sees the changed topology directly, and the flag tells it *where* the change is. | Keep the edge with an "out" feature (GCN/SAGE cannot read edge features). |
| D15 | **GCN and GraphSAGE use bus features and topology only. GATv2 also reads branch features** (r, x, b, rating, pre-outage loading and flow). | These are the standard forms of each layer. | Give all three edge features. Not possible for plain GCN without changing what "GCN" means. A Step 7 ablation turns GAT's edge features off to measure their effect. |
| D16 | **Decision threshold is tuned on validation for every model** (maximise F1), including the rules. | Fair: no model gets a hand-picked threshold. The test set is never used to choose anything. | Fixed 0.5 (unfair to rules and to imbalanced models). |
| D17 | **Outage descriptor features (added on Step 5, before any GNN result existed).** The two end buses of the tripped branch get: a flag, an "equivalent injection" (+P at the sending end, −P at the receiving end, where P is the branch's pre-outage active flow), and the branch's pre-outage loading. Flat models get the same two numbers (P and loading) as extra columns. All baselines were re-run afterwards. | Removing the branch from the graph also removed its pre-outage flow, the power that must reroute and the most important single number. Physics gives a clean fix (the compensation method behind outage distribution factors): tripping a branch that carried P is equivalent to injecting +P and −P at its ends into the remaining grid. Giving the flat models the same numbers keeps the comparison fair. | Keep the branch in the graph with an "out" marker (GCN/SAGE can't read edge features); give only the flag (loses the key quantity). |
| D18 | **Training setup for MLP and GNNs:** cross-entropy loss with the insecure class up-weighted by (#secure / #insecure) in training; Adam, learning rate 1e-3; batch 128 graphs (256 for the MLP); at most 60 epochs; stop after 8 epochs without validation PR-AUC improvement and keep the best epoch. | Up-weighting handles the class imbalance without throwing data away. The epoch cap keeps the 5-seed sweep affordable. It was set from timing (2-30 s per epoch on CPU), before any GNN result. | Oversampling insecure samples; focal loss; longer training. |
| D19 | **GNN shape:** linear encoder → 3 message-passing layers (64 units, residual connection + layer norm) → mean + max pooling → 2-layer head. Same for GCN, GraphSAGE and GATv2 (4 attention heads). | One shared skeleton, so differences come from the layer type. 3 layers = information travels 3 buses away. Max pooling matches the label, which is "any single violation". | Deeper models (tested as an ablation on Step 7); attention pooling. |
| D20 | **S2 outages rotate through 5 stratified folds (changed on Step 5).** Outages are sorted by how often they are insecure and dealt round-robin into 5 fixed folds. Seed *s* tests fold *s*, validates on fold *s*+1, trains on the rest. All models were re-run on the new S2. | **Why the change, honestly:** after the first GNN runs I inspected the case30 S2 seed-0 test set and found that only 1 of its 7 outages was ever insecure, so its score measured a single outage (which the DC rule happened to catch). I had flagged this risk on Step 3. Rotation tests every outage exactly once over the 5 seeds, and stratification gives each fold a mix of dangerous and harmless outages. The change treats all models identically and uses labels only to balance folds, never to train. | Keep random 60/20/20 per seed (a noisy, partial test); leave-one-outage-out (38-177 folds, too expensive). |
| D21 | **Step 7 ablations run on case118 unseen outages (S2) only**, with GAT (chosen on validation), 5 seeds; the shuffled-label check also runs once on known operating points. | S2 is where the GNNs led, so that is what needs explaining. Running every ablation on both splits would take 8+ hours on this CPU. | All ablations on both splits; ablations on all three GNNs. |

---

## 6. Results so far

### Step 1: grid inspection (container run, 21/10)

From `python -m pgs.inspect_grids`. Ratings use D3 where the file has none. Max load = worst branch loading at the default operating point.

| Case | Buses | Lines | Trafos | Gens | Real ratings | Islanding outages | Vmin | Vmax | Max load | Secure at default? |
|---|---|---|---|---|---|---|---|---|---|---|
| case14 | 14 | 15 | 5 | 5 | 0/20 | 1 | 1.010 | 1.090 | 0.720 | No (V > 1.06) |
| case30 | 30 | 41 | 0 | 6 | 41/41 | 3 | 0.961 | 1.000 | 1.088 | No (overload) |
| case39 | 39 | 35 | 11 | 10 | 46/46 | 11 | 0.982 | 1.064 | 0.764 | No (V > 1.06) |
| case118 | 118 | 173 | 13 | 54 | 0/186 | 9 | 0.943 | 1.050 | 0.909 | Yes |

"Gens" includes the slack. AC power-flow time per run was 43-75 ms here (no GPU, 4 CPUs).

**What this means:** case30's default point overloads a line, so its usable operating points will come from lower loads. That is fine, because we keep only secure points (D7).

### Step 2: case30 dataset (container run, 21/10, label hash `42676ea248`)

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

### Step 3: case118 dataset (container run, 21/10, label hash `97790e93b9`)

| Quantity | case30 | case118 |
|---|---|---|
| Samples | 11,400 | 21,240 |
| Operating points × outages | 300 × 38 | 120 × 177 |
| Islanding outages excluded | 3 | 9 |
| Candidates secure before outage | 38.2% | 46.3% |
| **Insecure** | **12.1%** | **9.1%** |
| ... overload / voltage | 6.4% / 8.0% | 3.2% / 6.1% |
| Non-converged | 0 | 0 |
| Outages insecure at every point / at no point | 1 / 12 | 5 / 53 |
| Generation time (4 CPUs) | 138 s | 693 s |

Split sizes (seed 0) and insecure rate of the **test** set across the 5 seeds (S2 with the rotating folds of D20):

| Grid | Split | Train / val / test | Test insecure % (5 seeds) |
|---|---|---|---|
| case30 | S1 | 7,980 / 1,710 / 1,710 | 11.1-12.3 |
| case30 | S2 | 4,620 / 360 / 360 | 6.7-16.7 |
| case118 | S1 | 14,868 / 3,186 / 3,186 | 6.8-9.3 |
| case118 | S2 | 8,820 / 648 / 648 | 6.5-9.7 |

S2 test folds contain 2-5 outages that are ever insecure on case30 (21-60 insecure test samples per fold), and 7-12 on case118 (41-63 insecure samples).

**What this means:**
- S2 test sets are small (360 and 648 samples), so S2 scores will swing between seeds. That is why we run 5 seeds and report ± spread.
- The first version of S2 (random 60/20/20 per seed) was worse. In case30 seed 0, only 1 of the 7 test outages was ever insecure, and all 33 insecure test samples came from it. Another seed had only 1.9% insecure test samples. D20 fixed that.
- S2 also uses less data for training (only 60% of outages × 70% of points).

### Step 4: baselines, seed 0 only (container run, CPU, 22/10; re-run after D17 and D20)

Test set. PR-AUC / recall / F1 for the insecure class. **One seed only:** treat as a first look, not a result. The 5-seed table comes on Step 6. These numbers use the outage descriptor (D17) and, for S2, the rotating outage folds (D20). Seed 0 tests outage fold 0.

| Model | case30 S1 | case30 S2 | case118 S1 | case118 S2 |
|---|---|---|---|---|
| Base rate (= random PR-AUC) | 0.112 | 0.167 | 0.086 | 0.097 |
| majority | 0.112 / 0.00 / 0.00 | 0.167 / 0.00 / 0.00 | 0.086 / 0.00 / 0.00 | 0.097 / 0.00 / 0.00 |
| cont_prior | 0.784 / 0.97 / 0.67 | 0.167 / 1.00 / 0.29 | 0.798 / 0.66 / 0.74 | 0.097 / 1.00 / 0.18 |
| base_loading | 0.186 / 0.46 / 0.29 | 0.219 / 0.37 / 0.30 | 0.143 / 0.23 / 0.20 | 0.150 / 0.21 / 0.19 |
| dc_rule | 0.452 / 0.58 / 0.55 | 0.738 / 0.60 / 0.66 | 0.272 / 0.35 / 0.28 | **0.274** / 0.32 / 0.27 |
| LR | 0.967 / 0.91 / 0.90 | **0.842 / 0.78 / 0.64** | 0.883 / 0.76 / 0.77 | 0.270 / 0.41 / 0.35 |
| LightGBM | **0.997 / 0.96 / 0.97** | 0.248 / 0.35 / 0.26 | **0.962 / 0.84 / 0.88** | 0.244 / 0.18 / 0.24 |
| MLP | 0.983 / 0.96 / 0.91 | 0.224 / 0.42 / 0.31 | 0.942 / 0.79 / 0.84 | 0.232 / 0.35 / 0.31 |

Shuffled-label check (case30 S1, LightGBM): PR-AUC **0.106** vs base rate 0.112 (the first run, before D17, gave 0.120). So no leakage detected.

**What this means:**
- **S1 (known outages) is easy.** Just memorising dangerous outages (`cont_prior`) gets ~0.8. LightGBM and MLP add load-level awareness and reach 0.94-0.997.
- **S2 (unseen outages) is hard for flat models,** especially on case118, where the best baseline reaches only 0.27 against a 0.10 base rate. *Why:* the outage is a one-hot column the model never saw switched on in training. Only the two descriptor numbers (tripped flow and loading) carry over to new outages.
- **One fold of case30 is not a verdict.** Logistic regression scores 0.842 on the case30 test fold but only 0.376 on its validation fold (a different set of 8 outages). With 7-8 outages per fold, each score mostly reflects *which* outages are held out. Only the 5-fold average on Step 6 can be trusted.
- **The outage descriptor helped flat models** (old S2 split, seed 0: LightGBM case30 went from 0.136 to 0.566 PR-AUC when D17 was added). Giving every model the same information was the right call.
- **The DC rule ignores voltage.** About two-thirds of case30's insecure samples have voltage violations, which DC cannot see. That is why it is weak on S1 (0.452).

### Step 5: GNNs, seed 0 only (container run, CPU, 22/10)

Same format and test sets as the Step 4 table (PR-AUC / recall / F1).

| Model | case30 S1 | case30 S2 | case118 S1 | case118 S2 |
|---|---|---|---|---|
| Base rate | 0.112 | 0.167 | 0.086 | 0.097 |
| Best baseline (from Step 4) | LightGBM 0.997 | LR 0.842 | LightGBM 0.962 | DC rule 0.274 |
| GCN | 0.989 / 0.97 / 0.94 | 0.246 / 0.27 / 0.26 | 0.947 / 0.80 / 0.85 | 0.402 / 0.43 / 0.45 |
| GraphSAGE | 0.989 / 0.94 / 0.95 | 0.179 / 0.10 / 0.09 | 0.950 / 0.75 / 0.85 | **0.468** / 0.18 / 0.26 |
| GATv2 | 0.994 / 0.98 / 0.95 | 0.160 / 0.00 / 0.00 | 0.942 / 0.79 / 0.86 | 0.406 / 0.25 / 0.30 |

Because single S2 folds are noisy, here is also the **validation** PR-AUC on S2 (outage fold 1, never used for training, but used for early stopping, so slightly optimistic for the neural models):

| Model | case30 S2 val (base 0.183) | case118 S2 val (base 0.074) |
|---|---|---|
| DC rule | 0.587 | 0.180 |
| LR | 0.376 | 0.205 |
| LightGBM | 0.430 | 0.236 |
| MLP | 0.325 | 0.168 |
| GCN | **0.896** | 0.542 |
| GraphSAGE | 0.685 | **0.584** |
| GATv2 | 0.822 | 0.577 |

**What this means:**
- **Known outages (S1): GNNs match the best flat model but do not beat it** (case30: 0.989-0.994 vs LightGBM 0.997; case118: 0.942-0.950 vs 0.962). When every outage has been seen in training, a graph is not needed.
- **Unseen outages on case118 (S2): the GNNs are the best models on both held-out folds.** On test they score 0.40-0.47 vs at most 0.27 for any baseline, and on validation 0.54-0.58 vs at most 0.24. This is the result the project was designed to test, but it is **one seed**. Step 6 decides.
- **Unseen outages on case30 flip between folds.** The GNNs are near chance on the test fold but the best models on the validation fold (GCN 0.896). Logistic regression does the opposite. With 7-8 outages per fold, case30 S2 is too small to judge from one fold.
- **Recall at the chosen threshold is low on S2** (0.10-0.43). PR-AUC says the *ranking* of risky cases is useful, but the threshold picked on validation outages does not transfer well to new outages. For screening, you would lower the threshold to raise recall and accept more false alarms.
- **Speed (indicative only):** GCN predicted the 1,710 case30 S1 test samples in 0.15 s on CPU, under 0.1 ms per sample. One pandapower AC power flow took about 43 ms here. That is not a fair engineering benchmark (batched neural network vs one unoptimised solver call), so we do not claim a speed-up figure.

### Step 6: all models, 5 seeds (container run, CPU, 22/10)

Full tables: `results/main_table.md`. Test PR-AUC, mean ± std over 5 seeds. For S2, each seed holds out a different fifth of the outages, so together the 5 seeds test every outage once.

| Model | case30 S1 | case30 S2 | case118 S1 | case118 S2 |
|---|---|---|---|---|
| Base rate (random) | 0.117 | 0.115 | 0.079 | 0.080 |
| cont_prior (memorise outages) | 0.794 ± 0.028 | 0.115 ± 0.041 | 0.812 ± 0.036 | 0.080 ± 0.014 |
| dc_rule | 0.445 ± 0.041 | **0.550 ± 0.179** | 0.291 ± 0.068 | 0.299 ± 0.101 |
| LR | 0.960 ± 0.012 | 0.334 ± 0.295 | 0.921 ± 0.029 | 0.233 ± 0.052 |
| LightGBM | **0.996 ± 0.002** | 0.437 ± 0.244 | **0.964 ± 0.015** | 0.237 ± 0.041 |
| MLP | 0.973 ± 0.009 | 0.208 ± 0.043 | 0.940 ± 0.011 | 0.196 ± 0.056 |
| GCN | 0.983 ± 0.007 | 0.452 ± 0.220 | 0.955 ± 0.015 | 0.367 ± 0.167 |
| GraphSAGE | 0.981 ± 0.007 | 0.463 ± 0.260 | 0.952 ± 0.018 | 0.390 ± 0.162 |
| GATv2 | 0.985 ± 0.006 | 0.369 ± 0.184 | 0.956 ± 0.014 | **0.404 ± 0.172** |

Fold by fold on **case118 S2**, each GNN beat the *best* baseline of that fold (picked with hindsight, which flatters the baselines) in **3 of 5 folds**. In the other two, the DC rule won.

**What this means (the honest headline):**
- **Known outages (S1): LightGBM is best on both grids.** The GNNs come close (0.98 vs 0.996; 0.955 vs 0.964) but never beat it in any of the 10 folds. A graph model is not needed when every outage was seen in training.
- **Unseen outages on the 118-bus grid (S2): GNNs have the highest average** (0.37-0.40 vs 0.30 for the best baseline, about 5× the 0.08 base rate). But the spread is large (± 0.17), and they win in only 3 of 5 folds. This is **promising evidence, not proof.**
- **Unseen outages on the 30-bus grid: no clear winner.** Every model swings wildly between folds (std 0.2-0.3). The DC rule has the best average. With only 7-8 outages per fold, this grid is too small to judge generalisation.
- **Recall on unseen outages is unreliable** (± 0.17-0.40). The threshold chosen on validation outages often does not suit new outages.
- The seed-0 first look overstated the GNN advantage on case118 S2 (0.40-0.47 vs 0.27). Running 5 seeds is exactly what caught that.

### Step 7: ablations and sanity checks (container run, CPU, 22/10)

All on **case118, unseen outages (S2)**, 5 seeds, using GAT (the GNN with the best *validation* score). Full table: `results/ablations.md`. Test PR-AUC, mean ± std (base rate 0.080).

| Variant | What was removed or changed | PR-AUC |
|---|---|---|
| Full GAT (Step 6) | nothing | 0.404 ± 0.172 |
| No topology | every grid connection removed: buses cannot exchange messages | 0.388 ± 0.120 |
| No branch features | GAT ignores impedance, rating and flow on lines | 0.431 ± 0.190 |
| 1 layer | information travels 1 bus instead of 3 | 0.412 ± 0.254 |
| 5 layers | information travels 5 buses | 0.402 ± 0.204 |
| **Shuffled labels**, GAT | trained on random labels | 0.131 ± 0.017 |
| **Shuffled labels**, LightGBM | trained on random labels | 0.088 ± 0.020 |

Fold by fold, the variants swing far more between folds (e.g. full GAT 0.21 to 0.68) than between each other. "No topology" beat full GAT in 2 of 5 folds.

**What this means (important, and honest):**
- **No ablation made a measurable difference.** Removing the grid connections entirely gave about the same score. So **the GNNs' lead on unseen case118 outages does *not* come from message passing over the grid topology.** The likely source is the architecture itself. It applies one shared small network to every bus, pools with mean + max, and reads the outage features (the tripped line's flow and loading on its end buses). That generalises to new outages better than the flat models' one-big-vector input. We did not prove this explanation; it is our best reading.
- **We can therefore NOT claim "the GNN learned the grid structure".** We can claim: "a bus-level shared-weight model beat flat baselines on unseen outages on average, but grid topology did not measurably contribute".
- **Leakage check passed.** Shuffled-label LightGBM sits at the base rate. Shuffled GAT is slightly above it (0.13). That is not leakage: a single input number with no learning at all, the tripped line's pre-outage loading, already scores 0.207 on the same folds. A network trained on noise can still produce scores that loosely follow such inputs. On known operating points, shuffled GAT scored 0.054 vs base 0.086 (1 seed).
- **A single input feature is a strong baseline on unseen outages:** tripped-line loading alone scores 0.207 ± 0.076, close to LR and LightGBM (0.23-0.24).

---

## 7. Limitations: what we can NOT claim

- **Not stability.** No dynamics, no transient or voltage-collapse simulation. Only steady-state limit checks.
- **Not real grids.** IEEE test cases are standard synthetic benchmarks.
- **case118 ratings are synthetic** (D3). The overload part of its label depends on that rule.
- **Single-branch outages only.** No generator outages and no N-2.
- **Non-convergence ≠ collapse.** The solver can fail for numerical reasons. In practice it never happened (0 of 32,640 samples), so our "insecure" label is entirely voltage and overload violations.
- **Small S2 test sets.** Unseen-outage results rest on 360 (case30) or 648 (case118) test samples per seed, with 21-63 insecure ones. Wide error bars are expected.
- **Load factor range is effectively 0.7-1.1.** Points with higher load were already insecure before any outage, so they were discarded (D7). The model has not seen heavier loading.
- **Thresholds do not transfer to unseen outages.** On S2, recall at the validation-tuned threshold was 0.00-0.43 even when PR-AUC was good. Any deployment would need its threshold set for the desired recall.
- **The S2 split was changed after looking at one test fold's contents** (D20). The change was model-agnostic and logged, but it is still a protocol change made after seeing early results.
- **The GNN advantage is not explained by topology.** Removing all grid connections did not measurably lower GAT's score on unseen outages (0.388 vs 0.404). We cannot claim the model learned grid structure.
- **Ablations used one GNN (GAT) and one split (case118 S2).**
- **No speed claim.** GNN inference was fast (under 0.1 ms per sample, batched, CPU), but we did not benchmark it against an optimised power-flow solver.

---

## 8. "If an interviewer asks..."

1. **Is this stability prediction?** No. It is static security assessment, meaning the steady state after an outage. Stability would need time-domain simulation of generator dynamics. I renamed the project to say that.
2. **Why IEEE test grids and not real data?** Real grid models are confidential. IEEE cases are the standard public benchmarks, so results are comparable and reproducible.
3. **Where do your line ratings come from?** case30 has real ratings in its file. case118's file has none (9900 MVA placeholders), so I set each rating to 1.1 × the worst flow the branch sees across all N-1 outages at nominal load. That makes the nominal grid N-1 secure, as a planned grid would be.
4. **Why drop islanding outages?** After islanding, part of the grid has no generator to balance it, so power flow is not defined in the same way. That is a different problem. I count them and exclude them.
5. **Why is non-convergence labelled insecure?** In practice an operator cannot trust that state. But it can also be a numerical failure, so I retry from 3 starting points and report the rate. It never happened on either grid (0 of 32,640 samples).
6. **How did you generate scenarios?** I scaled all loads by a random factor from 0.7 to 1.3 with per-bus noise, redispatched generators proportionally, kept only points that are secure before the outage, then ran every single-branch outage with AC power flow.
7. **What is your class balance?** 12.1% insecure on case30 (11,400 samples) and 9.1% on case118 (21,240 samples). So I report recall, F1 and PR-AUC on the insecure class, not accuracy.
8. **How do you know the pipeline is reproducible?** Seeds are fixed, versions are pinned, and a test runs generation twice with different numbers of parallel workers and checks that the label hash is identical.
9. **How did you prevent data leakage?**
   - Inputs contain only pre-outage quantities, and a test checks the input file's contents.
   - Splits are grouped by operating point, never random per sample, and S2 also holds out whole outages. Tests check that no group is shared.
   - Scaling statistics and decision thresholds come from training and validation data only.
   - A shuffled-label run must score at chance.
10. **Why not a random train/test split?** Each operating point appears with ~38-177 different outages. A random split would put the same operating point on both sides, and the model could recognise it instead of learning the physics.
11. **How do you represent the outage for the GNN?** I delete that branch from the graph. On its two end buses I add a flag, the branch's pre-outage loading, and an "equivalent injection": +P at one end and −P at the other, where P was its pre-outage flow. That is the compensation idea from power-system analysis, since tripping a branch acts like injecting its old flow into the remaining grid. So the model sees the new topology, where the change is, and how much power must reroute.
12. **Why PR-AUC and not accuracy or ROC-AUC?** Only about 10% of samples are insecure. Accuracy rewards saying "secure" every time. ROC-AUC can look high even when most alarms are false. PR-AUC focuses on the rare class, and its chance level is the base rate, which I always report next to it.
13. **One model scored 0.995-0.997. Wasn't that leakage?** I treated it as leakage until checked. Shuffled labels gave chance level (0.106-0.120 vs a 0.112 base rate), and feature importance pointed to physically sensible features. The task on known outages is just easy. That is why I also test on unseen outages.
14. **Why do classic models fail on unseen outages?** They see the outage as a one-hot column. If that column was never 1 in training, the model learned nothing about it. A graph model instead sees the outage as a change in a structure it knows.
15. **Why can a GNN handle an outage it never saw?** Weight sharing. The same message function runs at every bus and every branch. What it learns about how power reroutes around a missing branch in one part of the grid applies anywhere else. A flat model has separate weights for each input column, so nothing transfers.
16. **Did the GNN beat the baselines?** Partly. Over 5 seeds, on unseen outages in the 118-bus grid, the GNNs had the best average PR-AUC (0.37-0.40 vs 0.30 for the best baseline, base rate 0.08), but with high variance, and they won in only 3 of 5 outage folds. On known outages, LightGBM was best (0.964 vs 0.955 on case118). On the small 30-bus grid, no model was reliably better. My honest summary: GNNs help most where they should, on new outages in a bigger grid, but the evidence is moderate, not conclusive.
19. **Why run 5 seeds?** One run depends on luck: which samples and which outages land in the test set, and the random starting weights. With one seed, the GNN looked clearly better on unseen case118 outages (0.40-0.47 vs 0.27). Over 5 seeds the advantage shrank and the spread was large. Reporting mean ± std stops me from claiming a lucky result.
20. **What did your ablations show?** That nothing I removed mattered measurably, including the grid connections themselves. That was a surprise. It means the GNN's edge on unseen outages comes from its shared per-bus design and the outage features, not from passing messages along the grid. I report that instead of claiming the model "understands the grid".
21. **Your shuffled-label GNN scored above chance. Isn't that leakage?** LightGBM with shuffled labels sat at chance (0.088 vs 0.080). The GNN's 0.13 is below what one raw input, the tripped line's loading, scores with no learning at all (0.207). So the network's output loosely follows its inputs even when trained on noise. That is not label information leaking.
17. **Why did you change your test split halfway?** I looked at what one test fold contained and saw that only 1 of its 7 outages was ever insecure, so the score measured one outage. I switched to 5 rotating, balanced outage folds, so every outage is tested exactly once, and re-ran every model. I logged the change and the reason (decision D20). The change was the same for all models.
18. **Your recall on unseen outages is low. Is the model useless there?** The ranking is useful (PR-AUC is 3-5× the base rate on case118), but the threshold tuned on validation outages does not transfer well to new outages. In screening, you would pick a lower threshold and accept more false alarms, because missing an insecure case is the expensive error.

---

## 9. Resume bullet (draft, rewritten at the end from real results)

> *Draft after Step 7; finalised in Step 9.*
> **GNN-based N-1 Contingency Screening (Static Security Assessment)**: Built a pandapower pipeline generating 32k labelled (operating point, line-outage) samples on IEEE 30/118-bus grids (AC power flow, voltage and thermal limits). Benchmarked GCN/GraphSAGE/GAT against rule-based, DC power-flow and classic ML baselines (LR, LightGBM, MLP) with leakage-safe grouped splits over 5 seeds. GNNs led on unseen outages in the 118-bus grid (0.37-0.40 vs 0.30 PR-AUC, base rate 0.08) but with high variance; ablations showed grid topology did not drive the gain; LightGBM remained best on known outages (0.964).
