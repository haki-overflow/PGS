# Publication plan (after the exam)

## Verdict
As of 22/10 the project is a solid, honest student project but **not publishable in an IEEE
journal as it stands**. Realistic path: an IEEE conference paper first (e.g. PES General
Meeting, ISGT, PowerTech, SmartGridComm; check current deadlines), then a journal only if the
extended work produces a clearly new result. A literature review must come first: GNNs for
power flow and contingency analysis is a crowded area, and novelty is not yet established.

## Critique
1. **Weak main claim.** GNNs on unseen case118 outages: 0.37-0.40 ± 0.17 PR-AUC vs 0.30 for the
   best baseline, winning 3 of 5 folds. No significance test.
2. **The ablation undercuts the title.** Removing all grid connections did not change GAT's
   score, so the topology is not what helps.
3. **Small systems.** 30 and 118 buses are toy-sized; case30 S2 is noise (7-8 outages per fold).
   Expected: case300 plus 1,000+ buses (`case1354pegase`, `case2869pegase` in pandapower).
4. **Partly artificial labels.** case118 ratings are synthetic (1.1 × worst N-1 flow), with no
   sensitivity analysis. Operating points come from uniform load scaling and proportional
   dispatch, not OPF or realistic load profiles.
5. **Evaluation does not match the use case.** Screening is judged by missed critical
   contingencies at a given filtering ratio, not PR-AUC alone. Recall at the chosen threshold
   is unstable. There is no speed benchmark, even though speed is the motivation.
6. **Missing expected baselines.** LODF/PTDF screening, classical performance-index ranking
   (PI_MW, PI_V), prior GNN methods. No hyperparameter search with an equal budget.
7. **Narrow scope.** Branch outages only (no generator outages, no N-2); binary label only
   (no margin regression).
8. **Thin statistics.** 5 folds, mean ± std. Needs repeated grouped CV, paired tests, CIs.
9. **Authorship and AI use.** IEEE requires disclosure of AI-generated content. The author must
   have done substantial parts personally and be able to defend every choice in review.

**Strengths to keep:** leakage-safe grouped splits, logged protocol changes, 5 seeds,
shuffled-label checks, reproducible pipeline with fixed data hashes, an interesting negative
ablation result.

## Plan (~3-4 months part time)
| Phase | Time | Work |
|---|---|---|
| A. Contribution | 2 wks | Literature review. Recommended story: "Do GNNs need the graph for N-1 screening? A rigorous benchmark" (builds on the ablation). Alternative: a physics-informed model where topology demonstrably helps. |
| B. Realistic data | 3-4 wks | OPF dispatch, realistic load profiles, case300 + 1,000+ bus systems (GPU), generator outages, rating-margin sensitivity, margin targets. |
| C. Baselines | 2-3 wks | LODF screening and PI ranking (implemented by the author; exam material), explicit per-bus no-graph model, equal tuning budget for all models. |
| D. Evaluation | 2 wks | Screening curves (missed critical contingencies vs fraction filtered), paired tests over 10+ folds, timing vs a fast solver. |
| E. Writing | 3-4 wks | Conference paper with AI-use disclosure; internal review by a professor (ideally a faculty co-author). |

The exam (computer-aided power system analysis: load flow, sensitivity factors, contingency
analysis) covers exactly what Phases C and D need. Study it properly first.
