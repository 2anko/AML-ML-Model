---
name: AML project context
description: Key constraints and design decisions for the AML-ML-Model project
type: project
---

The model is a **risk ranker**, not a binary classifier. Output is a ranked list of customers by fraud probability score for investigator review. Binary predictions, thresholds, and confusion matrices are not the goal.

**Why:** The system is meant to prioritize which customers investigators review, not to auto-flag/block accounts.

**How to apply:** Optimize and evaluate using ranking metrics only: Recall@K (K=10,20,50), ROC-AUC, PR-AUC, and rank of each known fraud case. Do not optimize thresholds or report F1/accuracy.

Dataset is fixed at 1000 customers / 10 fraud cases — getting more labeled data is not possible. This is the only raw data available.

**How to apply:** Do not suggest acquiring more data. Focus on feature engineering, ranking model quality, and surfacing the ranked output clearly.
