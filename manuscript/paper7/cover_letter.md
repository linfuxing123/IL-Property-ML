# Cover Letter

[Date]

Dear Editor,

We are pleased to submit our manuscript, **"Predictor disagreement as a
property-aware diagnostic for distributional bias in materials machine
learning"**, for consideration as a Research Article in [Journal].

## What this paper does

Multi-model agreement is the de facto standard for assessing prediction
reliability in materials machine learning — yet its statistical structure is
rarely examined. Using 1.68 million ionic-liquid (IL) cation–anion pairs
scored by four predictor families (two gradient-boosted trees, a graph neural
network, and an independent ML model) across four properties, we establish a
set of results with direct methodological implications:

1. **A variance-decomposition identity** (Cov(m,d) = (Var₁−Var₂)/4 under equal
   variances, verified symbolically and numerically) that separates structural
   from heteroscedastic effects in predictor disagreement.

2. **Property-dependent disagreement–performance structure**: conductivity
   *saturates* (disagreement shrinks at high predicted values, r ≈ −0.3 to
   −0.5 across all paradigm pairs), while density, melting point, and
   viscosity *anti-saturate* (r = +0.15 to +0.27). We reject the
   physical-boundedness explanation and find partial support for a
   training-density-coupling mechanism (Welch t = −7.14, p = 1.1e−12; effect
   concentrated in the high-value regime).

3. **A validated diagnostic**: across all four properties, predictor
   disagreement is positively correlated with prediction error (r = +0.15 to
   +0.49; error grows 29–143% across disagreement quartiles) — disagreement is
   a **label-free diagnostic of prediction unreliability**. Conversely,
   disagreement-penalized ranking provides no gain: its value is diagnostic,
   not prescriptive.

We conclude that disagreement-based diagnostics must be property-aware and
that ensemble consensus must span paradigms (tree + GNN + chemical language
model) to provide independent evidence. These results give immediate,
quantitative guidance for cross-validation practice and active-learning design
in materials informatics.

## Novelty and fit

To our knowledge, this is the first systematic, large-scale (1.68M pairs)
study of the *statistical structure* of predictor disagreement in materials
ML, and the first to show its property dependence and label-free diagnostic
value. The variance-decomposition framework is general and transferable beyond
ionic liquids.

## Relation to prior submissions

This methods study builds on two of our submissions under review (refs [1,2])
but is a distinct, standalone contribution: it does not overlap with their
content beyond the shared IL data backbone.

## Availability

All data, code, and models are publicly available at
https://github.com/linfuxing123/IL-Property-ML (version DOI to be assigned on
release). No conflicts of interest.

Sincerely,

Fuxing Lin
Hunan Institute of Engineering
ORCID: 0000-0003-7588-6942
Email: 3612411485@qq.com
