# Cover Letter

[Date]

Dear Editor,

We are pleased to submit our manuscript, **"From 860 to 219,292 cations: how data
scale unlocks generative design of ionic liquids"**, for consideration as a
Research Article in ACS Central Science.

## What this paper does

Generative design of ionic liquids (ILs) has been widely proposed but consistently
fails at two separate stages: generating *chemically valid* new ions, and
predicting ions that *outperform* the known experimental optimum. Earlier work
(our JCIM submission, ref. [1]) exposed both failures on a small dataset (860
cations / 356 anions) but could not separate their causes. Here we separate them
experimentally by scaling the data 255-fold on the ion axis:

1. **Chemical validity is a data-scale problem — and it is now solved.** A SMILES
   autoregressive language model trained on 150k cations achieves 99.9% RDKit
   validity and 99.9% charge correctness, versus ~60% for the character-level VAE
   used previously. We show the VAE's failure is architectural (posterior
   collapse), not data-limited: scaling the VAE data 100-fold actually *degrades*
   it. Scaling the *model class* solves it.

2. **Surpassing the known optimum is not a data-scale problem.** Two independent
   8.3-million-IL predictor families (an ML model suite and a chemical language
   model), plus our own three predictors (GBM/HistGBM/MPNN), agree that no IL in
   the virtual space is predicted to beat the experimental best (EMIM-DCA,
   ln κ ≈ 0.64). Cross-predictor correlation on conductivity is only r = 0.20.
   We report this honestly rather than claiming a fragile "breakthrough".

3. **A constructive deliverable.** Screening 1.68 million novel cation–anion
   pairs with our retrained, IL-disjoint-validated predictors yields 7 unique,
   chemically valid, room-temperature-liquid candidates with three-predictor
   agreement (e.g. 1-vinyl-3-methylimidazolium DCA), ready for experimental or
   DFT/MD follow-up.

The paper contributes (i) a clear decomposition of the two bottlenecks of
generative materials design, (ii) a reproducible pipeline from 8.3M-scale virtual
libraries through triple-source cross-validation to a short candidate list, and
(iii) a candid statement of the distribution-bounded limit of predictors — a
finding we believe the community needs more than another inflated claim.

## Relation to prior submissions

This work builds on our two submissions currently under review: ref. [1] (JCIM,
inverse design under IL-disjoint validation, small data) and ref. [2] (ACS
Sustainable Chemistry & Engineering, GNN vs descriptors). The present manuscript
is a distinct, standalone methods study: it does not overlap with their content
beyond the shared experimental backbone and the honest-validation protocol.

## Availability

All data, code, and models are publicly available at
https://github.com/linfuxing123/IL-Property-ML (version DOI to be assigned on
release). No conflicts of interest.

Sincerely,

Fuxing Lin
Hunan Institute of Engineering
ORCID: 0000-0003-7588-6942
Email: 3612411485@qq.com
