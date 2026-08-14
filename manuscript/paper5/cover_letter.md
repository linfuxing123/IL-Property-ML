[Date]

Dear Editor,

I am pleased to submit the manuscript "From property prediction to inverse design: multi-property Pareto screening of ionic-liquid electrolytes under IL-disjoint validation" for consideration in the *Journal of Chemical Information and Modeling*.

The paper addresses a methodological gap in ionic-liquid (IL) chemical informatics: forward prediction of IL properties is well developed, but the inverse problem—selecting an ion pair for a target property profile—remains under-served and is typically validated with random splits that overstate generalization. We present a workflow that (i) uses predictors validated under IL-disjoint (group-level) cross-validation, (ii) cross-checks every proposed candidate with two further, algorithmically distinct models (histogram gradient boosting and a message-passing graph neural network), and (iii) applies multi-objective Pareto screening to the 238,500 cation–anion combinations formed from 795 known cations and 300 known anions, of which only 1,891 have been experimentally reported. The workflow proposes 11 three-model-consistent, unreported, room-temperature-liquid candidates, led by small heterocyclic cations (thiazolium/imidazolium) paired with dicyanamide, with predicted conductivity up to ln κ = 0.64.

The manuscript also reports, rather than hides, the limits of de novo generation at the available data scale: a character-level VAE trained on ~1,000 ions generates largely chemically unreasonable structures, and latent-space optimization regresses to the known optimum. We argue that data density—not model architecture—is the binding constraint on generative inverse design, a conclusion directly relevant to the JCIM community working on generative molecular design under data scarcity.

This work is original, has not been published elsewhere, and is not under consideration by another journal. All data, predictors, and code are openly available at https://github.com/linfuxing123/IL-Property-ML and archived on Zenodo (https://doi.org/10.5281/zenodo.21898948).

I am the sole and corresponding author. Thank you for your consideration.

Sincerely,
Fuxing Lin
Hunan Institute of Engineering
3612411485@qq.com · ORCID 0009-0003-7588-6942
