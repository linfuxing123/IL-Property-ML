# Cover Letter

**Manuscript title:** Data density as the binding constraint: a 7.7-fold expansion of ionic-liquid property data lifts group-disjoint prediction from cold start to transferable accuracy

**Corresponding author:** Fuxing Lin, Hunan Institute of Engineering,
ORCID: 0009-0003-7588-6942, 3612411485@qq.com

---

Dear Editors,

I am pleased to submit the companion manuscript, *Data density as the binding constraint: a 7.7-fold expansion of ionic-liquid property data lifts group-disjoint prediction from cold start to transferable accuracy*, for consideration as a Research Article in *Science*.

This study is the direct experimental test of the central hypothesis of my companion paper (currently under review at *Science*, preprint on ChemRxiv): that data density—not model capacity—is the binding constraint on extrapolation to unseen ionic liquids. My previous work established the honest evaluation framework and showed that viscosity and melting point, with roughly one record per ionic liquid (IL), are unpredictable under IL-disjoint cross-validation (R² ≈ −0.09), while conductivity and density, with denser coverage, are partially predictable. The unavoidable question that followed was whether the cold-start failures could be cured by data. This manuscript answers that question with a controlled, large-scale experiment.

I assembled the largest openly available multi-property IL dataset compiled to date: 88,077 experimental records spanning 1,891 unique ion pairs, harvested systematically from the NIST ILThermo v2.0 repository with standardized units and verified SMILES—a 7.7-fold expansion of the companion dataset. Re-running the identical model, features, and strict IL-disjoint 5-fold evaluation, I find that viscosity prediction rises from R² = −0.09 to 0.68, electrical conductivity from 0.55 to 0.70, density from 0.83 to 0.85, and melting point from ≈0 to 0.39 with 642 ILs. Three results should interest the broad *Science* readership. First, the causal role of data density is now demonstrated rather than inferred: no architectural change produced these gains, only coverage. Second, the leakage penalty of point-wise validation persists at scale (inflation ΔR² = 0.09–0.15), a caution that larger datasets do not by themselves discipline evaluation. Third, the data-curation process uncovered a unit inconsistency (S/m versus mS/cm) in previously published conductivity compilations—an invisible source of model degradation that the community should treat as a standard audit step.

The companion relationship is deliberate: the two manuscripts form a complete arc from methodology and negative result (Paper 1) to validation and positive transferability result (this paper), sharing one open codebase, one honest-evaluation toolkit, and one continuously versioned dataset (Zenodo DOI 10.5281/zenodo.21898949). All data, scripts, per-fold results, and figures are publicly available.

This manuscript is original, has not been published previously, and is not under consideration elsewhere. The author declares no competing interests.

Thank you for your consideration.

Sincerely,

Fuxing Lin
Hunan Institute of Engineering
3612411485@qq.com · ORCID 0009-0003-7588-6942
