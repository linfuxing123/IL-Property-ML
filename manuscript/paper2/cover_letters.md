# Cover letter — primary (Digital Discovery) & variant (I&EC Research)

> 2026-08-18 · 按"law → tool → benchmark"三句话卖点组织
> 用哪个版本取决于用户拍板的目标刊；投出前替换期刊名/读者定位句。

---

## Digital Discovery 版（首选，定律+基准类最对口）

Dear Editors,

We are pleased to submit our manuscript, **"Scaling laws of ionic-liquid property
prediction: data-density laws, leakage taxes, and optimal measurement
allocation,"** for consideration as a full Research Article in *Digital Discovery*.

Machine learning for ionic-liquid (IL) properties is widely assumed to be
limited by data scarcity—yet the field lacks quantitative answers to the
questions that matter for design: *how* much data, for *which* property, and
*where* to measure next. This manuscript answers all three.

1. **A law.** From 86,008 curated NIST ILThermo records (1,891 ILs), we
   establish the first IL-specific learning-curve laws R²(N) = a − b·N^(−γ)
   under strict IL-disjoint validation, with property-specific exponents
   (γ = 0.215 viscosity, 0.807 conductivity, 0.565 density, 0.288 melting
   point). The laws classify each property as data-limited, representation-
   limited, or diversity-limited—turning "more data helps" into a decision
   rule for where measurement, feature, and library-expansion budgets belong.
2. **A tool.** We quantify the "leakage tax" of random splits (up to
   ΔR² = +0.55 in redundancy-rich strata) and show that coverage-guided data
   acquisition beats random measurement by ΔR² = +0.33 at 250 measurements
   for viscosity. Cold-start decomposition locates the deficit in the anion
   dimension. We release a prioritized top-100 list of novel ILs to measure
   next, drawn from 8.3 M virtual ion pairs.
3. **A benchmark.** Dataset, split files, a multi-model leaderboard on
   identical folds, and one-command reproducibility are released openly
   (GitHub release v2.0, Zenodo DOI) so the community can audit and extend
   every number.

The work fits *Digital Discovery*'s scope as a data-centric methodological
contribution with an actionable experimental deliverable. We believe it will
be of immediate use to both ML practitioners and experimental IL researchers.

All data and code are openly available; there are no conflicts of interest.

Sincerely,
Fuxing Lin
Hunan Institute of Engineering · 3612411485@qq.com · ORCID 0009-0003-7588-6942

---

## I&EC Research 版（二投变体：测量预算优化/过程设计角度）

Dear Editors,

We are pleased to resubmit a substantially revised manuscript, **"Scaling laws
of ionic-liquid property prediction: data-density laws, leakage taxes, and
optimal measurement allocation,"** for consideration in *Industrial &
Engineering Chemistry Research*.

Ionic liquids are central to chemical-process engineering—as solvents,
electrolytes, and separation media—and engineering design requires
thermophysical property data across an enormous combinatorial space. This
manuscript addresses the practical question every process group faces:
**how should a limited experimental budget be spent on IL property
measurements?**

Using 86,008 curated records from the NIST ILThermo repository (1,891 ILs),
we establish quantitative learning-curve laws that tell the process engineer
(a) which properties are worth measuring more (viscosity: ≈5,700 ILs to reach
R² ≈ 0.80), (b) which are representation-limited and will not improve with
data alone (conductivity saturates at R² ≈ 0.73), and (c) which ion families
to prioritize (the anion dimension dominates the viscosity deficit).
Coverage-guided acquisition outperforms random measurement by ΔR² = +0.33 at
250 measurements, and we release a prioritized list of 100 ILs whose
measurement would most improve extrapolation, together with a reproducible
benchmark and dataset.

We respectfully note that this version is a substantive reframing of our
earlier submission (ie-2026-04274g): the core contribution has moved from a
validation study to a quantitative design tool with an actionable measurement
deliverable, which we believe matches the applied, decision-oriented
readership of I&EC Research.

All data and code are openly available; no conflicts of interest.

Sincerely,
Fuxing Lin
Hunan Institute of Engineering · 3612411485@qq.com · ORCID 0009-0003-7588-6942
