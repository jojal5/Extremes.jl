# Release notes

## 0.3.0
- Replace the dependency Mamba.jl by MambaLite.jl.
- Add the Flat distribution as a ContinuousUnivariateDistribution of Distributions.jl.

## 1.0.0
- Inference for the Gumbel distribution
- Replace abstract type EVA by AbstractExtremeValueModel
- Replace abstract type fittedEVA by AbstractFittedExtremeValueModel
- Add confidence/credible interval in quantile-quantile and return level plots.

## 1.0.1
- Documentation updates to comply with the requirements of the JOSS paper.

## 1.0.2
- Replication notebook reproducing the results and the figures of the JOSS paper.
- Refactor Hessian computations with PDMats.
- Implement generic Delta method using PDMats.

## Nightly