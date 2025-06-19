# Release notes

## 1.0.5
- Changed the initial value computation of the extreme value model to ensure support in ℝ for the GEV and in ℝ⁺ for the GP, by returning the estimated parameters of the Gumbel and Exponential distributions, respectively, using the method of moments.
- Updated dependencies.

## 1.0.4
- Changed the initial values in the Fremantle test set in ReproducingColesResults.jl to comply with Optim.jl v1.12.
- Updated dependencies.

## 1.0.3
- Added lightweight functions for simple fit of Generalized Extreme Value, Gumbel and Generalized Pareto distributions. See [`Extremes.fit`](@ref). 

## 1.0.2
- Replication notebook reproducing the results and the figures of the JOSS paper.
- Refactor Hessian computations with PDMats.
- Implement generic Delta method using PDMats.

## 1.0.1
- Documentation updates to comply with the requirements of the JOSS paper.

## 1.0.0
- Inference for the Gumbel distribution
- Replace abstract type EVA by AbstractExtremeValueModel
- Replace abstract type fittedEVA by AbstractFittedExtremeValueModel
- Add confidence/credible interval in quantile-quantile and return level plots.

## 0.3.0
- Replace the dependency Mamba.jl by MambaLite.jl.
- Add the Flat distribution as a ContinuousUnivariateDistribution of Distributions.jl.

## Nightly