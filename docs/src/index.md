# Extremes for Julia documentation

## High level API

```@docs
getcluster(y::Array{<:Real,1}, u₁::Real , u₂::Real=0.0)
getcluster(df::DataFrame, u₁::Real, u₂::Real=0.0)
gevfitbayes(y::Array{<:Real}; warmup::Int=0, niter::Int=1000, thin::Int=1, stepSize::Array{<:Real,1}=[.1,.1,.05])
gevfit(y::Array{T,1} where T<:Real)
gevfit(y::Array{Float64,1}, location_covariate::Array{Float64,1}; initialvalues::Array{Float64,1}=Float64[])
gpdfit(y::Array{T} where T<:Real; threshold::Real=0.0)
gpdfitbayes(data::Array{Float64,1}; threshold::Real=0, niter::Int = 10000, warmup::Int = 5000,  thin::Int = 1, stepSize::Array{<:Real,1}=[.1,.1])
```

## Low level API

```@docs
Extremes.gumbelfitpwmom(x::Array{T,1} where T<:Real)
Extremes.gevfitlmom(x::Array{T,1} where T<:Real)
Extremes.getinitialvalues(y::Array{T,1} where T<:Real)
Extremes.gevhessian(y::Array{N,1} where N<:Real,μ::Real,σ::Real,ξ::Real)
Extremes.gpdfitmom(y::Array{T} where T<:Real; threshold::Real=0.0)
```
