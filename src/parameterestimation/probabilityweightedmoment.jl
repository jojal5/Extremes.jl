"""
 pwm(x::Vector{<:Real},p::Int,r::Int,s::Int)::Real

Compute the empirical probability weighted moments defined by:
```math
M_{p,r,s} = \\mathbb{E}\\left[ X^p F^r(X) \\left\\{ 1-F(X) \\right\\}^s  \\right].
```
The unbiased empirical estimate is computed using the formula given by Landwehr et al. (1979).

*Reference:*
Landwehr, J. M., Matalas, N. C. and Wallis, J. R. (1979). Probability weighted moments compared with
 some traditional techniques in estimating Gumbel Parameters and quantiles. Water Resources Research,
 15(5), 1055–1064.

"""
function pwm(x::Vector{<:Real},p::Int,r::Int,s::Int)::Real

 @assert sign(p)>=0 "p should be a non-negative integer."
 @assert sign(r)>=0 "r should be a non-negative integer."
 @assert sign(s)>=0 "s should be a non-negative integer."

 y = sort(x)
 n = length(y)

 m = 1/n*sum( y[i]^p * binomial(i-1,r)/binomial(n-1,r) * binomial(n-i,s)/binomial(n-1,s) for i=1:n )

 return m

end

include(joinpath("probabilityweightedmoment", "probabilityweightedmoment_gev.jl"))
include(joinpath("probabilityweightedmoment", "probabilityweightedmoment_gp.jl"))
include(joinpath("probabilityweightedmoment", "probabilityweightedmoment_gumbel.jl"))
