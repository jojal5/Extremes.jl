struct GeneralizedMaximumLikelihoodEVA{T} <: fittedEVA{T}
    "Extreme value model definition"
    model::T
    "Generalized Maximum likelihood estimate"
    θ̂::Vector{Float64}
end


# TO-DO: Voir si on peut réutiliser les fonctions définies pour MaximumLikelihoodEVA

"""
    hessian(model::GeneralizedMaximumLikelihoodEVA)::Array{Float64, 2}
Calculates the Hessian matrix associated with the GeneralizedMaximumLikelihoodEVA model.
"""
function hessian(model::GeneralizedMaximumLikelihoodEVA)::Array{Float64, 2}
 
    fobj(θ) = -loglike(model.model, θ)
    return ForwardDiff.hessian(fobj, model.θ̂)

end

"""
    parametervar(fm::GeneralizedMaximumLikelihoodEVA)::Array{Float64, 2}
Compute the covariance parameters estimate of the fitted model `fm`.
"""
function parametervar(fm::GeneralizedMaximumLikelihoodEVA)::Array{Float64, 2}

    # Compute the parameters covariance matrix
    V = inv(hessian(fm))

    return V
end

"""
    loglike(fd::GeneralizedMaximumLikelihoodEVA)::Real
Compute the model loglikelihood evaluated at θ̂ if the maximum likelihood method has been used.
"""
function loglike(fm::GeneralizedMaximumLikelihoodEVA)::Real

    ll = loglike(fm.model, fm.θ̂)

    return ll

end

"""
    getdistribution(fittedmodel::GeneralizedMaximumLikelihoodEVA)::Vector{<:Distribution}
Return the fitted distribution in case of stationarity or the vector of fitted distribution in case of non-stationarity.
"""
function getdistribution(fittedmodel::GeneralizedMaximumLikelihoodEVA)::Vector{<:Distribution}

    model = fittedmodel.model
    θ̂ = fittedmodel.θ̂

    res = getdistribution(model, θ̂)

    return res

end

"""
    quantile(fm::GeneralizedMaximumLikelihoodEVA, p::Real)::Vector{<:Real}
Compute the quantile of level `p` from the fitted model by Generalized maximum likelihood. In the case of non-stationarity, the effective quantiles are returned.
"""
function quantile(fm::GeneralizedMaximumLikelihoodEVA, p::Real)::Vector{<:Real}

    @assert zero(p)<p<one(p) "the quantile level should be between 0 and 1."

    q = quantile(fm.model, fm.θ̂, p)

    return q

end

"""
    quantilevar(fd::GeneralizedMaximumLikelihoodEVA, level::Real)::Vector{<:Real}
Compute the variance of the quantile of level `level` from the fitted model `fm`.
"""
function quantilevar(fm::GeneralizedMaximumLikelihoodEVA, level::Real)::Vector{<:Real}

    θ̂ = fm.θ̂
    H = hessian(fm)

    q = quantile(fm, level)

    V = zeros(length(q))

    for i=1:length(q)

        f(θ::DenseVector) = quantile(fm.model,θ,level)[i]
        Δf(θ::DenseVector) = ForwardDiff.gradient(f, θ)
        G = Δf(θ̂)

        V[i] = G'/H*G

    end

    return V

end

"""
    returnlevel(fm::GeneralizedMaximumLikelihoodEVA{BlockMaxima}, returnPeriod::Real)::ReturnLevel
Compute the return level corresponding to the return period `returnPeriod` from the fitted model `fm`.
"""
function returnlevel(fm::GeneralizedMaximumLikelihoodEVA{BlockMaxima}, returnPeriod::Real)::ReturnLevel

      @assert returnPeriod > zero(returnPeriod) "the return period should be positive."

      # quantile level
      p = 1-1/returnPeriod

      return ReturnLevel(BlockMaximaModel(fm), returnPeriod, quantile(fm, p))

end


function cint(rl::ReturnLevel{GeneralizedMaximumLikelihoodEVA{BlockMaxima}}, confidencelevel::Real=.95)::Vector{Vector{Real}}

      @assert rl.returnperiod > zero(rl.returnperiod) "the return period should be positive."
      @assert zero(confidencelevel)<confidencelevel<one(confidencelevel) "the confidence level should be in (0,1)."

      # quantile level
      p = 1-1/rl.returnperiod

      q = quantile(rl.model.fm, p)

      # Compute the credible interval

      α = (1 - confidencelevel)

      v = quantilevar(rl.model.fm,p)

      qdist = Normal.(q,sqrt.(v))

      a = quantile.(qdist,α/2)
      b = quantile.(qdist,1-α/2)

      return slicematrix(hcat(a,b), dims=2)

end


"""
    showfittedEVA(io::IO, obj::GeneralizedMaximumLikelihoodEVA; prefix::String = "")
Displays a MaximumLikelihoodEVA with the prefix `prefix` before every line.
"""
function showfittedEVA(io::IO, obj::GeneralizedMaximumLikelihoodEVA; prefix::String = "")

    println(io, prefix, "GeneralizedMaximumLikelihoodEVA")
    println(io, prefix, "model :")
    showEVA(io, obj.model, prefix = prefix*"\t")
    println(io)
    println(io, prefix, "θ̂  :\t", obj.θ̂)

end

"""
    transform(fm::GeneralizedMaximumLikelihoodEVA{BlockMaxima})::GeneralizedMaximumLikelihoodEVA
Transform the fitted model for the original covariate scales.
"""
function transform(fm::GeneralizedMaximumLikelihoodEVA{BlockMaxima})::GeneralizedMaximumLikelihoodEVA

    locationcovstd = fm.model.location.covariate
    logscalecovstd = fm.model.logscale.covariate
    shapecovstd = fm.model.shape.covariate

    locationcov = Extremes.reconstruct.(locationcovstd)
    logscalecov = Extremes.reconstruct.(logscalecovstd)
    shapecov = Extremes.reconstruct.(shapecovstd)

    # Model on the original covariate scale
    model = BlockMaxima(fm.model.data, locationcov = locationcov, logscalecov = logscalecov, shapecov = shapecov)

    # Transformation of the parameter estimates
    θ̂ = deepcopy(fm.θ̂)
    ind = Extremes.paramindex(fm.model)

    for (var, par) in zip([locationcovstd, logscalecovstd, shapecovstd],[:μ, :ϕ, :ξ])
        if !isempty(var)
            a = getfield.(var, :scale)
            b = getfield.(var, :offset)

            # θ̂[ind[par][1]] = fm.θ̂[ind[par][1]] - sum( fm.θ̂[ind[par][1+i]] * b[i]/a[i] for i=1:length(a) )

            for i=1:length(a)
                θ̂[ind[par][1]] -= fm.θ̂[ind[par][1+i]]*b[i]/a[i]
                θ̂[ind[par][1+i]] = fm.θ̂[ind[par][1+i]]/a[i]
            end
        end
    end

    # Contruction of the fittedEVA structure
    return GeneralizedMaximumLikelihoodEVA(model, θ̂)

end

function cint(fm::GeneralizedMaximumLikelihoodEVA, clevel::Real=.95)::Array{Array{Float64,1},1}

    @assert 0<clevel<1 "the confidence level should be between 0 and 1."

    α = 1 - clevel

    V = parametervar(fm)

    confint = Vector{Vector{Float64}}()

    q = quantile.(Normal(0,1),[α/2, 1 - α/2])

    for i in eachindex(fm.θ̂)
        push!(confint, fm.θ̂[i] .+ q*sqrt(V[i,i]))
    end

    return confint

end
