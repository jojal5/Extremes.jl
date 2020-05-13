"""
    gevfit(y::DenseVector)

Fit the Generalized Extreme Value (GEV) distribution by maximum likelihood to the vector of data `y.
"""
function gevfit(y::DenseVector)

    data = Dict(:y => y)
    dataid = :y
    Covariate = Dict(:μ => Symbol[], :ϕ => Symbol[], :ξ => Symbol[])
    paramindex = paramindexing(Covariate)
    nparameters = getparameternumber(Covariate)

    model = EVA(GeneralizedExtremeValue, "ML", data, dataid, Covariate, nparameters, identity, identity, identity, paramindex, Float64[])

    gevfit!(model)

    return model

end

"""
    gevfit(data::Dict, dataid::Symbol)

Fit the Generalized Extreme Value (GEV) distribution by maximum likelihood to the vector of data contains in the dictionary `data`under the key `dataid`.

- `gevfit(data, dataid=:y)`: Fits the GEV distribution to the data stored in the dictionary `data`under the key `dataid`.

Data is a dictionary with Symbol as keys.
"""
function gevfit(data::Dict, dataid::Symbol)

    Covariate = Dict(:μ => Symbol[], :ϕ => Symbol[], :ξ => Symbol[])
    paramindex = paramindexing(Covariate)
    nparameters = getparameternumber(Covariate)

    model = EVA(GeneralizedExtremeValue, "ML", data, dataid, Covariate, nparameters, identity, identity, identity, paramindex, Float64[])

    gevfit!(model)

    return model

end

"""
    gevfit(data::Dict, dataid::Symbol, Covariate::Dict)

Fit a non-stationary Generalized Extreme Value (GEV) distribution by maximum likelihood to the vector of data contains in the disctionary `data`under the key `dataid`.

Covariate is a dictionary containing the covariates identifyer for each parameter (μ, ϕ, ξ).

The location parameter μ is a linear function using the covariates in `data` identified by the symbols in Covariate[:μ].
The logscale parameter ϕ is a linear function using the covariates in `data` identified by the symbols in Covariate[:ϕ].
The location parameter ξ is a linear function using the covariates in `data` identified by the symbols in Covariate[:ξ].

Example with a non-stationary location parameter:
```julia
# Sample size
n = 300

# Covariate
x = collect(1:n)

# Location as function of the covariate
μ = x*1/100

# Sample from the non-stationary GEV distribution
pd = GeneralizedExtremeValue.(μ,1,.1)
y = rand.(pd)

# Put the data in a dictionary
data = Dict(:y => y, :x => x, :n => n)

# Put the covariate identifier in a dictionary
Covariate = Dict(:μ => [:x], :ϕ => Symbol[], :ξ => Symbol[] )

# Estimate the parameters
gevfit(data, :y, Covariate=Covariate)
```

The covariate may be standardized to facilitate the estimation.

"""
function gevfit(data::Dict, dataid::Symbol ; Covariate::Dict)

    # Put empty Symbol array to stationary parameters
    for k in [:μ, :ϕ, :ξ]
        if !(haskey(Covariate,k))
            Covariate[k] = Symbol[]
        end
    end

    paramindex = paramindexing(Covariate)
    nparameters = getparameternumber(Covariate)

    locationfun = computeparamfunction(data, Covariate[:μ])
    logscalefun = computeparamfunction(data, Covariate[:ϕ])
    shapefun = computeparamfunction(data, Covariate[:ξ])

    model = EVA(GeneralizedExtremeValue, "ML", data, dataid, Covariate, nparameters,
        locationfun, logscalefun, shapefun, paramindex, Float64[])

    gevfit!(model)

    return model

end

"""
    gevfit(model::EVA)

Fit the non-stationary Generalized Extreme Value (GEV) distribution by maximum likelihood of the EVA model `model`.

"""
function gevfit!(model::EVA)

    initialvalues = getinitialvalue(model)

    fobj(θ) = -loglike(model, θ)

    res = optimize(fobj, initialvalues)

    if Optim.converged(res)
        model.results = Optim.minimizer(res)
    else
        @warn "The maximum likelihood algorithm did not find a solution. Maybe try with different initial values or with another method. The returned values are the initial values."
        model.results = initialvalues
    end

    return model

end


# function gevfit(data::Dict; dataid=nothing, locationid=nothing, logscaleid=nothing,
#     initialvalues::Vector{<:Real}=Float64[])
#
#     # GEVfit with a non-stationary location and logscale parameter
#
#     @assert haskey(data,dataid) "Invalid data key provided."
#
#     if !isnothing(locationid)
#         @assert haskey(data,locationid) "Invalid location key provided."
#     end
#
#     if !isnothing(logscaleid)
#         @assert haskey(data,logscaleid) "Invalid logscale key provided."
#     end
#
#
#     y = data[dataid]
#
#     if isempty(initialvalues)
#
#         if !isnothing(locationid)
#
#             b = mean(data[locationid])
#             a = std(data[locationid])
#
#             if isapprox(b,0,atol=eps())
#                 b = 0
#             end
#
#             if isapprox(a,1,atol=eps())
#                 a = 1
#             end
#
#             location_covariate = (data[locationid] .- b)/a
#         end
#
#
#         if !isnothing(logscaleid)
#             d = mean(data[logscaleid])
#             c = std(data[logscaleid])
#
#             if isapprox(d,0,atol=eps())
#                 d = 0
#             end
#
#             if isapprox(c,1,atol=eps())
#                 c = 1
#             end
#
#             logscale_covariate = (data[logscaleid] .- d)/c
#         end
#
#         μ₀, σ₀, ξ₀ = [getinitialvalue(GeneralizedExtremeValue(),y)...]
#
#         initialvalues = [μ₀, 0.0, log(σ₀), 0.0, ξ₀]
#
#     else
#         if isnothing(locationid)
#             insert!(initialvalues, 2, 0)
#         else
#             location_covariate = data[locationid]
#         end
#
#         if isnothing(logscaleid)
#             insert!(initialvalues, 4, 0)
#         else
#             logscale_covariate = data[logscaleid]
#         end
#
#     end
#
#
#     function locfunction(locationid)
#            fun = if isnothing(locationid)
#                function(β₀::Real, β₁::Real)
#                    β₀
#                end
#            else
#                function(β₀::Real, β₁::Real)
#                    β₀ .+ β₁ * location_covariate
#                end
#            end
#            return fun
#        end
#
#     μ = locfunction(locationid)
#
#
#     function logscalefunction(logscaleid)
#            fun = if isnothing(logscaleid)
#                function(ϕ₀::Real, ϕ₁::Real)
#                    exp(ϕ₀)
#                end
#            else
#                function(ϕ₀::Real, ϕ₁::Real)
#                    exp.(ϕ₀ .+ ϕ₁ * logscale_covariate)
#                end
#            end
#            return fun
#        end
#
#     μ = locfunction(locationid)
#     σ = logscalefunction(logscaleid)
#
# #     logl(β₀,β₁,ϕ₀,ϕ₁,ξ) = sum( logpdf.(GeneralizedExtremeValue.(μ(β₀,β₁), σ(ϕ₀,ϕ₁), ξ), y) )
#
#     function logl(β₀::Real,β₁::Real,ϕ₀::Real,ϕ₁::Real,ξ::Real)
#
#         if all( σ(ϕ₀,ϕ₁) .> 0)
#             pd = GeneralizedExtremeValue.(μ(β₀,β₁), σ(ϕ₀,ϕ₁), ξ)
#
#             if all(insupport.(pd,y))
#                 ll = sum( logpdf.(pd, y))
#             else
#                 ll = -Inf
#             end
#         else
#             ll = -Inf
#         end
#
#         return ll
#
#     end
#
#
# #     fobj = TwiceDifferentiable(θ -> -logl(θ...), initialvalues, autodiff=:forward)
#     fobj(θ) = -logl(θ...)
#
#     res = optimize(fobj, initialvalues)
#
#     if Optim.converged(res)
#         β̂₀, β̂₁, ϕ̂₀, ϕ̂₁, ξ̂ = [Optim.minimizer(res)...]
#     else
#         @warn "The maximum likelihood algorithm did not find a solution. Maybe try with different initial values or with another method. The returned values are the initial values."
#         β̂₀, β̂₁, ϕ̂₀, ϕ̂₁, ξ̂  = [initialvalues...]
#     end
#
#     fd = GeneralizedExtremeValue.(μ(β̂₀,β̂₁), σ(ϕ̂₀,ϕ̂₁), ξ̂)
#
#     return fd
#
# end
#
# """
#     gevfit(y::Vector{<:Real},initialvalues::Vector{<:Real}=Float64[])
#
# Fit the Generalized Extreme Value (GEV) distribution to data using maximum likelihood. Return an object of Type GeneralizedExtremeValue.
#
# - `gevfit(y)`:                Fits the GEV distribution to the data `y`.
# - `gevfit(y, initialvalues=initialvalues)`: Fits the GEV distribution to the data `y` using `initialvalues` as initial values.
#
# """
# function gevfit(y::Vector{<:Real};initialvalues::Vector{<:Real}=Float64[])
#     data_dict = Dict(:y => y)
#     fd = gevfit(data_dict, dataid=:y)
#     return fd
# end
#
#
# """
#     gpdfit(data::Dict; dataid=nothing, logscaleid=nothing,
#     initialvalues::Vector{<:Real}=Float64[], threshold::Real=0)
#
# Fit the Generalized Pareto (GEP) distribution to data using maximum likelihood. The log-scale parameter can vary as function of covariates.
# Return an object of type GeneralizedPareto or an array of type GeneralizedPareto of length equals to the data length when the log-scale parameter is varying.
#
# - `gpdfit(y)`: Fits the GP distribution to the data `y`.
# - `gpdfit(y, initialvalues=initialvalues)`: Fits the GP distribution to the data `y` using `initialvalues` as initial values.
# - `gpdfit(data, dataid=:y)`: Fits the GP distribution to the data stored in the dictionary `data`under the key `dataid`.
# - `gpdfit(data, dataid=:y, logscaleid=:x)`: Fits the GP distribution to the data stored in the dictionary `data`under the key `dataid` with the log-scale parameter varying as a linear function of the covariate under the key `logscaleid`.
# - `gpdfit(data, dataid=:y,...)`
#
# """
# function gpdfit(data::Dict; dataid=nothing, logscaleid=nothing,
#     initialvalues::Vector{<:Real}=Float64[], threshold::Real=0)
#
#     @assert haskey(data,dataid) "Invalid data key provided."
#
#     if !isnothing(logscaleid)
#         @assert haskey(data,logscaleid) "Invalid logscale key provided."
#     end
#
#
#     if threshold == 0
#         y = data[dataid]
#     else
#         y = data[dataid] .- threshold
#     end
#
#
#     if isempty(initialvalues)
#
#         if !isnothing(logscaleid)
#
#             d = mean(data[logscaleid])
#             c = std(data[logscaleid])
#
#             if isapprox(d,0,atol=eps())
#                 d = 0
#             end
#
#             if isapprox(c,1,atol=eps())
#                 c = 1
#             end
#
#             logscale_covariate = (data[logscaleid] .- d)/c
#         end
#
#        σ₀, ξ₀ = [getinitialvalue(GeneralizedPareto(),y)...]
#
#         initialvalues = [log(σ₀), 0.0, ξ₀]
#
#     else
#
#         if isnothing(logscaleid)
#             insert!(initialvalues, 2, 0)
#         else
#             logscale_covariate = data[logscaleid]
#         end
#
#     end
#
#
#     function logscalefunction(logscaleid)
#            fun = if isnothing(logscaleid)
#                function(ϕ₀::Real, ϕ₁::Real)
#                    exp(ϕ₀)
#                end
#            else
#                function(ϕ₀::Real, ϕ₁::Real)
#                    exp.(ϕ₀ .+ ϕ₁ * logscale_covariate)
#                end
#            end
#            return fun
#        end
#
#
#
#     σ = logscalefunction(logscaleid)
#
# #     logl(ϕ₀,ϕ₁,ξ) = sum( logpdf.(GeneralizedPareto.(0, σ(ϕ₀,ϕ₁), ξ), y) )
#
#     function logl(ϕ₀::Real,ϕ₁::Real,ξ::Real)
#
#         if all( σ(ϕ₀,ϕ₁) .> 0)
#             pd = GeneralizedPareto.(0, σ(ϕ₀,ϕ₁), ξ)
#
#             if all(insupport.(pd,y))
#                 ll = sum( logpdf.(pd, y))
#             else
#                 ll = -Inf
#             end
#         else
#             ll = -Inf
#         end
#
#         return ll
#
#     end
#
# #     fobj = TwiceDifferentiable(θ -> -logl(θ...), initialvalues, autodiff=:forward)
#     fobj(θ) = -logl(θ...)
#
#     res = optimize(fobj, initialvalues)
#
#     if Optim.converged(res)
#         ϕ̂₀, ϕ̂₁, ξ̂ = [Optim.minimizer(res)...]
#     else
#         @warn "The maximum likelihood algorithm did not find a solution. Maybe try with different initial values or with another method. The returned values are the initial values."
#         ϕ̂₀, ϕ̂₁, ξ̂  = [initialvalues...]
#     end
#
#     fd = GeneralizedPareto.(threshold, σ(ϕ̂₀,ϕ̂₁), ξ̂)
#
#     return fd
#
# end
#
# """
#     gpdfit(y::Vector{<:Real},initialvalues::Vector{<:Real}=Float64[], threshold::Real=0)
#
# Fit the Generalized Pareto (GP) distribution to data using maximum likelihood.
#
# - `gpdfit(y)`: Fits the GP distribution to the data `y`.
# - `gpdfit(y, threshold= threshold)`: Fits the GP distribution to the data `y` with the threshold `threshold`.
# - `gpdfit(y, initialvalues=initialvalues)`: Fits the GP distribution to the data `y` using `initialvalues` as initial values.
# - `gpdfit(y, ...)`
# """
# function gpdfit(y::Vector{<:Real}; initialvalues::Vector{<:Real}=Float64[], threshold::Real=0)
#     data = Dict(:y => y)
#     fd = gpdfit(data, dataid=:y, initialvalues=initialvalues, threshold=threshold)
#     return fd
# end
