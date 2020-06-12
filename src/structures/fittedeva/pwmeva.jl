struct pwmEVA{T<:EVA} <: fittedEVA
    "Extreme value model definition"
    model::T
    "Maximum likelihood estimate"
    θ̂::Vector{Float64}
end

"""
    parametervar(fm::pwmEVA, nboot::Int=1000)

Estimate the parameter estimates covariance matrix by bootstrap.
"""
function parametervar(fm::pwmEVA, nboot::Int=1000)

    @assert nboot>0 "the number of bootstrap samples should be positive."

    y = fm.model.data
    n = length(y)

    θ̂ = Array{Float64}(undef, nboot, length(fm.θ̂))

    for i=1:nboot
        y = rand(y, n)            # Generate a bootstrap sample
        θ̂[i,:] = gevfitpwm(y).θ̂   # Compute the parameter estimates
    end

    V = cov(θ̂)                    # Compute the approximate covariance matrix

    return V

end


"""
    Base.show(io::IO, obj::pwmEVA)

Override of the show function for the objects of type pwmEVA.

"""
function Base.show(io::IO, obj::pwmEVA)

    println(io, "pwmEVA")
    println("model :")
    showEVA(io, obj.model, prefix = "\t")
    println()
    println(io, "θ̂  :\t", obj.θ̂)

end
