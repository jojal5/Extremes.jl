"""
    gevfitbayes(y::Array{<:Real}; warmup::Int=0, niter::Int=1000, thin::Int=1, stepSize::Array{<:Real,1}=[.1,.1,.05])

Fits a GEV...
"""
function gevfitbayes(y::Array{<:Real}; warmup::Int=0, niter::Int=1000, thin::Int=1, stepSize::Array{<:Real,1}=[.1,.1,.05])

    @assert niter>warmup "The total number of iterations should be larger than the number of warmup iterations."

    μ = Array{Float64}(undef, niter)
    ϕ = Array{Float64}(undef, niter)
    ξ = Array{Float64}(undef, niter)

    fd = gevfit(y)
    μ[1] = Distributions.location(fd)
    ϕ[1] = log(Distributions.scale(fd))
    ξ[1] = Distributions.shape(fd)

    lu = log.(rand(Uniform(),niter,3))

    δ = randn(niter,3) .* repeat(stepSize',niter)

    acc = falses(niter,3)

    for i=2:niter

        μ₀ = μ[i-1]
        ϕ₀ = ϕ[i-1]
        ξ₀ = ξ[i-1]

        μ̃ = μ₀ + δ[i,1]

        pd_cand = GeneralizedExtremeValue(μ̃, exp(ϕ₀), ξ₀)
        pd_present = GeneralizedExtremeValue(μ₀, exp(ϕ₀), ξ₀)

        lr = loglikelihood(pd_cand, y) - loglikelihood(pd_present, y)

        if lr > lu[i,1]
            μ₀ = μ̃
            acc[i,1] = true
        end

        ϕ̃ = ϕ₀ + δ[i,2]

        pd_cand = GeneralizedExtremeValue(μ₀, exp(ϕ̃), ξ₀)
        pd_present = GeneralizedExtremeValue(μ₀, exp(ϕ₀), ξ₀)

        lr = loglikelihood(pd_cand, y) - loglikelihood(pd_present, y)

        if lr > lu[i,2]
            ϕ₀ = ϕ̃
            acc[i,2] = true
        end

        ξ̃ = ξ₀ + δ[i,3]

        pd_cand = GeneralizedExtremeValue(μ₀, exp(ϕ₀), ξ̃)
        pd_present = GeneralizedExtremeValue(μ₀, exp(ϕ₀), ξ₀)

        lr = loglikelihood(pd_cand, y) - loglikelihood(pd_present, y)

        if lr > lu[i,3]
            ξ₀ = ξ̃
            acc[i,3] = true
        end

        μ[i] = μ₀
        ϕ[i] = ϕ₀
        ξ[i] = ξ₀

    end

    acc = acc[warmup+1:niter,:]

    accRate = [count(acc[:,i])/size(acc,1) for i=1:size(acc,2)]

    itr = (warmup+1):thin:niter

    μ = μ[itr]
    σ = exp.(ϕ[itr])
    ξ = ξ[itr]

    if any(accRate.<.4) || any(accRate.>.7)
        @warn "Acceptation rates are $accRate for μ, ϕ and ξ respectively. Consider changing the stepsizes to obtain acceptation rates between 0.4 and 0.7."
    end

    fd = GeneralizedExtremeValue.(μ, σ, ξ)

    return fd

end


"""
    gpdfitbayes(data::Array{Float64,1}; threshold::Real=0, niter::Int = 10000, warmup::Int = 5000,  thin::Int = 1, stepSize::Array{<:Real,1}=[.1,.1])

Fits a Generalized Pareto Distribution (GPD)
"""
function gpdfitbayes(data::Array{Float64,1}; threshold::Real=0, niter::Int = 10000, warmup::Int = 5000,  thin::Int = 1, stepSize::Array{<:Real,1}=[.1,.1])

    @assert niter>warmup "The total number of iterations should be larger than the number of warmup iterations."

    if isapprox(threshold,0)
        y = data
    else
        y = data .- threshold
    end

    fd = Extremes.gpdfit(y)
    σ₀ = Distributions.scale(fd)
    ξ₀ = Distributions.shape(fd)

    ϕ = Array{Float64}(undef,niter)
    ξ = Array{Float64}(undef,niter)

    ϕ[1] = log(σ₀)
    ξ[1] = ξ₀

    acc = falses(niter,2)

    lu = log.(rand(Uniform(),niter,2))
    δ₁ = rand(Normal(),niter)*stepSize[1]
    δ₂ = rand(Normal(),niter)*stepSize[2]

    for i=2:niter


        ϕ̃ = ϕ[i-1] + δ₁[i]

        f = GeneralizedPareto( exp(ϕ[i-1]) , ξ[i-1] )
        f̃ = GeneralizedPareto( exp(ϕ̃) , ξ[i-1] )

        lr = loglikelihood(f̃,y) - loglikelihood(f,y)

        if lr > lu[i,1]
            ϕ[i] = ϕ̃
            acc[i,1] = true
        else
            ϕ[i] = ϕ[i-1]
        end


        ξ̃ = ξ[i-1] + δ₂[i]

        f = GeneralizedPareto( exp(ϕ[i]) , ξ[i-1] )
        f̃ = GeneralizedPareto( exp(ϕ[i]) , ξ̃ )

        lr = loglikelihood(f̃,y) - loglikelihood(f,y)

        if lr > lu[i,2]
            ξ[i] = ξ̃
            acc[i,2] = true
        else
            ξ[i] = ξ[i-1]
        end


    end

    acc = acc[warmup+1:niter,:]

    accRate = [count(acc[:,i])/size(acc,1) for i=1:size(acc,2)]

    println("Acceptation rate for ϕ is $(accRate[1])")
    println("Acceptation rate for ξ is $(accRate[2])")


    if any(accRate .< .4) | any(accRate .> .7)
        @warn "Acceptation rates are $accRate for ϕ and ξ respectively. Consider changing the stepsizes to obtain acceptation rates between 0.4 and 0.7."
    end


    itr = (warmup+1):thin:niter

    σ = exp.(ϕ[itr])
    ξ = ξ[itr]

    fd = GeneralizedPareto.(threshold, σ, ξ)

    return fd

end
