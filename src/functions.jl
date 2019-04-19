
function gumbelfitpwmom(x::Array{T,1} where T<:Real)

    n = length(x)
    y = sort(x)
    r = 1:n

    # Probability weighted moments
    b₀ = mean(y)
    b₁ = 1/n/(n-1)*sum( y[i]*(n-i) for i=1:n)

    # Gumbel parameters estimations
    σ̂ = (b₀ - 2*b₁)/log(2)
    μ̂ = b₀ - Base.MathConstants.eulergamma*σ̂

    pdfit = Gumbel(μ̂,σ̂)

    return pdfit
end


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



function gevfitlmom(x::Array{T,1} where T<:Real)

    n = length(x)
    y = sort(x)
    r = 1:n

    #     L-Moments estimations (Cunnane, 1989)
    b₀ = mean(y)
    b₁ = sum( (r .- 1).*y )/n/(n-1)
    b₂ = sum( (r .- 1).*(r .- 2).*y ) /n/(n-1)/(n-2)

    # GEV parameters estimations
    c = (2b₁ - b₀)/(3b₂ - b₀) - log(2)/log(3)
    k = 7.859c + 2.9554c^2
    σ̂ = k *( 2b₁-b₀ ) /(1-2^(-k))/gamma(1+k)
    μ̂ = b₀ - σ̂/k*( 1-gamma(1+k) )

    ξ̂ = -k

    pdfit = GeneralizedExtremeValue(μ̂,σ̂,ξ̂)

    return pdfit
end

function getinitialvalues(y::Array{T,1} where T<:Real)

    pd = gevfitlmom(y)

    # check if initial values are in the domain of the GEV
    valid_initialvalues = all(insupport(pd,y))

    if valid_initialvalues
        μ₀ = location(pd)
        σ₀ = scale(pd)
        ξ₀ = Distributions.shape(pd)
    else
        pd = gumbelfitpwmom(y)
        μ₀ = location(pd)
        σ₀ = scale(pd)
        ξ₀ = 0.0
    end

    initialvalues = [μ₀, σ₀, ξ₀]

    return initialvalues

end

# function gevfit(y::Array{N,1} where N; method="ml", initialvalues=Float64[], location_covariate=Float64[], logscale_covariate =Float64[])
"""
gevfit(y::Array{N,1} where N; method="ml", initialvalues::Array{Float64}, location_covariate::Array{Float64})

Fit GEV parameters over data y.
"""

function gevfit(y::Array{T,1} where T<:Real)

    initialvalues = getinitialvalues(y)

    fobj(μ, ϕ, ξ) = loglikelihood(GeneralizedExtremeValue(μ,exp(ϕ),ξ),y)
    mle = Model(with_optimizer(Ipopt.Optimizer, print_level=0,sb="yes"))
    JuMP.register(mle,:fobj,3,fobj,autodiff=true)
    @variable(mle, μ, start = initialvalues[1])
    @variable(mle, ϕ, start = log(initialvalues[2]))
    @variable(mle, ξ, start = initialvalues[3])
    @NLobjective(mle, Max, fobj(μ, ϕ, ξ) )
    JuMP.optimize!(mle)

    μ̂ = JuMP.value(μ)
    σ̂ = exp(JuMP.value(ϕ))
    ξ̂ = JuMP.value(ξ)

    if termination_status(mle) == MOI.OPTIMAL
        # error("The algorithm did not find a solution.")
        @warn "The algorithm did not find a solution. Maybe try with different initial values or with another method."
    end

    fd = GeneralizedExtremeValue(μ̂,σ̂,ξ̂)

    return fd

end

function gevfit(y::Array{Float64,1}, location_covariate::Array{Float64,1}; initialvalues::Array{Float64,1}=Float64[])

    if !isempty(initialvalues)
        pd = GeneralizedExtremeValue.(initialvalues[1] .+ initialvalues[2]*location_covariate,initialvalues[3],initialvalues[4])
        if any(.!insupport.(pd,y))
            error("Invalid initial values.")
        end
    else
        initialvalues = getinitialvalues(y)
        splice!(initialvalues,2:1,0)
    end


    if isapprox(mean(location_covariate),0,atol=eps())
        b = 0
    else
        b = mean(location_covariate)
    end

    if isapprox(mean(location_covariate),1,atol=eps())
        a = 1
    else
        a = std(location_covariate)
    end

    x = (location_covariate .- b)./a



    fobj(μ₀, μ₁, ϕ, ξ) = sum(logpdf.(GeneralizedExtremeValue.(μ₀ .+ μ₁*x, exp(ϕ), ξ), y))
    mle = Model(with_optimizer(Ipopt.Optimizer, print_level=2,sb="yes"))
    JuMP.register(mle,:fobj,4,fobj,autodiff=true)
    @variable(mle, μ₀, start=  initialvalues[1])
    @variable(mle, μ₁, start = initialvalues[2])
    @variable(mle, ϕ, start = log(initialvalues[3]))
    @variable(mle, ξ, start= initialvalues[4])
    @NLobjective(mle, Max, fobj(μ₀, μ₁, ϕ, ξ) )
    JuMP.optimize!(mle)

    μ̃₀ = JuMP.value(μ₀)
    μ̃₁ = JuMP.value(μ₁)
    μ̂₀ = μ̃₀ - b/a*μ̃₁
    μ̂₁ = μ̃₁/a

    σ̂ = exp(JuMP.value(ϕ))
    ξ̂ = JuMP.value(ξ)

    if termination_status(mle) == MOI.OPTIMAL
        @warn "The algorithm did not find a solution. Maybe try with different initial values."
    end

    μ̂ = μ̂₀ .+ μ̂₁*location_covariate

    fd = GeneralizedExtremeValue.(μ̂,σ̂,ξ̂)

    return fd

    #
    # θ = [μ̂₀ μ̂₁ exp(JuMP.value(ϕ)) JuMP.value(ξ)]
    # logl = getobjectivevalue(mle)
    # bic = -2*logl + 4*log(length(y))



end



function gevhessian(y::Array{N,1} where N<:Real,μ::Real,σ::Real,ξ::Real)

    #= Estimate the hessian matrix evaluated at (μ, σ, ξ) for the iid gev random sample y =#

    logl(θ) = sum(gevloglike.(y,θ[1],θ[2],θ[3]))

    H = ForwardDiff.hessian(logl, [μ σ ξ])

end


function gpdfitmom(y::Array{T} where T<:Real; threshold::Real=0.0)

    if isapprox(threshold,0)
        ȳ = mean(y)
        s² = var(y)
    else
        ȳ = mean(y .- threshold)
        s² = var(y .- threshold)
    end

    ξ̂ = 1/2*(1-ȳ^2/s²)
    σ̂ = (1-ξ̂)*ȳ

    return GeneralizedPareto(threshold,σ̂,ξ̂)

end

function gpdfit(y::Array{T} where T<:Real; threshold::Real=0.0)

    # get initial values
    fd = gpdfitmom(y::Array{Float64}, threshold=threshold)
    if all(insupport(fd,y))
        σ₀ = scale(fd)
        ξ₀ = Distributions.shape(fd)
    else
        σ₀ = mean(y .- threshold)
        ξ₀ = 0.0
    end


    # fobj(ϕ, ξ) = sum(logpdf.(GeneralizedPareto(0,exp(ϕ),ξ),y))
    fobj(ϕ, ξ) = loglikelihood(GeneralizedPareto(threshold,exp(ϕ),ξ),y)
    mle = Model(with_optimizer(Ipopt.Optimizer, print_level=0,sb="yes"))
    JuMP.register(mle,:fobj,2,fobj,autodiff=true)
    @variable(mle, ϕ, start = log(σ₀))
    @variable(mle, ξ, start = ξ₀)
    @NLobjective(mle, Max, fobj(ϕ, ξ) )
    JuMP.optimize!(mle)

    σ̂ = exp(JuMP.value(ϕ))
    ξ̂ = JuMP.value(ξ)

        if termination_status(mle) == MOI.OPTIMAL
            # error("The algorithm did not find a solution.")
            @warn "The algorithm did not find a solution. Maybe try with different initial values."
        end

    fd = GeneralizedPareto(threshold,σ̂,ξ̂)
    
end
