# Additional features

```@setup portpirie
using Extremes, DataFrames, Distributions, Gadfly, Optim
```

Additional features on fitted model by maximum likelihood that were not included in Coles are also provided. They are presented here for the Port Pirie example.

Loading the annual maximum sea-levels at Port Pirie:
```@example portpirie
data = Extremes.dataset("portpirie")
first(data,5)
```

Fitting the GEV parameters by maximum likelihood:
```@repl portpirie
fm = gevfit(data, :SeaLevel)
```

## Akaike information criterion (AIC)

The AIC of the fitted model can be obtained with the function [`aic`](@ref)
```@repl portpirie
aic(fm)
```

## Bayesian information criterion (BIC)


The BIC of the fitted model can be obtained with the function [`bic`](@ref)
```@repl portpirie
bic(fm)
```

## Profile likelihood


```@example portpirie

df = DataFrame( ξ = Float64[], pl = Float64[])

# Computing the profile loglikelihood for different values of ξ

for ξ in -.3:.01:.3

    fobj(θ::DenseVector) = -Extremes.loglike(fm.model, [θ[1], θ[2], ξ])

    res = optimize(fobj, [3., 0.])
    
    push!(df, [ξ, -Optim.minimum(res)])
    
end

set_default_plot_size(12cm, 8cm)

plot(df, x=:ξ, y=:pl, Geom.line, Guide.ylabel("Profile log likelihood"))
```


```@example portpirie

# 95% quantile of the Chi squared distribution with one degree of freedom
c = quantile(Chisq(1), .95)

# Maximum and maximum position of the profile log likelihood
fm = findmax(df.pl)

# High for the confidence interval
h = fm[1]-.5*c

# left bound
l = argmin(abs.(h.-df.pl[1:fm[2]]))

# right bound
r = fm[2] -1 + argmin(abs.(h.-df.pl[fm[2]:end]))

plot(df, x=:ξ, y=:pl, Geom.line,
    xintercept=[df.ξ[l], df.ξ[r]], Geom.vline(color="red", style=:dash),
    yintercept = [fm[1], h], Geom.hline(color="red", style=:dash),
    Guide.ylabel("Profile log likelihood"))
```


The corresponding 95% confidence interval for ξ based on the profile log likelihood is the following:
```@repl portpirie
println("[", df.ξ[l]," , " , df.ξ[r] , "]")
```

