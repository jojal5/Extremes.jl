
# Declustering threshold exceedances

When the data are dependent, the exceedances may have a tendency to cluster. It is common to assume that the marginal distribution of each exceedance is the Generalized Pareto (see for example Chapter 5 of Coles , 2001). Various methods have been proposed for dealing with the problem of dependent exceedances. The most widely-adopted one consists in declustering the exceedances, which corresponds to a filtering of the exceedances to obtain a set of that are approximately independent. As summarized by Coles (2001), the declustering approach consists in:
- identifying cluster of extreme values;
- retrieving the cluster maxima;
- assuming that each cluster maxima are independent;
- fitting the GP distribution to the cluster maxima.

In *Extremes.jl*, two methods for identifying cluster of extreme values are implemented in the [`getcluster`](@ref) function: the runs method and the two thresholds method. The function returns a vector of [`Cluster`](@ref) type.


## Data loading

 Both methods for identifying the clusters are illustrated with the Wooster dataset studied by Coles (2001). This dataset consists in the daily temperature minimum recorded in Wooster, Ohio, from 1983 to 1988. The attention is restricted to the winter months (November to February) and to the negated series for using the model defined for maxima.

```@setup wooster
using Extremes, Dates, DataFrames, Distributions, Gadfly
```

```@example wooster
data = Extremes.dataset("wooster")
set_default_plot_size(12cm, 8cm) # hide
plot(data, x=:Date, y=:Temperature, Geom.point)
```

```@example wooster
df = copy(data)
df[!,:Temperature] = -data[:,:Temperature]
filter!(row -> month(row.Date) ∈ (1,2,11,12), df)
plot(df, x=:Date, y=:Temperature, Geom.point)
```


## The runs method

Two cluster definitions are used in *Extremes.jl*. The first one is based on the *runs method*:
exceedances separated by fewer than ``r`` non-exceedances are assumed to belong to the same cluster (see Chapter 10 of Beirlant *et al.*, 2004). The parameter ``r`` is generally referred to as the *runlength* parameter. When ``r = 0``, each exceedance forms a separate cluster.

When using the threshold of -10°F and the runlength of 4, 17 clusters are idenfied:

```@example wooster
threshold =  -10.0
cluster = getcluster(df[:,:Temperature], threshold, runlength=4)
nothing # hide
```

The GP distribution can be fitted on the cluster maximal exceedances (the cluster maximum can be computed with the `maximum` method applied to the Cluster type):
```@repl wooster
z = maximum.(cluster) .- threshold
```


## The two thresholds method

With the two thresholds method, a cluster of threshold exceedances is defined as a streak of data higher than a second threshold ``u₂`` whose at least one data is higher than a first threshold ``u₁``, where ``u₁ ≥ u₂``.

When using the ``u₁ = 0°F`` as the first threshold and ``u₂ = -10°F`` as the second threshold, 11 clusters are idenfied:

```@example wooster
u₁ = 0.0
u₂ = -10.0
cluster = getcluster(df[:,:Temperature], u₁, u₂)
nothing # hide
```
The GP distribution can be fitted on the cluster maximal exceedances:
```@repl wooster
z = maximum.(cluster) .- u₁
```
