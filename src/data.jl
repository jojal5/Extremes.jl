"""
    dataset(name::String)::DataFrame

Load the dataset associated with `name`.

Some datasets used by Coles (2001) are available using the following names:
 - `portpirie`: annual maximum sea-levels in Port Pirie,
 - `glass`: breaking strengths of glass fibers
 - `fremantle`: annual maximum sea-levels in Fremantle
 - `rain`: daily rainfall accumulations in south-west England
 - `wooster`: daily minimum temperatures recorded in Wooster
 - `dowjones`: daily closing prices of the Dow Jones Index
 These datasets have been retrieved using the R package [*ismev*](https://cran.r-project.org/web/packages/ismev/index.html).

# Examples
```julia-repl
julia> Extremes.dataset("portpirie")
```
"""
function dataset(name::String)::DataFrame

    filename = joinpath(dirname(@__FILE__), "..", "data", string(name, ".csv"))
    if isfile(filename)
        return CSV.read(filename, DataFrame)
    end
    error("There is no dataset with the name '$name'")

end
