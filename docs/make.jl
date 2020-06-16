using Documenter, Extremes, DataFrames

CI = get(ENV, "CI", nothing) == "true"

makedocs(sitename = "Extremes.jl",
    #doctest = false,
    format = Documenter.HTML(
    prettyurls = CI,
    ),
    pages = [
       "index.md",
       "contributing.md",
       #"functions.md",
       ]

)

if CI
    deploydocs(
    repo   = "github.com/jojal5/Extremes.jl.git",
    target = "build"
    )
end
