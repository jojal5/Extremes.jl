using Documenter, Extremes, DataFrames, Cairo, Fontconfig

CI = get(ENV, "CI", nothing) == "true"

makedocs(sitename = "Extremes.jl",
    #doctest = false,
    format = Documenter.HTML(
    prettyurls = CI,
    ),
    pages = [
       "index.md",
       "getting_started.md",
       "contributing.md",
       "functions.md"
       ]

)

if CI
    deploydocs(
    repo   = "github.com/jojal5/Extremes.jl.git",
    devbranch = "dev",
    push_preview = true,
    target = "build"
    )
end
