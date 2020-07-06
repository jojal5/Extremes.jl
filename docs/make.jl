using Documenter, Extremes, DataFrames, Cairo, Fontconfig

CI = get(ENV, "CI", nothing) == "true"

makedocs(sitename = "Extremes.jl",
    #doctest = false,
    format = Documenter.HTML(
    prettyurls = CI,
    ),
    pages = [
       "index.md",
       "Tutorial" =>["Getting started" => "tutorial/index.md",
            "Block maxima" => "tutorial/BlockMaxima.md",
            "Threshold exceedance" => "tutorial/ThresholdExceedance.md",
            "Declustering" => "tutorial/Declustering.md",
            "Non-stationary block maxima" => "tutorial/blockmaxima_ns.md",],
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
