using Documenter, Extremes, Cairo, Fontconfig

CI = get(ENV, "CI", nothing) == "true"

makedocs(sitename = "Extremes.jl",
    #doctest = false,
    format = Documenter.HTML(
    prettyurls = CI,
    ),
    pages = [
       "index.md",
       "Tutorial" =>["Getting started" => "tutorial/index.md",
            "Block maxima" => "tutorial/blockmaxima.md",
            "Threshold exceedance" => "tutorial/thresholdexceedance.md",
            "Declustering" => "tutorial/declustering.md",
            "Non-stationary block maxima" => "tutorial/blockmaxima_ns.md",
            "Non-stationary threshold exceedance" => "tutorial/thresholdexceedance_ns.md"],
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
