using Documenter, Extremes, DataFrames

makedocs(sitename = "Extremes.jl",    
    pages = [
       "index.md",       
       ]

)

deploydocs(
    repo   = "github.com/jojal5/Extremes.jl.git",
    target = "build"    
)
