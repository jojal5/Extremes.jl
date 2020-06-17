using DataFrames, Dates, SpecialFunctions
using Distributions, Extremes
using Test
using LinearAlgebra, Random
using Mamba
using Statistics

# Set the seed for reproductible test results
Random.seed!(12)

@testset "Extremes.jl" begin
    include("data_test.jl")
    # include("parameterestimation_test.jl")
    include("reproducingColesResults.jl")
    include("structures_test.jl")
    include("utils_test.jl")
end;
