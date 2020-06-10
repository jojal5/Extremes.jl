@testset "fittedeva.jl" begin
    include(joinpath("fittedeva", "bayesianeva_test.jl"))
    include(joinpath("fittedeva", "maximumlikelihoodeva_test.jl"))
    include(joinpath("fittedeva", "pwmeva_test.jl"))

end
