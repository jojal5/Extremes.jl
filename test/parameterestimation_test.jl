@testset "parameterestimation.jl" begin
    include(joinpath("parameterestimation", "bayesian_test.jl"))
    include(joinpath("parameterestimation", "maximumlikelihood_test.jl"))
    include(joinpath("parameterestimation", "probabilityweightedmoment_test.jl"))
    include(joinpath("parameterestimation", "lightweight_functions_test.jl"))

end
