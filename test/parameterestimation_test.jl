@testset "parameterestimation.jl" begin
    include(joinpath("parameterestimation", "bayesian_test.jl"))
    include(joinpath("parameterestimation", "maximumlikelihood_test.jl"))
    include(joinpath("parameterestimation", "probabilityweightedmoment_test.jl"))

end
