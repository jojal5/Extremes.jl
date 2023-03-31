@testset "structures.jl" begin
    include(joinpath("structures", "cluster_test.jl"))
    include(joinpath("structures", "dataitem_test.jl"))
    include(joinpath("structures", "AbstractExtremeValueModel_test.jl"))
    include(joinpath("structures", "AbstractFittedExtremeValueModel_test.jl"))
    include(joinpath("structures", "returnlevel_test.jl"))

end
