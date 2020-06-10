@testset "eva.jl" begin
    @testset "computeparamfunction(covariates)" begin

    end

    @testset "loglike(model, θ)" begin

    end

    @testset "quantile(model, θ, p)" begin

    end

    @testset "validatestationarity(model)" begin

    end

    @testset "Base.show(io, obj)" begin

    end

    @testset "showparamfun(name, param)" begin

    end

    include(joinpath("eva", "blockmaxima_test.jl"))
    include(joinpath("eva", "thresholdexceedance_test.jl"))

end
