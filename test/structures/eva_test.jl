@testset "eva.jl" begin
    @testset "computeparamfunction(covariates)" begin
        # TODO : Test with empty covariates

        # TODO : Test with covariates not empty

    end

    @testset "loglike(model, θ)" begin
        # TODO : Test with known values

    end

    @testset "quantile(model, θ, p)" begin
        # TODO : Test with p < 0 or p > 1

        # TODO : Test with known values

    end

    @testset "validatestationarity(model)" begin
        # TODO : Test with non-stationary model

        # TODO : Test with stationary model

    end

    @testset "Base.show(io, obj)" begin
        # TODO : Test outputs correctly

    end

    @testset "showparamfun(name, param)" begin
        # TODO : Test with known values
    end

    include(joinpath("eva", "blockmaxima_test.jl"))
    include(joinpath("eva", "thresholdexceedance_test.jl"))

end
