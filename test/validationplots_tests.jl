@testset "validationplots.jl" begin
    @testset "ecdf(y)" begin
        n = 10
        data = -1 * collect(1:n)

        y, p = Extremes.ecdf(data)

        # y is length n and sorted
        @test length(y) == n
        @test issorted(y)

        # p is length n, sorted and between 0 and 1.
        @test length(p) == n
        @test issorted(p)
        @test p[1] >= 0 && p[end] <= 1

    end

    @testset "checkstationarity(m)" begin
        y = Variable("y", collect(1:10))

        sfm = MaximumLikelihoodEVA(BlockMaxima(y), [1.0, 1.0 ,0.0])
        nsfm = MaximumLikelihoodEVA(BlockMaxima(y, logscalecov = [y]), [1.0, 1.0, 1.0 ,0.0])

        # Model stationary and shouldbestationary no info
        @test_logs Extremes.checkstationarity(sfm.model)

        # Model non-stationary and shouldbestationary info
        @test_logs (:info, "The graph is optimized for stationary models and the model provided is not.") Extremes.checkstationarity(nsfm.model)

    end

    @testset "checknonstationarity(m)" begin
        y = Variable("y", collect(1:10))

        sfm = MaximumLikelihoodEVA(BlockMaxima(y), [1.0, 1.0 ,0.0])
        nsfm = MaximumLikelihoodEVA(BlockMaxima(y, logscalecov = [y]), [1.0, 1.0, 1.0 ,0.0])

        # Model stationary and !shouldbestationary info
        @test_logs (:info, "The graph is optimized for non-stationary models and the model provided is not.") Extremes.checknonstationarity(sfm.model)

        # Model non-stationary and !shouldbestationary no info
        @test_logs Extremes.checknonstationarity(nsfm.model)

    end

    include(joinpath("validationplots", "plots_std_tests.jl"))
    include(joinpath("validationplots", "plots_tests.jl"))

end
