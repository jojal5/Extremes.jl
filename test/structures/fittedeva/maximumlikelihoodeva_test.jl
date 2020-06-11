@testset "maximumlikelihoodeva.jl" begin
    @testset "hessian(model)" begin
        # TODO : Test with known values

    end

    @testset "parametervar(fm)" begin
        # TODO : Test with known values

    end

    @testset "loglike(fd)" begin
        # TODO : Test with known values

    end

    @testset "getdistribution(fittedmodel)" begin
        # TODO : Test with known values

    end

    @testset "quantile(fm, p)" begin
        # TODO : Test with p < 0 or p > 1

        # TODO : Test with known values

    end

    @testset "quantilevar(fm, level)" begin
        # TODO : Test witj known values

    end

    @testset "returnlevel(fm, returnPeriod, confidencelevel)" begin
        # TODO : Test with returnPeriod < 0

        # TODO : Test with confidencelevel < 0 or confidencelevel > 1

        # TODO : Test with known values

    end

    @testset "returnlevel(fm, threshold, nobservation, nobsperblock, returnPeriod, confidencelevel)" begin
        # TODO : Test when implemented
    end

    @testset "Base.show(io, obj)" begin
        # TODO : Test outputs correctly
        
    end

end
