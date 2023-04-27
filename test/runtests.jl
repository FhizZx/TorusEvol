# Test style inspired by the Distributions.jl package
using TorusEvol
using Test
using Random

const TEST_SEED = 123456
printstyled("Running tests:\n", color=:blue)

const all_tests = ["distributions/WrappedNormal", "models/TKF92", "distributions/EvolHMM"]

tests = ["distributions/EvolHMM"]

@testset "TorusEvol" begin
    @testset "Test $t" for t in tests
        include("$t.jl")
    end
end
