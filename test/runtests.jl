# Test style inspired by the Distributions.jl package
using TorusEvol
using Test
using Random

include("distributions/torus_distributions.jl")
include("distributions/processes.jl")

const TEST_SEED = 123456
printstyled("Running tests:\n", color=:blue)

const all_tests = ["distributions/WrappedNormal", "models/TKF92", "distributions/EvolHMM"]

tests = ["distributions/EvolHMM"]

@testset "TorusEvol" begin
    @testset "Test $t" for t in all_tests
        include("$t.jl")
    end
    print(to)
end
