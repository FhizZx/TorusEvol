# Test style inspired by the Distributions.jl package
using TorusEvol
using Test
using Random

printstyled("Running tests:\n", color=:blue)
Random.seed!(345679)

const tests = ["distributions/WrappedNormal"]

@testset "TorusEvol" begin
    @testset "Test $t" for t in tests
        include("$t.jl")
    end
end
