using Distributions
using LogExpFunctions
using LinearAlgebra

# __________________________________________________________________________________________
# Testing Methods

# Compute how much of the mass of the unwrapped distribution is recovered by 𝕃 into [-π, π)ᵈ
# Should be close to 1.0
function totalmass(wn::WrappedNormal; step=π/100)
    d = length(wn)
    grid = hcat(map(collect, vec(collect(Base.product(fill(-π:step:π, d)...))))...)
    A = (2π)^d
    exp(logsumexp(logpdf(wn, grid))) * A / size(grid, 2)
end;
const NUM_WN_TESTS=30
@testset "Wrapped Normal dim=2 test $v" for v ∈ 1:NUM_WN_TESTS
    Random.seed!(TEST_SEED+v)
    dim = 2
    μ = rand(Uniform(-π, π), 2)
    σ = sqrt.(rand(InverseGamma(), 2)) ./ π
    Σ = Hermitian(σ .* rand(LKJ(dim, 1.0)) .* σ')
    wn = WrappedNormal(μ, Σ)
    n = unwrapped(wn)
    num_samples = 200
    x = rand(dim, num_samples)

    @test logpdf(wn, x) ≈ logpdf.(Ref(wn), eachcol(x)) atol=1e-14
    mass = totalmass(wn; step=π/100)
    display(Σ)
    @test isapprox(mass, 1.0; atol=1e-4)
end
