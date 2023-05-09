using Distributions
using LogExpFunctions
using LinearAlgebra


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

    check_mass(wn)
end
