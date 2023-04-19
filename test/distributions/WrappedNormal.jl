using Distributions

# __________________________________________________________________________________________
# Testing Methods

# Compute how much of the mass of the unwrapped distribution is recovered by 𝕃 into [-π, π)ᵈ
# Should be close to 1.0
function totalmass(wn::WrappedNormal; step=π/100)
    d = length(wn)
    grid = map(collect, vec(collect(Base.product(fill(-π:step:π, d)...))))
    A = (2π)^d
    sum(pdf(wn, grid)) * A / length(grid)
end;

@testset "Wrapped Normal dim=2 tests" begin
    dim = 2
    μ = cmod(rand(dim))
    Σ = rand(LKJ(dim, 1.0))
    wn = WrappedNormal(μ, Σ)
    n = unwrapped(wn)

    num_samples = 2000000
    x = rand(dim, num_samples)

    @test logpdf(wn, x) ≈ logpdf.(Ref(wn), eachcol(x)) atol=1e-14
    mass = totalmass(wn)
    @test mass ≈ 1.0 atol=1e-7
end
