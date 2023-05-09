using Distributions

# __________________________________________________________________________________________
# Testing Methods

# Compute how much of the mass of the unwrapped distribution is recovered by 𝕃 into [-π, π)ᵈ
# Should be close to 1.0
function totalmass(dd::ContinuousMultivariateDistribution; step=π/100)
    d = length(dd)
    grid = hcat(map(collect, vec(collect(Base.product(fill(-π:step:π, d)...))))...)
    A = (2π)^d
    exp(logsumexp(logpdf(dd, grid))) * A / size(grid, 2)
end

function check_mass(d::ContinuousMultivariateDistribution)
    mass = totalmass(d; step=π/100)
    @test isapprox(mass, 1.0; atol=1e-4)
end
