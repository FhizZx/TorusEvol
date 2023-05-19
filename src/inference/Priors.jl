using Turing, DynamicPPL
using LinearAlgebra
using LogExpFunctions
using Plots, StatsPlots
using Random
using Distributions

import Base: length, eltype
import Distributions: _rand!, logpdf

struct ScaledBeta <: ContinuousUnivariateDistribution
    be::Beta
    function ScaledBeta(α::Real, β::Real)
        new(Beta(α, β))
    end
end
Distributions.rand(rng::AbstractRNG, d::ScaledBeta) = rand(rng, d.be)*2 - 1
Distributions.logpdf(d::ScaledBeta, x::Real) = logpdf(d.be, (x+1) / 2)


struct CompetingExponential <: ContinuousMultivariateDistribution
    ex::Exponential
    function CompetingExponential(rate::Real)
        new(Exponential(rate))
    end
end
Base.eltype(d::CompetingExponential) = Float64
Base.length(d::CompetingExponential) = 2

function Distributions._rand!(rng::AbstractRNG, d::CompetingExponential, x::AbstractVector{<:Real})
    λ = rand(rng, d.ex)
    μ = rand(rng, d.ex)
    if λ > μ
        tmp = λ; λ = μ; μ=tmp
    end
    x .= [λ, μ]
    return x
end

function Distributions._logpdf(d::CompetingExponential, x::AbstractArray)
    if x[1] > x[2]
        return -Inf
    end
    return log(2) + logpdf(d.ex, x[1]) + logpdf(d.ex, x[2])
end

@model function tkf92_prior()
    λμ ~ CompetingExponential(0.1)
    λ = λμ[1]; μ = λμ[2]
    r ~ Uniform(0.0, 1.0)

    # Require birth rate lower than death rate
    if λ > μ || λ ≤ 0 || μ ≤ 0 || r ≤ 0 || r ≥ 1
        μ = NaN; λ = NaN
    end
    return λ, μ, r
end;

@model function jwndiff_prior()
    μ ~ filldist(Uniform(-π, π), 2)
    σ² ~ filldist(Gamma(π * 0.1), 2)
    α ~ filldist(Gamma(π * 0.1), 2)
    γ ~ Exponential(1.0)   # jumping rate
    α_corr ~ ScaledBeta(3, 3)

    # Require valid covariance matrices
    if any(σ² .≤ 0) || any(α .≤ 0) || γ ≤ 0
        σ² .= NaN; α .= NaN; γ = NaN
    end
    α_cov = α_corr * sqrt(α[1] * α[2])
    if α_cov^2 > α[1]*α[2]
         α_cov = NaN
    end

    return μ[1], μ[2], sqrt(σ²[1]), sqrt(σ²[2]), α[1], α[2], α_cov, γ
end;
