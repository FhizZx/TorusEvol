using Distributions, Random
using PDMats

import Base: length, eltype
import Distributions: _pdf, _pdf!, _logpdf, _logpdf!, _rand!


# ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄


# wₖ(θ) gives the probability of the winding number of the unwrapped X ~ N(μ, Σ)
# being equal to 𝕃[k] ∈ 2πℤᵈ, given θ = cmod(X), sampled over the distribution of WN(μ, Σ)

# i.e. wₖ(θ) is a measure of the contribution of each point in the 2πℤᵈ lattice to the pdf
struct WindingNumber{T <: Real,
                     Cov  <: AbstractPDMat{T},
                     Mean <: AbstractVector{T}
                    } :< DiscreteUnivariateDistribution
    wn::WrappedNormal{T, Cov, Mean}
    θ::AbstractVector{T}
    dist::Categorical        # underlying distribution
    lp::AbstractVector{T}    # precomputed log probabilities
end


# __________________________________________________________________________________________
# Constructors
function WindingNumber(wn::WrappedNormal, θ::AbstractVector)
    shifted_lattice = θ .+ lattice(wn)
    p = pdf(unwrapped(wn), shifted_lattice) ./ pdf(wn, shifted_lattice)
    lp = logpdf(unwrapped(wn), shifted_lattice) .- logpdf(wn, shifted_lattice)
    dist = Categorical(p)
    return WindingNumber(wn, θ, dist, lp)
end


# __________________________________________________________________________________________
# Distribution Methods

# Domain Dimension
length(w::WindingNumber) = length(dist)

# Domain field type
eltype(w::WindingNumber) = eltype(dist)

# Support over 2πℤᵈ ⊂ ℝᵈ
support(w::WindingNumber) = lattice(w.wn)

# Generate samples according to w
function _rand!(rng::AbstractRNG, w::WindingNumber, x::VecOrMat{<: Real})
    # sample from the truncated lattice
    n = size(x, 2)
    x .= Int.(rand(rng, w.C, n))
    return x
end

# Log density of wₖ(θ) over 2πℤᵈ
_logpdf(w::WindingNumber, K::AbstractVector{<: Real}) = _logpdf!(zeros(1), w, K)[1]

function _logpdf!(r::AbstractArray{<: Real},
                  w::WindingNumber, K::AbstractArray{<: Real})
    r .= w.lp[K...]
    r
end;

fulllogdist(w::WindingNumber) = log(pdf(w.dist));
