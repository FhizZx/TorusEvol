using Distributions, Random
using PDMats
using LogExpFunctions, LinearAlgebra
using Plots
using DistributionsAD
using Bijectors
using Statistics

import Base: length, eltype, show, size
import Distributions: _logpdf, _logpdf!, mean, _rand!

struct MyCategorical <: ContinuousUnivariateDistribution
    raw_cat::Categorical
    lp::AbstractVector{<:Real}
end

function MyCategorical(p::AbstractVector{<:Real})
    if sum(p) != 1
        p = normalize(p)
        @warn "Probabilities in categorical construction do not sum up to 1"
    end
    MyCategorical(Categorical(p), log(p))
end

logpdf(d::MyCategorical) = d.lp


# Domain Dimension
Base.length(wn::WrappedNormal) = length(wn.𝛷)

# Domain field type
Base.eltype(wn::WrappedNormal) = eltype(wn.𝛷)

# __________________________________________________________________________________________
# Distribution Methods

function _rand!(rng::AbstractRNG, d::MyCategorical, x::AbstractVecOrMat{<: Real})
    x .= _rand!(rng, d.raw_cat, x')
    return x
end

function _logpdf(d::MyCategorical, x::AbstractVector{<: Real})
    return logsumexp(logpdf(wn.𝛷, cmod.(x) .+ wn.𝕃))
end
