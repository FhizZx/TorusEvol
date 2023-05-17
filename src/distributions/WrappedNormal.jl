using Distributions, Random
using PDMats
using LinearAlgebra
using Plots
using DistributionsAD
using Bijectors
using Statistics
using LogExpFunctions

import Base: length, eltype, show, size
import Distributions: _logpdf, _logpdf!, mean, _rand!

using FastLogSumExp: mat_logsumexp_dual_reinterp! as fastlogsumexp!

#using FastLogSumExp:

# __________________________________________________________________________________________
# Helper functions

# Map x to [-π, π)
cmod(x::Real) = rem(x,2π, RoundNearest)

cmod(v::AbstractArray{<:Real}) = cmod.(v)

# Returns [-r,r]ᵈ ⊂ ℤᵈ
hyper_cube(r::Integer, d::Integer) =
    reshape(hcat(map(collect, vec(collect(Base.product(fill(-r:r, d)...))))...), d, :)

# Returns 2π[-r,r]ᵈ ⊂ 2πℤᵈ
twoπ_hyper_cube(r::Integer, d::Integer) =
    2π .* hyper_cube(r, d)

# Returns {x ∈ 2π[-r,r]ᵈ | d(x, 𝛷) < R} w.r.t Mahalanobis distance
function discrete_ellipsoid(𝛷::MvNormal, r::Real, R::Real)
    d = length(𝛷)
    cube = twoπ_hyper_cube(r, d)
    mindist = 3 * sqmahal(𝛷, zeros(d))
    ellipsoid = cube[:, sqmahal(𝛷, cube) .< max(R, mindist)]
    ellipsoid
end

function discrete_ellipsoid(𝛷::TuringDenseMvNormal, r::Real, R::Real)
    n = MvNormal(mean(𝛷), cov(𝛷))
    discrete_ellipsoid(n, r, R)
end

# ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄
# Wrapped Normal Distribution

# WN(μ, Σ) - analogue of N(μ, Σ) over 𝕋ᵈ
struct WrappedNormal <: ContinuousMultivariateDistribution
    𝛷::ContinuousMultivariateDistribution # the underlying unwrapped distribution
    𝕃::AbstractMatrix{<:Real}             # 𝕃 ⊂ 2πℤᵈ truncated torus lattice
                                          #    used to collect (most of) the probability mass
                                          #    of 𝛷 into [-π, π)ᵈ
end


# # __________________________________________________________________________________________
# # Constructors & Getters

# 𝛷 = N(μ, Σ)
# 𝕃 = 2π[-r,r]ᵈ ∩ B(𝛷, R) (in sqmahal distance)
function WrappedNormal(μ::AbstractVector{<:Real}, Σ)
    𝛷 = MvNormal(cmod(μ), Σ)
    # R = 12.0
    # r = ceil(Int, R * 1.0)
    # 𝕃 = discrete_ellipsoid(𝛷, r, R)

    #temp - consider only the 8 neighbouring quadrants in the lattice
    𝕃 = twoπ_hyper_cube(1, length(μ))

    WrappedNormal(𝛷, 𝕃)
end

# # Make μ and Σ have the same element type
# function WrappedNormal(μ::AbstractVector{<:Real}, Σ::AbstractPDMat{<:Real})
#     Rtype = Base.promote_eltype(μ, Σ)
#     WrappedNormal(convert(AbstractArray{Rtype}, μ), convert(AbstractArray{Rtype}, Σ))
# end

# # Ensure Σ positive definite
# function WrappedNormal(μ::AbstractVector{<:Real}, Σ::AbstractMatrix{<:Real})
#     # will throw an error if Σ is not positive definite
#     WrappedNormal(μ, PDMat(Σ))
# end

function WrappedNormal(μ::Real, Σ::Real)
    WrappedNormal([μ], [Σ;;])
end



# Underlying unwrapped distribution
unwrapped(wn::WrappedNormal) = wn.𝛷

# Truncated lattice used to recover some of the mass of 𝛷 from ℝᵈ
lattice(wn::WrappedNormal) = wn.𝕃


# __________________________________________________________________________________________
# Distribution Methods

const WN_GRID_SIZE = 25
const WN2_DOMAIN = Domain(hcat(map(collect, vec(collect(Base.product(fill(-π:(2π/WN_GRID_SIZE):π, 2)...))))...), WN_GRID_SIZE*WN_GRID_SIZE)

function domain(wn::WrappedNormal)
    if length(wn) == 2
        return WN2_DOMAIN
    end
    return []
end

# Domain Dimension
Base.length(wn::WrappedNormal) = length(wn.𝛷)

# Domain field type
Base.eltype(wn::WrappedNormal) = eltype(wn.𝛷)

# Generate samples according to WN
#optimized
function _rand!(rng::AbstractRNG, wn::WrappedNormal, x::AbstractVecOrMat{<: Real})
    _rand!(rng, wn.𝛷, x)
    x .= cmod.(x)
    return x
end

# Log density of WN over 𝕋ᵈ
function _logpdf(wn::WrappedNormal, x::AbstractVector{<: Real})
    # Handling missing values
    if any(isnan.(x))
        return 0.0
    end
    return logsumexp(logpdf(wn.𝛷, cmod.(x) .+ wn.𝕃))
end

#optimized
function _logpdf!(r::AbstractArray{<: Real},
                  wn::WrappedNormal, X::AbstractVecOrMat{<: Real})
    # Handling missing values
    if any(isnan.(X))
        return 0.0
    end
    shifted_X = cmod.(X)

    tape = similar(r, length(r), 2)
    tape .= -Inf
    r = @view tape[:, 1]
    shifted_logp = @view tape[:, 2]
    prev_col = [0.0, 0.0]
    for col ∈ eachcol(wn.𝕃)
        shifted_X .+= col .- prev_col
        prev_col = col

        logpdf!(shifted_logp, wn.𝛷, shifted_X)
        r .= logaddexp.(r, shifted_logp)
    end
    copy(r)
end

# function _fastlogpdf!(r::AbstractArray{<: Real},
#                       wn::WrappedNormal, X::AbstractMatrix{<: Real})
#     shifted_X = cmod.(X)

#     tape = Array{Real}(undef, length(r), 2)
#     tape .= -Inf
#     r = @view tape[:, 1]
#     shifted_logp = @view tape[:, 2]
#     prev_col = [0.0, 0.0]
#     for col ∈ eachcol(wn.𝕃)
#         shifted_X .+= col .- prev_col
#         prev_col = col
#         #logpdf!(shifted_logp, wn.𝛷, shifted_X)
#         shifted_logp .= logpdf(wn.𝛷, shifted_X)
#         # check dual
#         fastlogsumexp!(r, tape)
#     end
#     copy(r)
# end


# This gives lower numerical errors but allocates much more memory
function _accuratelogpdf!(r::AbstractArray{<: Real},
                          wn::WrappedNormal, X::AbstractMatrix{<: Real})
    d = size(X, 1); n = size(X, 2); m = size(wn.𝕃, 2)
    a = reshape(X, d, n, :) .+ reshape(wn.𝕃, d, :, m)
    lps = reshape(logpdf(wn.𝛷, reshape(a, d, :)), n, m)
    logsumexp!(r, lps)
    r
end


# Mean of WN over 𝕋ᵈ
mean(wn::WrappedNormal) = mean(wn.𝛷)

Statistics.cov(wn::WrappedNormal) = Statistics.cov(wn.𝛷)

Bijectors.bijector(wn::WrappedNormal) = Bijectors.Logit{1, Real}(-π, π)


# __________________________________________________________________________________________
# Plotting Methods

# Print distribution parameters
show(io::IO, wn::WrappedNormal) = print(io, "WrappedNormal(" *
                                          "\ndim: " * string(length(wn)) *
                                          "\nμ: " * string(wn.𝛷.μ) *
                                          "\nΣ: " * string(wn.𝛷.Σ) *
                                          "\n)")

# Plot the points in 𝕃
function plotlattice(wn::WrappedNormal, plt)
    @assert length(wn) <= 3
    scatter!(plt, eachrow(wn.𝕃)...,size=(400,400), title="", label="𝕃")
end


# multiple inits
# function _WrappedNormals(μ::VecOrMat{<:Real}, Σ::AbstractMatrix{<: Real})
#     res = []
#     𝛷 = MvNormal(zeros(size(μ, 1)), Σ)
#     R = 10.0
#     r = ceil(Int, R * 1.5)
#     𝕃 = discrete_ellipsoid(𝛷, r, R)
#     for i ∈ axes(μ, 2)
#         @views 𝛷_i = 𝛷 + μ[:, i]
#         wn = WrappedNormal{eltype(𝛷_i), typeof(𝛷_i.Σ), typeof(mean(𝛷_i))}(𝛷_i, 𝕃)
#         push!(res, wn)
#     end
#     res
# end
