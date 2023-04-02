using Distributions, Random
using PDMats
using LogExpFunctions, LinearAlgebra
using Plots
using DistributionsAD

import Base: length, eltype, show
import Distributions: _logpdf, _logpdf!, mean, _rand!

# __________________________________________________________________________________________
# Helper functions

# Map x to [-π, π)
cmod(x::Real) = rem2pi(x, RoundNearest)

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
    mindist = 2 * sqmahal(𝛷, zeros(d))
    ellipsoid = cube[:, sqmahal(𝛷, cube) .< max(R*R, mindist)]
    ellipsoid
end

function discrete_ellipsoid(𝛷::TuringDenseMvNormal, r::Real, R::Real)
    n = MvNormal(mean(𝛷), cov(𝛷))
    discrete_ellipsoid(n, r, R)
end

# ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄
# Wrapped Normal Distribution

# WN(μ, Σ) - analogue of N(μ, Σ) over 𝕋ᵈ
struct WrappedNormal{T    <: Real,
                     Cov  <: AbstractPDMat{T}, # enforce that covariance is positive def
                     Mean <: AbstractVector{T}
                    } <: ContinuousMultivariateDistribution

    𝛷::MvNormal{T, Cov, Mean}  # the underlying unwrapped distribution
    𝕃::Matrix{T}               # 𝕃 ⊂ 2πℤᵈ truncated torus lattice
                               #    used to collect (most of) the probability mass
                               #    of 𝛷 into [-π, π)ᵈ
end


# __________________________________________________________________________________________
# Constructors & Getters

# TODO - fiddle with the constants r, R and see if a large lattice is necessary

# 𝛷 = N(μ, Σ)
# 𝕃 = 2π[-r,r]ᵈ ∩ B(𝛷, R) (in sqmahal distance)
function WrappedNormal(μ::AbstractVector{T}, Σ::AbstractPDMat{T}) where T <: Real
    𝛷 = MvNormal(cmod(μ), Σ)
    R = 10.0
    r = ceil(Int, R * 1.5)
    𝕃 = discrete_ellipsoid(𝛷, r, R)
    WrappedNormal{T, typeof(Σ), typeof(mean(𝛷))}(𝛷, 𝕃)
end

# Make μ and Σ have the same element type
function WrappedNormal(μ::AbstractVector{<:Real}, Σ::AbstractPDMat{<:Real})
    Rtype = Base.promote_eltype(μ, Σ)
    WrappedNormal(convert(AbstractArray{Rtype}, μ), convert(AbstractArray{Rtype}, Σ))
end

# Ensure Σ positive definite
function WrappedNormal(μ::AbstractVector{<:Real}, Σ::AbstractMatrix{<:Real})
    # will throw an error if Σ is not positive definite
    WrappedNormal(μ, PDMat(Σ))
end

function WrappedNormal(μ::Real, Σ::Real)
    WrappedNormal([μ], [Σ;;])
end

# Underlying unwrapped distribution
unwrapped(wn::WrappedNormal) = wn.𝛷

# Truncated lattice used to recover some of the mass of 𝛷 from ℝᵈ
lattice(wn::WrappedNormal) = wn.𝕃


# __________________________________________________________________________________________
# Distribution Methods

# Domain Dimension
length(wn::WrappedNormal) = length(wn.𝛷)

# Domain field type
eltype(wn::WrappedNormal) = eltype(wn.𝛷)

# Generate samples according to WN
function _rand!(rng::AbstractRNG, wn::WrappedNormal, x::VecOrMat{<: Real})
    x .= cmod(_rand!(rng, wn.𝛷, x))
    return x
end

# Log density of WN over 𝕋ᵈ
_logpdf(wn::WrappedNormal, x::AbstractVector{<: Real}) = _logpdf!(zeros(1), wn, x)[1]

function _logpdf!(r::AbstractArray{<: Real},
                  wn::WrappedNormal, X::AbstractArray{<: Real})
    n = size(wn.𝕃, 2)
    Θ = cmod.(X)
    lp = logpdf(wn.𝛷, hcat(map(θ -> θ .+ wn.𝕃, eachcol(Θ))...))
    r .= map(v -> logsumexp(v), eachcol(reshape(lp, n, :)))
    r
end

# Mean of WN over 𝕋ᵈ
mean(wn::WrappedNormal) = mean(wn.𝛷)


# __________________________________________________________________________________________
# Plotting Methods

# Print distribution parameters
show(io::IO, wn::WrappedNormal) = print(io, "WrappedNormal(" *
                                          "\ndim: " * string(length(wn)) *
                                          "\nμ: " * string(wn.𝛷.μ) *
                                          "\nΣ: " * string(wn.𝛷.Σ) *
                                          "\n)")

# Plot heatmap of the density over 𝕋²
function plotpdf(wn::ContinuousDistribution; step=π/100)
    ticks = (-π):step:π
    if length(wn) == 2
        grid = hcat([[j, i] for i in ticks, j in ticks]...)
        z = reshape(pdf(wn, grid), length(ticks), :)
        heatmap(ticks, ticks, z, size=(400, 400), title="WN Density",
                xlabel="ϕ angles", ylabel="ψ angles")
    else
        throw("plotting not implemented for d != 2")
    end
end

# Scatter plot of samples from wn
function plotsamples(wn::ContinuousDistribution, n_samples)
    samples = rand(wn, n_samples)
    scatter(eachrow(samples)...,size=(400,400),
            title="WN Samples", label="", alpha=0.3)
end

# Plot the points in 𝕃
function plotlattice(wn::WrappedNormal)
    @assert length(wn) <= 3
    scatter(eachrow(wn.𝕃)...,size=(400,400), title="WN Lattice", label="")
end


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
