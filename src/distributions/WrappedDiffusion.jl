using Distributions, Random
using PDMats
using LogExpFunctions, LinearAlgebra
using Plots

using Bijectors
import Base: length, eltype, show
import Distributions: _logpdf, _logpdf!, mean, _rand!


# ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄
# Wrapped Diffusion

# WD(μ, Σ, Α) - analogue of OU(μ, Σ, Α) over 𝕋ᵈ, with stationary distribution ~ WN(μ, ½Α⁻¹Σ)
struct WrappedDiffusion <: AbstractProcess{ContinuousMultivariateDistribution}
    μ::AbstractVector{<:Real}           # mean
    Σ::AbstractMatrix{<:Real}           # infinitesimal covariance
    Α::AbstractMatrix{<:Real}           # drift
    _½Α⁻¹Σ::AbstractMatrix{<:Real}      # stationary covariance

    statdist::WrappedNormal # stationary distribution

    r::Real                 # memoized 𝑟(Α)
    q::Real                 # memoized 𝑞(Α)
end


# __________________________________________________________________________________________
# Constructors

function WrappedDiffusion(μ::AbstractVector{T},
                          Σ::AbstractPDMat{T},
                          Α::AbstractMatrix{T}) where T <: Real
    # Make sure ½Α⁻¹Σ is positive definite
    _½Α⁻¹Σ = PDMat(Symmetric(0.5 * inv(Α) * Σ))
    μ = cmod(μ)
    w = WrappedNormal(μ, _½Α⁻¹Σ)

    WrappedDiffusion(μ, Σ, Α, _½Α⁻¹Σ, w, 𝑟(Α), 𝑞(Α))
end

# Make μ, Σ and Α have the same element type
function WrappedDiffusion(μ::AbstractVector{<:Real},
                          Σ::AbstractPDMat{<:Real},
                          Α::AbstractMatrix{<:Real})
    R = Base.promote_eltype(μ, Σ, Α)
    WrappedDiffusion(convert(AbstractArray{R}, μ),
                     convert(AbstractArray{R}, Σ),
                     convert(AbstractArray{R}, Α))
end

# Ensure Σ positive definite
function WrappedDiffusion(μ::AbstractVector{<:Real},
                          Σ::AbstractMatrix{<:Real},
                          Α::AbstractMatrix{<:Real})
    WrappedDiffusion(μ, PDMat(Σ), Α)
end

# Constructor for 2-dimensional drift
# σ -- variance coefficient of each angle
# α -- drift coefficient for each angle, as well as their drift covariance
function WrappedDiffusion(μ_𝜙::Real, μ_𝜓::Real,
                          σ_𝜙::Real, σ_𝜓::Real,
                          α_𝜙::Real, α_𝜓::Real, α_cov::Real)
    @assert (α_𝜙 * α_𝜓 > α_cov^2) string(α_𝜙) * " " * string(α_𝜓)* " " * string(α_cov)
    @assert σ_𝜙 * σ_𝜓 > 0

    Σ = PDiagMat([σ_𝜙^2, σ_𝜓^2])
    Α = [α_𝜙 (σ_𝜙*α_cov/σ_𝜓); (σ_𝜓*α_cov/σ_𝜙) α_𝜓]
    WrappedDiffusion([μ_𝜙, μ_𝜓], Σ, Α)
end

# __________________________________________________________________________________________
# Probability methods

# Domain Dimension
Base.length(𝚯::WrappedDiffusion) = length(𝚯.μ)

# Domain field type
Base.eltype(𝚯::WrappedDiffusion) = eltype(𝚯.statdist);

# Mean of stationary distribution of 𝚯
mean(𝚯::WrappedDiffusion) = 𝚯.μ

# Stationary truncated lattice
lattice(𝚯::WrappedDiffusion) = lattice(𝚯.statdist)

# Stationary distribution
statdist(𝚯::WrappedDiffusion) = 𝚯.statdist

# Transition distribution
transdist(𝚯::WrappedDiffusion, t::Real, θ₀::AbstractVector{<:Real}) =
    WrappedDiffusionNode(𝚯, t, θ₀)

# Optimized for several transition starting points
function transdist!(r::AbstractVector, 𝚯::WrappedDiffusion,
                    t::Real, Θ₀::AbstractVecOrMat{<:Real})
    _WrappedDiffusionNodes!(r, 𝚯, t, Θ₀)
end

function recenter(𝚯::WrappedDiffusion, μ::AbstractVector{<:Real})
    return WrappedDiffusion(μ, 𝚯.Σ, 𝚯.Α)
end

# __________________________________________________________________________________________
# Transition Density Computation Methods

𝑟(Α::AbstractMatrix{<:Real}) = tr(Α) / 2;
𝑞(Α::AbstractMatrix{<:Real}) = √(abs(det(Α - 𝑟(Α)I)))
function 𝑎(t::Real, r::Real, q::Real)
    E = exp(-r*t) * (cosh(q*t) + r * sinh(q*t) / q)
    isnan(E) ? 0.0 : E # handle 0 * inf
end
function 𝑏(t::Real, r::Real, q::Real)
    E = exp(-r*t)*(sinh(q*t) / q)
    isnan(E) ? 0.0 : E # handle 0 * inf
end
𝑠(t::Real, r::Real, q::Real) = 1 - 𝑎(2t, r, q);
𝑖(t::Real, r::Real, q::Real) = 𝑏(2t, r, q) / 2;

# e⁻ᵗᴬ - the 'drag' of the diffusion - attraction to current point
function drift_coefficient(𝚯::WrappedDiffusion, t::Real)
    if length(𝚯) == 2
        return 𝑎(t, 𝚯.r, 𝚯.q) * I + 𝑏(t, 𝚯.r, 𝚯.q) * 𝚯.Α
    else
        throw("e⁻ᵗᴬ only implemented for 2 dimensions")
    end
end

# Γₜ - accumulated covariance through time
function Γ(𝚯::WrappedDiffusion, t::Real)
    @assert t > 0
    if length(𝚯) == 2
        Γₜ = 𝑠(t, 𝚯.r, 𝚯.q) * Matrix(𝚯._½Α⁻¹Σ) + 𝑖(t, 𝚯.r, 𝚯.q) * Matrix(𝚯.Σ)
    else
        throw("Γ only implemented for 2 dimensions")
    end
    Γₜ
end


# ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄
# Wrapped Diffusion Transition Distribution Node

# This defines a pdf p(x) = tpdₜ(θ₀, x | 𝚯)
# which is the distribution of a particle moving according to a wrapped normal diffusion
# 𝚯, started at location θ₀ and allowed to diffuse for time t
struct WrappedDiffusionNode <: ContinuousMultivariateDistribution
    𝚯::WrappedDiffusion
    t::Real
    θ₀::AbstractVector{<:Real}

    # driftdists[k] = WN(μᵐₜ, Γₜ), where m = 𝕃[k]
    driftdists::AbstractVector{WrappedNormal}

    # winddist[k] = wₖ(θ) gives the probability of the winding number of the unwrapped
    # X ~ N(μ, Σ) being equal to 𝕃[k] ∈ 2πℤᵈ, given θ = cmod(X), sampled over the
    # distribution of WN(μ, Σ)
    # i.e. wₖ(θ) is a measure of the contribution of each point in the 2πℤᵈ lattice to the
    # overall pdf
    winddist::Categorical      # Winding Number Weight Distribution

end


# __________________________________________________________________________________________
# Constructors
function WrappedDiffusionNode(𝚯::WrappedDiffusion, t::Real, θ₀::AbstractVector{<:Real})
    # hacky - idea is that if an angle is missing we want the transition
    # distribution to be as diffuse as possible
    if any(isnan.(θ₀))
        driftdists = [WrappedNormal([0.0, 0.0], 5*I)]
        return WrappedDiffusionNode(𝚯, t, θ₀, driftdists, Categorical([1.0]))
    end


    e⁻ᵗᴬ = drift_coefficient(𝚯, t)
    μᴹₜ = cmod(mean(𝚯) .+ e⁻ᵗᴬ * (θ₀ - mean(𝚯) .+ lattice(𝚯)))
    Γₜ = Γ(𝚯, t)
    driftdists = WrappedNormal.(eachcol(μᴹₜ), Ref(Γₜ))

    wn = statdist(𝚯)
    shifted_lattice = θ₀ .+ lattice(wn)
    p = exp.(logpdf(unwrapped(wn), shifted_lattice) .- logpdf(wn, θ₀))
    winddist = Categorical(p)

    WrappedDiffusionNode(𝚯, t, θ₀, driftdists, winddist)
end

# Optimized for several nodes
function _WrappedDiffusionNodes!(r::AbstractVector, 𝚯::WrappedDiffusion,
                                 t::Real, Θ₀::AbstractVecOrMat{<:Real})
    e⁻ᵗᴬ = drift_coefficient(𝚯, t)
    Γₜ = PDMat(Γ(𝚯, t))
    wn = statdist(𝚯)
    𝕃 = lattice(WrappedNormal(zeros(length(𝚯)), Γₜ))

    for i ∈ axes(Θ₀, 2)
        @views θ₀ = Θ₀[:, i]
        if any(isnan.(θ₀))
            driftdists = [WrappedNormal([0.0, 0.0], 5*I)]

            r[i] = WrappedDiffusionNode(𝚯, t, θ₀, driftdists, Categorical([1.0]))
        else
            shifted_lattice = θ₀ .+ lattice(wn)
            μᴹₜ = cmod(mean(𝚯) .+ e⁻ᵗᴬ * (shifted_lattice .- mean(𝚯)))

            normals = MvNormal.(eachcol(μᴹₜ), Ref(Γₜ))
            driftdists = WrappedNormal.(normals, Ref(𝕃))

            p = logpdf(unwrapped(wn), shifted_lattice) .- logpdf(wn, θ₀)
            p .= exp.(p .- logsumexp(p))
            winddist = Categorical(p)
            r[i] = WrappedDiffusionNode(𝚯, t, θ₀, driftdists, winddist)
        end
    end
    r
end


# __________________________________________________________________________________________
# Distribution Methods

# Domain Dimension
length(d::WrappedDiffusionNode) = length(d.𝚯);

# Domain field type
eltype(d::WrappedDiffusionNode) = eltype(d.𝚯);

# optimized
# Generate samples according to the transition distribution
function _rand!(rng::AbstractRNG, d::WrappedDiffusionNode, x::AbstractVecOrMat{<:Real})
    n = size(x, 2)

    # step 1 - sample from winddist
    windnums = rand(rng, d.winddist, n)

    # step 2 - sample from WN(μᵐₜ, Γₜ)
    for w ∈ eachindex(d.driftdists)
        indx = windnums .== w
        if any(indx)
            @views @timeit to "rand" _rand!(rng, d.driftdists[w], x[:, indx])
        end
    end

    x
end

# Log density of WN over 𝕋ᵈ
function _logpdf(d::WrappedDiffusionNode, x::AbstractVector{<: Real})
    t = d.t
    θ₀ = d.θ₀
    θₜ = cmod(x)
     # if t == 0, distribution degenerates into Dirac(θ₀)
    if t < eps(typeof(t))
        return θ₀ .== θₜ ? Inf : -Inf
    end

    return logsumexp(logpdf.(d.driftdists, Ref(θₜ)) .+ log.(pdf(d.winddist)))
end

function Distributions._logpdf!(r::AbstractArray{<:Real},
                                d::WrappedDiffusionNode, X::AbstractVecOrMat{<: Real})
    t = d.t
    wrapped_X = cmod.(X)

    # if t == 0, distribution degenerates into Dirac(θ₀)
    if t < eps(typeof(t))
        r .= map(θₜ -> d.θ₀ == θₜ ? Inf : -Inf, eachcol(wrapped_X))
    else
        r .= -Inf
        tape = similar(r)
        tape .= -Inf
        for i ∈ eachindex(d.driftdists)
            tape .= _logpdf!(tape, d.driftdists[i], wrapped_X)
            r .= logaddexp.(r, tape .+ log(pdf(d.winddist)[i]))
        end
    end

    r
end


Bijectors.bijector(d::WrappedDiffusionNode) = Bijectors.Logit{1, Real}(-π, π)


# __________________________________________________________________________________________
# Plotting Methods

# Print distribution parameters
show(io::IO, 𝚯::WrappedDiffusion) = print(io, "WrappedDiffusion(" *
                                          "\ndim: " * string(length(𝚯)) *
                                          "\nμ: " * string(𝚯.μ) *
                                          "\nΣ: " * string(𝚯.Σ) *
                                          "\nΑ: " * string(𝚯.Α) *
                                          "\n)")

show(io::IO, n::WrappedDiffusionNode) = print(io, "WrappedDiffusionNode(" *
                                              "\n𝚯: " * string(n.𝚯) *
                                              "\nt: " * string(n.t) *
                                              "\nθ₀: " * string(n.θ₀) *
                                              "\n)")
