using Distributions, Random
using PDMats
using LogExpFunctions, LinearAlgebra
using Plots

import Base: length, eltype, show
import Distributions: _logpdf, _logpdf!, mean, _rand!


# ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄
# Wrapped Diffusion

# WD(μ, Σ, Α) - analogue of OU(μ, Σ, Α) over 𝕋ᵈ, with stationary distribution ~ WN(μ, ½Α⁻¹Σ)
struct WrappedDiffusion{T <: Real,
                        InfCov <: AbstractPDMat{T},
                        Drift <: AbstractMatrix{T},
                        StatCov <: AbstractPDMat{T},
                        Mean <: AbstractVector{T}
                        }

    μ::Mean                 # mean
    Σ::InfCov               # infinitesimal covariance
    Α::Drift                # drift
    _½Α⁻¹Σ::StatCov         # stationary covariance

    statdist::WrappedNormal{T, StatCov, Mean} # stationary distribution

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

    WrappedDiffusion{T, typeof(Σ), typeof(Α), typeof(_½Α⁻¹Σ), typeof(μ)}(
                     μ, Σ, Α, _½Α⁻¹Σ,
                     w, 𝑟(Α), 𝑞(Α))
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
function WrappedDiffusion(μ::AbstractVector{<:Real},
                          σ_𝜙::Real, σ_𝜓::Real,
                          α_𝜙::Real, α_𝜓::Real, α_cov::Real)
    @assert length(μ) == 2
    @assert (α_𝜙 * α_𝜓 > (α_cov^2)) string(α_𝜙) * " " * string(α_𝜓)* " " * string(α_cov)
    @assert σ_𝜙 * σ_𝜓 > 0

    Σ = PDiagMat([σ_𝜙^2, σ_𝜓^2])
    Α = [α_𝜙 (σ_𝜙*α_cov/σ_𝜓); (σ_𝜓*α_cov/σ_𝜙) α_𝜓]
    WrappedDiffusion(μ, Σ, Α)
end

# __________________________________________________________________________________________
# Probability methods

# Domain Dimension
length(𝚯::WrappedDiffusion) = length(𝚯.μ)

# Domain field type
eltype(𝚯::WrappedDiffusion) = eltype(𝚯.statdist);

# Mean of stationary distribution of 𝚯
mean(𝚯::WrappedDiffusion) = 𝚯.μ

# Stationary truncated lattice
lattice(𝚯::WrappedDiffusion) = lattice(𝚯.statdist)

# Stationary distribution
statdist(𝚯::WrappedDiffusion) = 𝚯.statdist

# Transition distribution
transdist(𝚯::WrappedDiffusion, t::Real, θ₀::AbstractVector{<:Real}) =
    WrappedDiffusionNode(𝚯, t, θ₀)


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
function WrappedDiffusionNode(𝚯, t::Real, θ₀)
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


# __________________________________________________________________________________________
# Distribution Methods

# Domain Dimension
length(d::WrappedDiffusionNode) = length(d.𝚯);

# Domain field type
eltype(d::WrappedDiffusionNode) = eltype(d.𝚯);


# Generate samples according to the transition distribution
function _rand!(rng::AbstractRNG, d::WrappedDiffusionNode, x::VecOrMat{<:Real})
    n = size(x, 2)

    # step 1 - sample from winddist
    windnums = rand(rng, d.winddist, n)

    # step 2 - sample from WN(μᵐₜ, Γₜ)
    for i ∈ 1:n
        x[:, i] .= rand(rng, d.driftdists[windnums[i]])
    end

    x
end

# Log density of WN over 𝕋ᵈ
_logpdf(d::WrappedDiffusionNode, x::AbstractVector{<: Real}) = _logpdf!(Array{Real}(undef, 1), d, x)[1]

function _logpdf!(r::AbstractArray{<:Real},
                  d::WrappedDiffusionNode, X::AbstractArray{<: Real})
    t = d.t

    # if t == 0, distribution degenerates into Dirac(θ₀)
    if t < eps(typeof(t))
        r .= map(θₜ -> d.θ₀ == θₜ, cmod(X))
    else
        # r .= logsumexp(logpdf.(d.driftdists, Ref(cmod(X))) .+
        #                log.(pdf(d.winddist));
        #                dims=1)
        logsumexp!(r, logpdf.(d.driftdists, Ref(cmod(X))) .+ log.(pdf(d.winddist)))
    end

    r
end


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



function anim_tpd_2D(𝚯, θ₀)
    times=[0.00001, 0.0001, 0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7,0.8,
           0.9, 1.0, 1.2, 1.4, 1.7, 2.0, 3.0, 4.0, 5.0, 6.0, 10.0, 15.0, 20.0,
           60.0, 100.0, 180.0]

    anim = @animate for t ∈ times
        println(t)
        plotpdf(transdist(𝚯, t, θ₀))
    end

    gif(anim, "tpd.gif", fps=5)
end

function anim_logtpd_2D(𝚯, θ₀)
    times=[0.0, 0.00001, 0.0001, 0.001, 0.01, 0.1]

    anim = @animate for t ∈ times
        println(t)
        plot_logtpd_2D(𝚯, t, θ₀)
    end

    gif(anim, "images/logtpd.gif", fps=1)
end


w = WrappedNormal([2.5, 0.5], PDiagMat([0.6, 1.5]))
diff = WrappedDiffusion([0.0, 1.0], 1.0, 1.0, 4.0, 1.0, 1.0)

using Random

function diffuse(𝚯, θ₀; Δt=0.05, n_steps=250)
    θ = θ₀
    samples = zeros(length(θ), n_steps)
    anim = @animate for i ∈ 1:n_steps
        θ = rand(transdist(𝚯, Δt, θ))
        samples[:, i] = θ
        l = max(1, i - 2)
        scatter(samples[1, l:i], samples[2, l:i], size=(400,400), xlims=(-π, π), ylims=(-π, π), title="WN DIffusion, i = " * string(i), label="", alpha=1.0)
    end
    gif(anim, "images/diffusion.gif", fps=10)

end

function diffuse2(𝚯, θ₀; Δt=0.05, n_steps=250)
    θ = θ₀
    anim = @animate for i ∈ 1:n_steps
        t = transdist(𝚯, Δt, θ)
        θ = rand(t)
        plotpdf(t; step=π/5)
    end
    gif(anim, "images/diffusion.gif", fps=10)

end
# # Transition probability from θ₀ to θₜ, in time t
# function tpd(𝚯::WrappedDiffusion, t::Real, θ₀::AbstractVector,
#              θₜ::AbstractVector)
#     # if t == 0, distribution degenerates into Dirac(θ₀)
#     if t < eps(typeof(t))
#         return θ₀ ≈ θₜ
#     end

#     e⁻ᵗᴬ= drift_coefficient(𝚯, t)
#     Γₜ = Γ(𝚯, t)
#     driftdist = WrappedNormal(𝚯.μ, Γₜ)
#     winddist = WindingNumber(𝚯, θ₀)

#     pdf(driftdist, θₜ .- e⁻ᵗᴬ * (θ₀ - 𝚯.μ .+ lattice(𝚯))) ⋅ pdf(winddist, lattice(𝚯))
# end

# function _tpd!(r::AbstractArray{<:Real},
#                𝚯::WrappedDiffusion, t::Real, θ₀::AbstractVector{<: Real},
#                x::AbstractMatrix{<: Real})
#     # if t == 0, distribution degenerates into Dirac(θ₀)
#     if t < eps(typeof(t))
#         r .= map(θₜ -> θ₀ ≈ θₜ, x)
#     end

#     e⁻ᵗᴬ= drift_coefficient(𝚯, t)
#     Γₜ = Γ(𝚯, t)
#     drifted_wn= WrappedNormal(𝚯.μ, Γₜ)
#     winding_dist = WindingNumber(𝚯, θ₀)

#     r .= map(θₜ -> pdf(drifted_wn, θₜ .- e⁻ᵗᴬ * (θ₀ - 𝚯.μ .+ lattice(𝚯))) ⋅
#                   pdf(winding_dist, lattice(𝚯)), x)
#     r
# end

# function _logtpd(𝚯::WrappedDiffusion, t::Real,
#                  θ₀::AbstractVector{<: Real},
#                  θₜ::AbstractVector{<: Real})
#     # if t == 0, distribution degenerates into Dirac(θ₀)
#     if t < eps(typeof(t))
#         return θ₀ ≈ θₜ
#     end

#     e⁻ᵗᴬ= drift_coefficient(𝚯, t)

#     Γₜ = Γ(𝚯, t)
#     drifted_wn= WrappedNormal(𝚯.μ, Γₜ)
#     winding_dist = WindingNumber(𝚯, θ₀)

#     logsumexp(logpdf(drifted_wn, θₜ .- e⁻ᵗᴬ * (θ₀ - 𝚯.μ .+ lattice(𝚯))) .+
#               logpdf(winding_dist, lattice(𝚯)))
# end

# function _logtpd!(r::AbstractArray{<:Real},
#                   𝚯::WrappedDiffusion, t::Real, θ₀::AbstractVector{<: Real},
#                   x::AbstractArray{<: Real})
#     # if t == 0, distribution degenerates into Dirac(θ₀)
#     if t < eps(typeof(t))
#         r .= map(θₜ -> θ₀ == θₜ, x)
#     end

#     e⁻ᵗᴬ= drift_coefficient(𝚯, t)
#     Γₜ = Γ(𝚯, t)
#     drifted_wn= WrappedNormal(𝚯.μ, Γₜ)
#     winding_dist = WindingNumber(statdist(𝚯), θ₀)

#     r .= map(θₜ -> logsumexp(logpdf(drifted_wn, θₜ .- e⁻ᵗᴬ * (θ₀ - 𝚯.μ .+ lattice(𝚯))) .+
#                             logpdf(winding_dist, lattice(𝚯))),
#              eachcol(x))
#     r
# end

# # Sample from the tpd, given t time has passed and starting position θ₀
# function _sample(𝚯::WrappedDiffusion, t::Real, θ₀::AbstractVector)

#     # step 1 - sample M from winding_dist
#     winding_dist = WindingNumber(𝚯, θ₀)
#     M = rand(winding_dist)

#     # step 2 - sample from WN(μᵐₜ, Γₜ)
#     e⁻ᵗᴬ= drift_coefficient(𝚯, t)
#     Γₜ = Γ(𝚯, t)
#     μᵐₜ = cmod(𝚯.μ + shifted_drifted_μ(𝚯, e⁻ᵗᴬ, θ₀, M))
#     drifted_normal = WrappedNormal(μᵐₜ, Γₜ)

#     θₜ = rand(drifted_normal, 1)[:,1]

#     return θₜ
# end
# function plot_tpd_2D(𝚯::WrappedDiffusion, t::Real, θ₀::AbstractVector; step=π/30)
#     node = drifteddist(𝚯, t, θ₀)
#     Z = [pdf(node,[j,i]) for i in -π:step:π, j in -π:step:π]
#     heatmap(-π:step:π, -π:step:π, Z, size=(400,400), title="TPD t=" * string(t))
# end

# function plot_logtpd_2D(𝚯::WrappedDiffusion, t::Real, θ₀::AbstractVector; step=π/100)
#     node = drifteddist(𝚯, t, θ₀)
#     Z = [logpdf(node,[j,i]) for i in -π:step:π, j in -π:step:π]
#     heatmap(-π:0.1:π, -π:0.1:π, Z, size=(400,400), title="logTPD t=" * string(t))
# end
