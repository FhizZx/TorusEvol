using Distributions, DistributionsAD
using LinearAlgebra
using Memoization

# __________________________________________________________________________________________
# Interface for abstract (continuous time) Markovian Processes
abstract type AbstractProcess{D <: Distribution} end

# The stationary distribution of the process
function statdist(p::AbstractProcess) end

# The transition distribution at time t starting from x₀
function transdist(p::AbstractProcess, t::Real, x₀) end


# __________________________________________________________________________________________
# Product of several processes - when modelling certain evolutionary events independently
struct ProductProcess{D <: Distribution} <: AbstractProcess{D}
    processes::AbstractVector{AbstractProcess{D}}
end

statdist(p::ProductProcess) = arraydist(statdist.(p.processes))
transdist(p::ProductProcess, t::Real, x₀) = arraydist(transdist.(p.procceses, Ref(t), Ref(x₀)))

# __________________________________________________________________________________________
# Jumping process - returns to stationary distribution at some rate γ
struct JumpingProcess
    p # original process
    γ::Real            # jumping rate
end

rate(j::JumpingProcess) = j.γ
raw_process(j::JumpingProcess) = j.p

jumping(p, rate::Real) = JumpingProcess(p, rate)

statdist(j::JumpingProcess) = statdist(j.p)
transdist(j::JumpingProcess, t::Real, x₀) = JumpingProcessNode(j, t, x₀)

struct JumpingProcessNode <: ContinuousMultivariateDistribution
    statdist
    raw_transdist
    jump_prob :: Real # probability of jumping to the stationary distribution
    lp::Real # log(1 - exp(-γt))
    lnp::Real # -γt
end

function JumpingProcessNode(j::JumpingProcess, t::Real, x₀)
    jump_prob = 1 - exp(- rate(j) * t)
    return JumpingProcessNode(statdist(raw_process(j)),
                              transdist(raw_process(j), t, x₀),
                              jump_prob,
                              log1mexp(-rate(j) * t),
                              -rate(j) * t)
end

# Domain Dimension
length(d::JumpingProcessNode) = length(d.statdist)

# Domain field type
eltype(d::JumpingProcessNode) = eltype(d.statdist)

function _rand!(rng::AbstractRNG, d::JumpingProcessNode, x::VecOrMat{<:Real})
    n = size(x, 2)
    has_jumped = rand(rng, Bernoulli(d.jump_prob), n)
    stat_indices = findall(has_jumped)
    trans_indices = findall(.!has_jumped)
    if length(stat_indices) > 0
        x[:, stat_indices] .= rand(rng, d.statdist, length(stat_indices))
    end
    if length(trans_indices) > 0
        x[:, trans_indices] .= rand(rng, d.raw_transdist, length(trans_indices))
    end

    return x

end

_logpdf(d::JumpingProcessNode, x::AbstractVector{<: Real}) = _logpdf!(zeros(1), d, x)[1]

function _logpdf!(r::AbstractArray{<: Real},
                  d::JumpingProcessNode, X::AbstractArray{<: Real})
    r .= logaddexp.(d.lp .+ logpdf(d.statdist, X),
                    d.lnp .+ logpdf(d.raw_transdist, X))
    return r
end

# __________________________________________________________________________________________
# Continuous Time Markov Chain (with discrete state)
struct CTMC #<: AbstractProcess{DiscreteUnivariateDistribution}
    statdist::Categorical     # stationary distribution
    Q::AbstractMatrix{<:Real} # rate matrix
end

function CTMC(statvec::AbstractVector{<:Real}, Q::AbstractMatrix{<:Real})
    @assert all(isapprox.(transpose(statvec) * Q, 0.0; atol=1e-16)) string(transpose(statvec) * Q) # statvec ∈ Ker(Qᵗ)
    return CTMC(Categorical(statvec), Q)
end

function CTMC(Q::AbstractMatrix{<:Real})
    # compute statdist from eq: statdist @ Q = 0
    n = size(Q, 1)
    statdist = transpose(Q) \ zeros(n)
    return CTMC(statdist, Q)
end

@memoize transmat(c::CTMC, t::Real) = exp(t * c.Q)

statdist(c::CTMC) = c.statdist

function transdist(c::CTMC, t::Real, x₀::Real)
    P = transmat(c, t)
    return Categorical(vec(P[round(Int, x₀), :]))
end
