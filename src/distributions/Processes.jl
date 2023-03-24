using Distributions, DistributionsAD
using LinearAlgebra
using Memoization

# __________________________________________________________________________________________
# Interface for abstract (continuous time) Markovian Processes
abstract type AbstractProcess{F<:VariateForm, S<:ValueSupport} end

# The stationary distribution of the process
function statdist(p::AbstractProcess{F, S}) <: Distribution{F, S} end

# The transition distribution at time t starting from x₀
function transdist(p::AbstractProcess{F, S}, t::Real, x₀) <: Distribution{F, S} end


# __________________________________________________________________________________________
# Product of several processes - when modelling certain evolutionary events independently
struct ProductProcess :< AbstractProcess
    processes::AbstractVector{AbstractProcess}
end

statdist(p::ProductProcess) = arraydist(statdist.(p.processes))
transdist(p::ProductProcess, t::Real, x₀) = arraydist(transdist.(p.procceses, Ref(t), Ref(x₀)))


# __________________________________________________________________________________________
# Continuous Time Markov Chain (with discrete state)
struct CTMC <: AbstractProcess
    statdist::Categorical     # stationary distribution
    Q::AbstractMatrix{<:Real} # rate matrix
end

function CTMC(Q::AbstractMatrix{<:Real})
    # compute statdist from eq: statdist @ Q = 0
    n = size(Q, 1)
    statdist = transpose(Q) \ zeros(n)
    return CTMC(statdist, Q)
end

@memoize transmat(c::CTMC, t::Real) = exp(t * c.Q)

statdist(c::CTMC{T}) = c.statdist
function transdist(c::CTMC, t::Real{T}, x₀::T)
    P = transmat(c, t)
    return Categorical(vec(P[x₀, :]))
end
