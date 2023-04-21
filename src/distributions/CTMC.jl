using Distributions

# ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄
# Continuous Time Markov Chain (with discrete state)
struct CTMC #<: AbstractProcess{DiscreteUnivariateDistribution}
    statdist::Categorical     # stationary distribution
    Q::AbstractMatrix{<:Real} # rate matrix
    B::AbstractMatrix{<:Real}
end

function CTMC(statvec::AbstractVector{<:Real}, Q::AbstractMatrix{<:Real})
    @assert all(isapprox.(transpose(statvec) * Q, 0.0; atol=1e-16)) string(transpose(statvec) * Q) # statvec ∈ Ker(Qᵗ)
    B = Matrix{Real}(I, length(statvec), length(statvec))
    return CTMC(Categorical(statvec), Q, B)
end

function CTMC(Q::AbstractMatrix{<:Real})
    # compute statdist from eq: statdist @ Q = 0
    n = size(Q, 1)
    statdist = transpose(Q) \ zeros(n)
    return CTMC(statdist, Q)
end

@memoize transmat(c::CTMC, t::Real) = expv(t, c.Q, c.B)

statdist(c::CTMC) = c.statdist

function transdist(c::CTMC, t::Real, x₀::Real)
    P = transmat(c, t)
    return Categorical(vec(P[round(Int, x₀), :]))
end
