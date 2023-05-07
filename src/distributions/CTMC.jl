using Distributions
using ExponentialAction
using Memoization
using LinearAlgebra

# ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄
# Continuous Time Markov Chain (time-reversible with discrete state)
struct CTMC <: AbstractProcess{Categorical}
    statdist::Categorical      # stationary distribution
    Q::AbstractMatrix{<:Real}  # rate matrix
    _I::AbstractMatrix{<:Real} # identity used to compute exp action
end

function CTMC(statvec::AbstractVector{<:Real}, Q::AbstractMatrix{<:Real})
    @assert all(isapprox.(transpose(statvec) * Q, 0.0; atol=1e-16)) string(transpose(statvec) * Q) # statvec ∈ Ker(Qᵗ)
    _I = Matrix{Real}(I, length(statvec), length(statvec))
    return CTMC(Categorical(statvec), Q, _I)
end

function CTMC(Q::AbstractMatrix{<:Real})
    # compute statdist from eq: statdist @ Q = 0
    #todo catch errors when Q doesn't have statdist
    n = size(Q, 1)
    statdist = transpose(Q) \ zeros(n)
    return CTMC(statdist, Q)
end

# Create a matrix r[i, j] = ℙ[Xᵢ, Yⱼ | process p]
# which gives the joint probability of points Xᵢ and Yⱼ under process p
function jointlogpdf!(r::AbstractMatrix{<:Real}, c::CTMC, t::Real,
                      X::AbstractVecOrMat,
                      Y::AbstractVecOrMat)
    @views r .= jointlp(c, t)[vec(X), vec(Y)]
    return r
end

function statlogpdf!(r::AbstractVector{<:Real}, c::CTMC,
                     X::AbstractVecOrMat)
    # need to take transpose to make X vector so that it works with Categorical
    logpdf!(r, statdist(c), X')
end

@memoize transmat(c::CTMC, t::Real) = expv(t, c.Q, c._I)

statdist(c::CTMC) = c.statdist

function jointlp(c::CTMC, t::Real)
    P = transmat(c, t)
    return log.(P) .+ logpdf(statdist(c))
end

function transdist(c::CTMC, t::Real, x₀)
    P = transmat(c, t)
    return Categorical(P[x₀, :])
end

num_states(c::CTMC) = size(c.Q, 1)


show(io::IO, c::CTMC) = print(io, "CTMC(" *
                                  "\nnum states: " * string(num_states(c)) *
                                  "\nstat dist: " * string(round.(statdist(c).p; digits=3)) *
                                  "\nQ matrix: " * string(round.(c.Q; digits=3)) *
                                  "\n)")


# ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄
# CTMC based substitution process

# given some rate matrix Q, we can find the transition matrix P(t)
# by exponentiating Q

# i.e. P(t) = exp(t * Q)

# p(i, j) = π(i) * P(t)[i, j]

# p(i, _) = π(i)

# so the substitution model should implement likelihood Methods
# for i -> j as well as equilibrium frequencies

# Q = S diag(Π) where S is symmetric exchangeability matrix,
#        Π is equilibrium frequencies

function SubstitutionProcess(S::LowerTriangular, Π::AbstractVector)
    S = Symmetric(S, :L)
    S[diagind(S)] .= 0.0
    Q = S * Diagonal(Π)
    Q[diagind(Q)] .= sum(-Q, dims = 2)
    mean_replacement = -dot(Q[diagind(Q)], Π)
    Q ./= mean_replacement
    @assert isapprox(-dot(Q[diagind(Q)], Π), 1.0; atol=1e-8) "mean replacement rate should be 1"
    @assert all(isapprox.(sum(-Q, dims = 2), 0.0; atol=1e-8)) "rows of Q should sum to 0"
    @assert all(isapprox.(Diagonal(Π) * Q, (Diagonal(Π) * Q)'; atol=1e-8)) "Q should satisfy detailed balance w.r.t. Π"

    return CTMC(Π, Q)
end
