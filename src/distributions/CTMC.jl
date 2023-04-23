using Distributions
using ExponentialAction
using Memoization

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
                      X::AbstractVector,
                      Y::AbstractVector)
    r .= jointlp(c, t)[X, Y]
    return r
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
