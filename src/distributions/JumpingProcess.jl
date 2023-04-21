using Distributions
using Bijectors

# ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄
# Jumping process - returns to stationary distribution with some rate γ
struct JumpingProcess{D <: ContinuousMultivariateDistribution,
                      RawProc <: AbstractProcess{D}
                      } <: AbstractProcess{D}
    p::RawProc         # original process
    γ::Real            # jumping rate
end

rate(j::JumpingProcess) = j.γ
raw_process(j::JumpingProcess) = j.p

jumping(p, rate::Real) = JumpingProcess(p, rate)

statdist(j::JumpingProcess) = statdist(j.p)
transdist(j::JumpingProcess, t::Real, x₀::AbstractVector{<:Real}) = JumpingProcessNode(j, t, x₀)

#todo optimize logpdf/jointlogpdf for large data size


struct JumpingProcessNode <: ContinuousMultivariateDistribution
    statdist
    raw_transdist
    jump_prob :: Real # probability of jumping to the stationary distribution
    lp::Real # log(1 - exp(-γt))
    lnp::Real # -γt
end

function JumpingProcessNode(j::JumpingProcess, t::Real, x₀::AbstractVector{<:Real})
    jump_prob = 1 - exp(- rate(j) * t)
    JumpingProcessNode(statdist(raw_process(j)),
                       transdist(raw_process(j), t, x₀),
                       jump_prob,
                       log1mexp(-rate(j) * t),
                       -rate(j) * t)
end

# Domain Dimension
length(d::JumpingProcessNode) = length(d.statdist)

# Domain field type
eltype(d::JumpingProcessNode) = eltype(d.statdist)

function _rand!(rng::AbstractRNG, d::JumpingProcessNode, x::AbstractVecOrMat{<:Real})
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

_logpdf(d::JumpingProcessNode, x::AbstractVector{<: Real}) = _logpdf!(Array{Real}(undef, 1), d, x)[1]

function _logpdf!(r::AbstractArray{<: Real},
                  d::JumpingProcessNode, X::AbstractVecOrMat{<: Real})
    r .= logaddexp.(d.lp .+ logpdf(d.statdist, X),
                    d.lnp .+ logpdf(d.raw_transdist, X))
    return r
end

Bijectors.bijector(d::JumpingProcessNode) = bijector(d.raw_transdist)
