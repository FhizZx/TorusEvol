using Distributions
using Bijectors
using Memoization

# ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄
# Jumping process - returns to stationary distribution with some rate γ
struct JumpingProcess{D <: Distribution,
                      RawProc <: AbstractProcess{D}
                      } <: AbstractProcess{D}
    p::RawProc         # original process
    γ::Real            # jumping rate
end

rate(j::JumpingProcess) = j.γ
raw_process(j::JumpingProcess) = j.p

jumping(p, rate::Real) = JumpingProcess(p, rate)

function JumpingWrappedDiffusion(μ_𝜙::Real, μ_𝜓::Real,
                                 σ_𝜙::Real, σ_𝜓::Real,
                                 α_𝜙::Real, α_𝜓::Real, α_cov::Real,
                                 γ::Real)
    return jumping(WrappedDiffusion(μ_𝜙, μ_𝜓, σ_𝜙, σ_𝜓, α_𝜙, α_𝜓, α_cov), γ)
end

statdist(j::JumpingProcess) = statdist(j.p)
transdist(j::JumpingProcess, t::Real, x₀::AbstractVector{<:Real}) = JumpingProcessNode(j, t, x₀)
function transdist!(r::AbstractVector, j::JumpingProcess,
                    t::Real, X₀::AbstractVecOrMat{<:Real})
    jump_prob = 1 - exp(- rate(j) * t)
    lp = log1mexp(-rate(j) * t)
    lnp = -rate(j) * t
    a = Array{Distribution}(undef, size(r)); transdist!(a, raw_process(j), t, X₀)
    r .= JumpingProcessNode.(Ref(statdist(raw_process(j))),
                             a, Ref(jump_prob), Ref(lp), Ref(lnp))
    r
end

# optimized for jumping processes
# Create a matrix r[i, j] = ℙ[Xᵢ, Yⱼ | process p]
# which gives the joint probability of points Xᵢ and Yⱼ under process p
function jointlogpdf!(r::AbstractMatrix{<:Real}, p::JumpingProcess{D}, t::Real,
                      X::AbstractVecOrMat{<:Real},
                      Y::AbstractVecOrMat{<:Real}) where D <: Distribution
    n = size(X, 2)
    m = size(Y, 2)
    # Construct transition distributions for each datapoint in Y
    transdists = Array{D}(undef, m)
    transdist!(transdists, raw_process(p), t, Y)

    #r .= logpdf(statdist(p), Y)'
    lp = fill(log1mexp(-rate(p) * t), n)
    lnp = -rate(p) * t

    # Make each row of r into the log probability of the stationary distribution at Y
    # Add to each column the log transition density from Y to X
    tpd = similar(r, n); tpd .=-Inf
    stX = similar(tpd); stX .=-Inf
    for j ∈ 1:m
        logpdf!(tpd, transdists[j], X)
        logpdf!(stX, statdist(p), X)
        stY = logpdf(statdist(p), Y[:, j])
        r[:, j] .= stY .+ logaddexp.(lp .+ stX, lnp .+ tpd)
    end
    return r
end

# Print distribution parameters
show(io::IO, j::JumpingProcess) = print(io, "JumpingProcess(" *
                                            "\nprocess: " * string(raw_process(j)) *
                                            "\njump rate γ: " * string(rate(j)) *
                                            "\n)")



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

show(io::IO, n::JumpingProcessNode) = print(io, "JumpingProcessNode(" *
                                            "\nstat dist: " * string(n.statdist) *
                                            "\nraw trans dist: " * string(n.raw_transdist) *
                                            "\njump prob: " * string(n.jump_prob) *
                                            "\n)")
