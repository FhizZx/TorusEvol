using Distributions
using LogExpFunctions

# Matrix distribution of size N_X+1 * N_Y+1
# over the joint log probabilities of 2 sequences of lengths
# N_X and N_Y
struct PairDataHMM <: ContinuousMatrixDistribution
    model::TKF92
    α::AbstractArray{<:Real}
    N_X::Integer
    N_Y::Integer
end

function PairDataHMM(model::TKF92, N_X::Integer, N_Y::Integer)
    α = Array{Real}(undef, N_X + 1, N_Y + 1, num_states(model))
    return PairDataHMM(model, α, N_X, N_Y)
end

size(d::PairDataHMM) = (d.N_X+1, d.N_Y+1)

function _rand(rng::AbstractRNG, d::PairDataHMM, A::AbstractMatrix{<:Real})
    A .= rand(d.N_X+1, d.N_Y+1)
    return A
end

# lℙ[X, Y | params]
function _logpdf(d::PairDataHMM, emission_lps::AbstractMatrix{<:Real})
    α = d.α
    model = d.model
    logp = forward!(α, model, emission_lps; known_ancestor=true)
    return logp
end

# α[state, indices] = ℙ[joint data up to indices, state = last state in HMM]
function forward!(α::AbstractArray{<:Real}, model::TKF92,
                  emission_lps::AbstractArray{<:Real};
                  known_ancestor=false)
    D = num_descendants(model)

    α = fill!(α, -Inf)
    # Initial state
    α[ones(Int, D+1)..., START_INDEX] = 0
    axs = [a .- 1 for a ∈ axes(emission_lps)]
    grid = map(collect, collect(Iterators.product(axs...))) # 0:N_X x 0:N_Y
    end_corner = size(emission_lps) # N_X + 1 by N_Y + 1

    #values = known_ancestor ? values(model) : descendant_values(model)
    #A = known_ancestor ? transmat(model) :
    A = transmat(model)
    # Recursion
    for indices ∈ grid, s ∈ proper_state_ids(model)
        state = state_values(model)[s]
        curr_αind = 1 .+ indices
        prev_αind = curr_αind .- state
        if all(prev_αind .>= 1)
            obs_ind = @. ifelse(state == 1, indices, end_corner)

            @views α[curr_αind..., s] = emission_lps[obs_ind...] +
                                        logsumexp(A[:, s] .+ α[prev_αind..., :])
        end
    end

    logp = logsumexp(A[:, END_INDEX] .+ α[end_corner..., :])

    return logp
end
