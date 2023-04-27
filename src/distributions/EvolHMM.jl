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
    α = Array{Real}(undef, num_states(model), N_X + 1, N_Y + 1)
    return PairDataHMM(model, α, N_X, N_Y)
end

size(d::PairDataHMM) = (d.N_X+1, d.N_Y+1)

#function _rand(rng::AbstractRNG, d::PairDataNode, A::AbstractMatrix{<:Real})

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
    α[START_INDEX, ones(Int, D+1)...] = 0
    grid = Iterators.product(axes(α)[2:end]...)
    end_corner = 1 .+ size(α)

    state_ids = state_ids(model)
    values = known_ancestor ? values(model) : descendant_values(model)
    #A = known_ancestor ? transmat(model) :

    # Recursion
    for indices ∈ grid, s ∈ state_ids
        if all(indices .+ 1 .>= s)
            state = values[s]
            @. obs_ind = ifelse(state == 1, indices, end_corner)
            curr_ind = 1 .+ indices
            prev_ind = curr_ind .- state
            @views α[s, curr_ind...] = emission_lps[obs_ind...] +
                                       logsumexp([A[:, s] .+ α[:, prev_ind...]])
        end
    end

    logp = logsumexp([A[:, END_INDEX] .+ α[:, end_corner]])

    return logp
end
