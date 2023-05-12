using Distributions
using LogExpFunctions
using FastLogSumExp
using ForwardDiff

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

function _rand!(rng::AbstractRNG, d::PairDataHMM, A::AbstractMatrix{<:Real})
    A .= fill(42.0, d.N_X+1, d.N_Y+1)
    # @info "bad"
    return A
end

# lℙ[X, Y | params]
function _logpdf(d::PairDataHMM, emission_lps::AbstractMatrix{<:Real})
    #@info "good"
    α = d.α
    model = d.model
    logp = forward!(α, model, emission_lps)
    return logp
end

# α[state, indices] = lℙ[joint data up to indices, state = last state in HMM]
function forward!(α::AbstractArray{<:Real}, model::TKF92,
                  emission_lps::AbstractArray{<:Real})
    K = num_known_nodes(model)

    α = fill!(α, -Inf)
    # Initial state
    α[ones(Int, K)..., START_INDEX] = 0
    axs = [a .- 1 for a ∈ axes(emission_lps)]
    grid = map(collect, collect(Iterators.product(axs...))) # 0:N_X x 0:N_Y
    end_corner = size(emission_lps) # N_X + 1 by N_Y + 1

    A = transmat(model)
    # Recursion

    tape = Array{Real}(undef, num_states(model))
    tape .= -Inf

    curr_αind = ones(Int, K)
    prev_αind = ones(Int, K)
    for indices ∈ grid, s ∈ reverse(proper_state_ids(model))
        state = state_values(model)[s]
        curr_αind .= 1 .+ indices
        prev_αind .= curr_αind .- state
        if all(prev_αind .>= 1)
            obs_ind = @. ifelse(state == 1, indices, end_corner)
            tape .= A[:, s] .+ α[prev_αind..., :]
            state_lp = logsumexp(tape)
            α[curr_αind..., s] = emission_lps[obs_ind...] + state_lp
        end
    end

    logp = logsumexp(A[:, END_INDEX] .+ α[end_corner..., :])

    return logp
end


function backward_sampling(α::AbstractArray{<:Real}, model::TKF92)

    end_corner = size(α)[1:end-1] # N_X + 1 by N_Y + 1
    A = transmat(model)

    # the log probability of doing a backstep to a state from current state
    lps = Array{Real}(undef, num_states(model))

    # first, step back from the END state
    lps .= A[:, END_INDEX] .+ α[end_corner..., :]
    lps .-= logsumexp(lps)

    s = rand(Categorical(exp.(lps)))
    curr_αind = end_corner
    align_cols = Domino[]

    # keep doing back steps until the START state is reached
    while s != START_INDEX
        col = state_align_cols(model)[s]
        state = state_values(model)[s]
        # if ancestor unknown, cannot keep track of index
        if !model.known_ancestor && s == no_survivors_ancestor_id(model)
            l = 1 + rand(Geometric(1 - model.full_del_rate))
            append!(align_cols, fill(col, l))
        else
            push!(align_cols, col)
        end

        # probabilistic backtracking through α
        prev_αind = curr_αind .- state
        lps .= A[:, s] .+ α[prev_αind..., :] .- α[curr_αind..., s]
        lps .-= logsumexp(lps)
        curr_αind = prev_αind

        s = rand(Categorical(exp.(lps)))
    end
    return Alignment(hcat(reverse(align_cols)...))
end

# Complete alignment over descendants by sampling aligned ancestor
# α[state, m+1] = lℙ[data in alignment[1:m], state = last state in HMM]
function forward_anc(alignment::Alignment, model::TKF92,
                     emission_lps::AbstractVector{<:Real})
    M = length(alignment)
    @assert model.known_ancestor == false

    α = Array{Real}(undef, M+1, num_states(model))
    α .= -Inf
    # Initial state
    α[1, START_INDEX] = 0
    grid = 0:M
    end_corner = M+1

    A = transmat(model)
    # Recursion

    tape = Array{Real}(undef, num_states(model))
    tape .= -Inf

    desc_values = align_state_desc_values[model.D]
    anc_state_id = no_survivors_ancestor_id(model)
    for m ∈ grid, s ∈ reverse(proper_state_ids(model))
        curr_αind = 1 + m
        # if current state only includes ancestor, we don't advance in the alignment
        prev_αind = (s == anc_state_id) ? curr_αind : curr_αind - 1
        if prev_αind >= 1 && (s == anc_state_id || desc_values[s] == alignment[m])
            tape .= A[:, s] .+ α[prev_αind, :]
            # # only allow transitioning from states with desc_value == alignment[m-1]
            # # OR from the anc_state
            # # a bit hacky
            # tmp = tape[anc_state_id]
            # if m > 1
            #     tape[desc_values .!= Ref(alignment[m-1])] .= -Inf
            # # if m == 1, need to transition from the start or from anc state
            # else
            #     tape[state_ids(model) .!= START_INDEX] .= -Inf
            # end
            # tape[anc_state_id] = tmp

            state_lp = logsumexp(tape)

            # if ancestor only state, there was no descendant emission
            emission_lp = (s == anc_state_id) ? 0 : emission_lps[m]

            α[curr_αind..., s] = emission_lp + state_lp
        end
    end
    return α
end

function backward_sampling_anc(α::AbstractMatrix{<:Real}, model::TKF92)
    M = size(α, 1) - 1
    @assert model.known_ancestor == false
    end_corner = M+1
    A = transmat(model)

    # the log probability of doing a backstep to a state from current state
    lps = Array{Real}(undef, num_states(model))

    # first, step back from the END state
    lps .= A[:, END_INDEX] .+ α[end_corner, :]
    lps .-= logsumexp(lps)

    s = rand(Categorical(exp.(lps)))
    curr_αind = end_corner
    align_cols = Domino[]
    anc_state_id = no_survivors_ancestor_id(model)
    # keep doing back steps until the START state is reached
    while s != START_INDEX
        col = state_align_cols(model)[s]
        # as ancestor unknown, cannot keep track of index
        if s == anc_state_id
            l = 1 + rand(Geometric(1 - model.full_del_rate))
            append!(align_cols, fill(col, l))
        else
            push!(align_cols, col)
        end

        # probabilistic backtracking through α
        prev_αind = (s == anc_state_id) ? curr_αind : (curr_αind - 1)
        lps .= A[:, s] .+ α[prev_αind, :] .- α[curr_αind, s]
        lps .-= logsumexp(lps)
        curr_αind = prev_αind

        s = rand(Categorical(exp.(lps)))
    end
    return Alignment(hcat(reverse(align_cols)...))
end



#todo - sample full alignment from partial alignment
