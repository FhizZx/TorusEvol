using Distributions


# ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄
# Distribution over unconditioned alignments
# generated by a pairhmm model

# ℙ[M | τ]
struct AlignmentDistribution <: DiscreteMatrixDistribution
    τ::TKF92  # alignment model
    max_length::Integer
    function AlignmentDistribution(τ::TKF92; max_length=3000)
         new(τ, max_length)
    end
end

size(d::AlignmentDistribution) = (num_descendants(d.τ)+1, d.max_length)

function alignment_to_tkf92state_ids(M::Alignment, τ::TKF92)
    s = START_INDEX
    D = num_descendants(τ)
    flags = align_state_flags[D]
    state_ids = Int[]
    for col ∈ M
        anc_value = col[1]
        desc_values = col[2:end]
        if anc_value == 1
            # scenario changes
            state = gen_ancestor_state(desc_values)
            flags .= desc_values
        else
            # scenario hasn't changed
            state = gen_descendant_state(desc_values, flags)
        end
        q = state_ids_dict(τ)[state]
        s = q
        push!(state_ids, s)
    end
    return state_ids
end

function tkf92state_ids_to_alignment(ids::AbstractVector{<:Integer}, τ::TKF92)
    return Alignment(hcat(state_align_cols(τ)[ids]...))
end



function _logpdf(d::AlignmentDistribution, data::AbstractMatrix{<:Integer})
    M = Alignment(data)
    τ = d.τ
    A = transmat(τ)

    s = START_INDEX
    res = 0
    for q ∈ alignment_to_tkf92state_ids(M, τ)
        res += A[s, q]
        s = q
    end
    res += A[s, END_INDEX]
    return res
end

function _rand!(rng::AbstractRNG, d::AlignmentDistribution,
                M::AbstractMatrix{<:Integer})
    model = d.τ
    max_length = d.max_length
    P = exp.(transmat(model))

    length = max_length + 1
    seq_lengths = fill(0, size(d, 1))

    # reject samples which exceed match length or ones which have null sequences
    while length > max_length || any(seq_lengths .< 3)
        i = 1
        s = START_INDEX
        while s != END_INDEX
            s = rand(rng, Categorical(vec(P[s, :])))
            col = state_align_cols(model)[s]

            if !model.known_ancestor && s == no_survivors_ancestor_id(model)
                l = 1 + rand(rng, Geometric(1 - model.full_del_rate))
                M[:, i:(i+l-1)] .= col
                i += l
            elseif s != END_INDEX
                M[:, i] .= col
                i += 1
            end
        end
        length = i-1
        seq_lengths = [count(M[j, 1:length] .== 1) for j ∈ eachindex(seq_lengths)]
    end
    M[:, (length+1):end] .= 0

    return M
end


# ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄
# ℙ[M | X, Y, τ]
struct ConditionedPairAlignmentDistribution <: DiscreteMatrixDistribution
    τ::TKF92 # alignment model
    α::AbstractArray{<:Real}
    unconditioned::AlignmentDistribution
    lp_data::Real
end

size(d::ConditionedPairAlignmentDistribution) = (num_descendants(d.τ)+1, d.max_length)

# lℙ[M_XY | X, Y]
# function _logpdf(d::ConditionedPairAlignment, M::AbstractMatrix{<:Real})
#     logpdf(d.unconditioned, M) + logpdf(X, Y |) - d.lp_data

#     α = d.α
#     model = d.model
#     logp = forward!(α, model, emission_lps)
#     return logp
# end

function _rand!(rng::AbstractRNG, d::ConditionedPairAlignmentDistribution,
                M::AbstractMatrix{<:Integer})
    alignment = _backward_sampling(rng, d.α, d.τ)
    M[:, length(alignment)] .= data(alignment)
    M[:, length(alignment) + 1] .= 0
    return M
end

# function _backward_logpdf(α::AbstractArray{<:Real}, model::TKF92, M::Alignment)
#     end_corner = size(α)[1:end-1] # N_X + 1 by N_Y + 1
#     A = transmat(model)

#     res = 0

#     # first, step back from the END state
#     res +=
#     lps .= A[:, END_INDEX] .+ α[end_corner..., :]
#     lps .-= logsumexp(lps)

#     s = rand(rng, Categorical(exp.(lps)))
#     curr_αind = end_corner
#     align_cols = Domino[]

#     # keep doing back steps until the START state is reached
#     while s != START_INDEX
#         col = state_align_cols(model)[s]
#         state = state_values(model)[s]
#         # if ancestor unknown, cannot keep track of index
#         if !model.known_ancestor && s == no_survivors_ancestor_id(model)
#             l = 1 + rand(rng, Geometric(1 - model.full_del_rate))
#             append!(align_cols, fill(col, l))
#         else
#             push!(align_cols, col)
#         end

#         # probabilistic backtracking through α
#         prev_αind = curr_αind .- state
#         lps .= A[:, s] .+ α[prev_αind..., :] .- α[curr_αind..., s]
#         lps .-= logsumexp(lps)
#         curr_αind = prev_αind

#         s = rand(rng, Categorical(exp.(lps)))
#     end
#     return Alignment(hcat(reverse(align_cols)...))
# end

function _backward_sampling(rng::AbstractRNG,
                            α::AbstractArray{<:Real}, model::TKF92)

    end_corner = size(α)[1:end-1] # N_X + 1 by N_Y + 1
    A = transmat(model)

    # the log probability of doing a backstep to a state from current state
    lps = similar(α, num_states(model))

    # first, step back from the END state
    lps .= A[:, END_INDEX] .+ α[end_corner..., :]
    lps .-= logsumexp(lps)

    s = rand(rng, Categorical(exp.(lps)))
    curr_αind = end_corner
    align_cols = Domino[]

    # keep doing back steps until the START state is reached
    while s != START_INDEX
        col = state_align_cols(model)[s]
        state = state_values(model)[s]
        # if ancestor unknown, cannot keep track of index
        if !model.known_ancestor && s == no_survivors_ancestor_id(model)
            l = 1 + rand(rng, Geometric(1 - model.full_del_rate))
            append!(align_cols, fill(col, l))
        else
            push!(align_cols, col)
        end

        # probabilistic backtracking through α
        prev_αind = curr_αind .- state
        lps .= A[:, s] .+ α[prev_αind..., :] .- α[curr_αind..., s]
        lps .-= logsumexp(lps)
        curr_αind = prev_αind

        s = rand(rng, Categorical(exp.(lps)))
    end
    return Alignment(hcat(reverse(align_cols)...))
end
