

# struct ConditionedPairAlignment <: DiscreteMatrixDistribution
#     τ::TKF92 # alignment model
#     α::AbstractArray{<:Real}
# end

# size(d::PairDataHMM) = (2, size(d.α, 1) + size(d.α, 2) - 1)

# #todo - tuesday
# # # lℙ[M_XY | X, Y]
# # function _logpdf(d::ConditionedPairAlignment, M::AbstractMatrix{<:Real})
# #     α = d.α
# #     model = d.model
# #     logp = forward!(α, model, emission_lps)
# #     return logp
# # end

# function _rand!(rng::AbstractRNG, d::ConditionedPairAlignment,
#                 M::AbstractMatrix{<:Integer})
#     alignment = backward_sampling(rng, d.α, d.τ)
#     M[:, length(alignment)] .= data(alignment)
#     M[:, length(alignment) + 1] .= 0
#     return A
# end

# function backward_sampling(rng::AbstractRNG,
#                            α::AbstractArray{<:Real}, model::TKF92)

#     end_corner = size(α)[1:end-1] # N_X + 1 by N_Y + 1
#     A = transmat(model)

#     # the log probability of doing a backstep to a state from current state
#     lps = similar(α, num_states(model))

#     # first, step back from the END state
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
