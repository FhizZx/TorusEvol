

function forward!(α::AbstractArray{<:Real}, model::TKF92,
                  jointlogpdf::AbstractArray{<:Real},
                  statlogpdfs::AbstractArray{AbstractVector{<:Real}};
                  known_ancestor=true)
    D = num_descendants(model)

    α = fill!(α, -Inf)
    # Initial state
    α[START, zeros(D)] = 0
    grid = Iterators.product(axes(α)[2:end]...)

    state_ids = proper_state_ids(model)
    # Recursion
    for indices ∈ grid, s ∈ state_ids
        domino = value(s)
        prev_indices = _prev_indices(indices, domino)
        α[s, indices...] = _lp(domino, indices, observations) + logsumexp([A[q, s] + α[q, (indices - domino)...]])
    end

    # Finally need to enter the END state to complete the chain
    logp = logsumexp([α[s, end_corner...] + A[s, END] for s in states])

    return logp
end
