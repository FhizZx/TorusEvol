

function forward!(α, A, observations)
    α = fill!(α, -Inf)
    # Initial state
    α[2, 2, START] = 0

    # Recursion
    for indices ∈ grid, s ∈ states
        domino = onehot[s]
        prev_indices = _prev_indices(indices, domino)
        α[s, indices...] = _lp(domino, indices, observations) + logsumexp([A[q, s] + α[q, (indices - domino)...]])
    end

    # Finally need to enter the END state to complete the chain
    logp = logsumexp([α[s, end_corner...] + A[s, END] for s in states])

    return logp
end
