struct EvolHMM <: ContinuousMatrixDistribution
    M::Integer                      # Total alignment length M = sum(lengths)
    lengths::AbstractVector{Int}
    𝚯::AbstractProcess
    A::AbstractMatrix{Real}
    t::Real

    α::Array{Real}
end

function EvolHMM(n, m, A, sub, diff, t)
    α = Array{Real}(undef, n+2, m+2, 4)
    return PairHMM(n, m, A, sub, diff, t, α)
end

function _logpdf(d::EvolHMM, x::AbstractMatrix{<:Real})

    return forward!(d.α, x[1:3, 1:d.n], x[4:6, 1:d.m], d.A, d.sub, d.diff, d.t)
end

size(d::PairHMM) = (6, max(d.n, d.m))

length(d::PairHMM) = product(size(d))





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
#todo compute gradient of forward w.r.t A and observations

function backward_sampling(α, A)
    n_x, n_y = size(α) .- 2
    states = 1:4
    logp = logsumexp([α[n_x+2, n_y+2, q] + A[q, END] for q in states])

    s = rand(Categorical(exp.([α[n_x+2, n_y+2, q] + A[q, END] - logp for q in states])))
    indices =

    alignment = []
    while s != START
        domino = onehot[s]
        push!(alignment, domino)

        new_indices = indices - onehot[s]

        lps = [A[q, s] + α[q, new_indices...] - α[s, indices...] for q in states]

        indices = new_indices
        s = rand(Categorical(normalize(exp.([lpS, lpM, lpD, lpI]), 1)))
    end
    return reverse(alignment)
end
