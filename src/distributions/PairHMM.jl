using OffsetArrays
using LogExpFunctions

AbstractProtein{T} =

@enum AlignStates START=1 MATCH=2 DELETE=3 INSERT=4 END=5
NUM_ALIGN_STATES = 5

struct PairHMM{Monomer}
    A::Matrix{Real}                 # alignment transition matrix
    B::AbstractProcess{Monomer}
    t::Real # time
end

function _logpdf(hmm::PairHMM{Monomer},
                 data::Tuple{AbstractVector{Monomer}, AbstractVector{Monomer}}) where T <: Real
    P_a, P_b = data
    n_a = size(P_a)
    n_b = size(P_b)

    A = hmm.A
    B = hmm.B
    statdist = statdist(B)

    # Apply the forward algorithm to compute p(P_a, P_b | hmm)
    # Complexity O(num_states * n_a * n_b)

    # α[i, j, s] = log p(P_a[0:i], P_b[0:j], last_state = s)
    states = 1:4
    α = OffsetArray{T}(-Inf, -1:n_a, -1:n_b, states)

    # Initial state
    α[0, 0, START] = 0

    # Recursion
    for i ∈ 0:n_a, j ∈ 0:n_b
        if i == j == 0
            continue
        end
        α[i, j, MATCH] = logpdf(statdist, P_a[i]) +
                         logpdf(transdist(B, t, P_a[i]), P_b[i]) +
                         logsumexp([A[q, M] + α[i - 1, j - 1, q] for q in states])

        α[i, j, DELETE] = logpdf(statdist, P_a[i]) +
                          logsumexp([A[q, D] + α[i - 1, j, q] for q in states])

        α[i, j, INSERT] = logpdf(statdist, P_b[j]) +
                          logsumexp([A[q, I] + α[i, j - 1, q] for q in states])

    end

    # Finally need to enter the END state to complete the chain
    logp = logsumexp([α[n_a, n_b, s] + A[s, END] for s in states])
    return logp
end
