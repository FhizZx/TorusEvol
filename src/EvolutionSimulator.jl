using OffsetArrays
using BioStructures
# evolution simulation - sample descendant from ancestor

sequence(chain) = aa_to_id.(collect(string(LongAA(chain, standardselector))))[2:(end-1)]
angles(chain) = copy(transpose(hcat(filter(x -> !isnan(x), phiangles(chain))[1:(end-1)],
                     filter(x -> !isnan(x), psiangles(chain))[2:end])))


# A = transition matrix
# n = length of ancestor

# use the forward algorithm to compute
# α[i, M/I/D | hmm] until i = n, then backtrack through the matrix
# to generate dominoes
function sample_alignment_from_ancestor(A, n, pii)
    states = 1:4
    α = OffsetArray{Real}(undef, -1:n, states)
    α = fill!(α, -Inf)

    # Initial state
    α[0, START] = 0

    # Recursion
    for i ∈ 0:n
        α[i, MATCH] = logsumexp([A[q, MATCH] + α[i - 1, q] for q in states])

        α[i, DELETE] = logsumexp([A[q, DELETE] + α[i - 1, q] for q in states])

        α[i, INSERT] = logsumexp([A[q, INSERT] + α[i, q] for q in states])
    end

    # Backtrack sampling
    logp = logsumexp([α[n, q] + A[q, END] for q in states])


    s = rand(Categorical(exp.([α[n, q] + A[q, END] - logp for q in states])))
    i = n

    alignment = []
    while s != START
        if s == INSERT
            n_ins = 1 + rand(Geometric(1-pii))
            append!(alignment, fill(INSERT, n_ins))
        else
            push!(alignment, s)
        end
        new_i = s ∈ [MATCH, DELETE] ? (i-1) : i
        lpS = A[START, s] + α[new_i, START] - α[i, s]
        lpM = A[MATCH, s] + α[new_i, MATCH] - α[i, s]
        lpD = A[DELETE, s] + α[new_i, DELETE] - α[i, s]
        lpI = A[INSERT, s] + α[new_i, INSERT] - α[i, s]
        i = new_i
        s = rand(Categorical(exp.([lpS, lpM, lpD, lpI])))
    end
    return reverse(alignment)
end

function sample_descendant(X, alignment, subprocess, evolprocess, t)
    indicesX = findall(∈([MATCH, DELETE]), alignment)
    indicesY = findall(∈([MATCH, INSERT]), alignment)

    alignmentX = alignment[indicesX]
    alignmentY = alignment[indicesY]

    n = length(alignmentY)
    Y = Array{Real}(undef, 3, n)

    indicesIY = findall(==(INSERT), alignmentY)
    indicesMX = findall(==(MATCH), alignmentX)
    indicesMY = findall(==(MATCH), alignmentY)

    if length(indicesIY) > 0
        Y[1, indicesIY] .= rand(statdist(subprocess), length(indicesIY))
        Y[2:3, indicesIY] .= rand(statdist(evolprocess), length(indicesIY))
    end
    if length(indicesMY) > 0
        Y[1, indicesMY] .= rand.(transdist.(Ref(subprocess), Ref(t), vec(X[1, indicesMX])))
        Y[2:3, indicesMY] .= hcat(rand.(transdist.(Ref(evolprocess), Ref(t), eachcol(X[2:3, indicesMX])))...)
    end

    filled_alignment = Array{Union{Float64, Char}}(undef, 6, length(alignment))
    filled_alignment = fill!(filled_alignment, '-')
    filled_alignment[1, indicesX] = id_to_aa.(X[1, :])
    filled_alignment[2:3, indicesX] .= X[2:3, :]
    filled_alignment[4, indicesY] = id_to_aa.(Y[1, :])
    filled_alignment[5:6, indicesY] .= Y[2:3, :]

    return Y, filled_alignment
end


function simulator(t; λ = 0.0001, μ = 0.000101, r = 0.7, γ = 0.1)
    chain = read("data/1A3N.pdb", PDB)["A"]

    X = vcat(reshape(sequence(chain), 1, :), angles(chain))

    subprocess = WAG_SubstitutionProcess
    diff = WrappedDiffusion([0.0, 1.0], 1.0, 1.0, 4.0, 1.0, 1.0)
    evolprocess = jumping(diff, γ)

    mat, pii = TKF92TransitionMatrix(λ, μ, r, t)
    alignment = sample_alignment_from_ancestor(mat, size(X, 2), pii)
    Y, filled_alignment = sample_descendant(X, alignment, subprocess, evolprocess, t)

    return filled_alignment
end
