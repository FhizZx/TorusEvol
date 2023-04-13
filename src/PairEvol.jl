using Turing, ReverseDiff
using BioAlignments

include("EvolutionSimulator.jl")
Turing.setadbackend(:reversediff)
Turing.setrdcache(true)

@model function PairEvol(data, n, m)

    # Sample time
    t ~ Exponential(1)

    sub = WAG_SubstitutionProcess

    # Sample structure parameters
    # todo more principles prior definition
    mean ~ filldist(Uniform(-π, π), 2)
    var ~ filldist(Exponential(1), 4)
    # α_corr ~ truncated(Normal(0, 1), -1, 1)
    γ ~ Beta(1, 1)

    Θ = (mean, var..., 0.0)
    diff = jumping(WrappedDiffusion(Θ...), γ)

    # Make substitution model that incorporates both sequence and structure
    # submodel = ProductProcess(seqmodel, structmodel)


    # Sample indel parameters
    λ ~ Exponential(1)
    seq_length ~ Exponential(0.01)
    μ = (seq_length+1) * λ
    r ~ Uniform(0,1)
    Λ = (λ, μ, r)
    A, _ = TKF92TransitionMatrix(Λ..., t)


    # Finally create the PairHMM model from the substitution and indel models
    α = Array{Real}(undef, n+2, m+2, 4)
    pairmodel = PairHMM(n, m, A, sub, diff, t, α)
    data ~ pairmodel

    params = (t, diff)
    return params
end

function sample_posterior(chainX, chainY, l, n_samples)
    X = vcat(reshape(sequence(chainX), 1, :), angles(chainX))
    X[isnan.(X)] .= 0.0
    Y = vcat(reshape(sequence(chainY), 1, :), angles(chainY))
    Y[isnan.(Y)] .= 0.0
    X = X[:, 1:l]
    Y = Y[:, 1:l]
    n = size(X, 2)
    m = size(Y, 2)
    data = Array{Float64}(undef, 6, max(n, m))
    data[1:3, 1:n] .= X
    data[4:6, 1:m] .= Y

    model = PairEvol(data, n, m)
    alg = HMC(0.05, 10, :t, :mean, :var, :γ, :r, :λ, :seq_length)

    chain = sample(model, alg, n_samples)
    return chain
end



struct PairHMM <: ContinuousMatrixDistribution
    n::Int
    m::Int
    A::Matrix{Real}
    sub::CTMC
    diff::JumpingProcess
    t::Real
    α::Array{Real}
end

function PairHMM(n, m, A, sub, diff, t)
    α = Array{Real}(undef, n+2, m+2, 4)
    return PairHMM(n, m, A, sub, diff, t, α)
end

function _logpdf(d::PairHMM, x::AbstractMatrix{<:Real})
    return forward!(d.α, x[1:3, 1:d.n], x[4:6, 1:d.m], d.A, d.sub, d.diff, d.t)
end

size(d::PairHMM) = (6, max(d.n, d.m))

length(d::PairHMM) = product(size(d))





function forward!(α, X, Y, A, sub, diff, t)
    n_x = size(X, 2)
    n_y = size(Y, 2)
    α = fill!(α, -Inf)
    states = 1:4

    # Initial state
    α[2, 2, START] = 0

    #todo implement better categorical distribution
    #can optimize this more by not allocating statX/statY/trans
    statX = logpdf.(statdist(sub), X[1, :]) .+ logpdf(statdist(diff), X[2:3, :])
    statY = logpdf.(statdist(sub), Y[1, :]) .+ logpdf(statdist(diff), Y[2:3, :])
    trans = Array{Real}(undef, n_x, n_y)
    for j ∈ 1:n_y
        trans[:, j] .= logpdf.(transdist(sub, t, Y[1, j]), X[1, :]) .+
                       logpdf(transdist(diff, t, Y[2:3, j]), X[2:3, :])
    end

    # Recursion
    for i ∈ 0:n_x, j ∈ 0:n_y
        if i > 0 && j > 0
            α[i+2, j+2, MATCH] = statX[i] + trans[i, j] +
                             logsumexp([A[q, MATCH] + α[i+2 - 1, j+2 - 1, q] for q in states])
        end

        if i > 0
            α[i+2, j+2, DELETE] = statX[i] +
                              logsumexp([A[q, DELETE] + α[i+2 - 1, j+2, q] for q in states])
        end

        if j > 0
            α[i+2, j+2, INSERT] = statY[j] +
                              logsumexp([A[q, INSERT] + α[i+2, j+2 - 1, q] for q in states])
        end

    end

    # Finally need to enter the END state to complete the chain
    logp = logsumexp([α[n_x+2, n_y+2, s] + A[s, END] for s in states])

    return logp
end

function backward_sampling(α, A)
    n_x, n_y = size(α) .- 2
    states = 1:4
    logp = logsumexp([α[n_x, n_y, q] + A[q, END] for q in states])
    s = rand(Categorical(exp.([α[n_x, n_y, q] + A[q, END] - logp for q in states])))

    i = n_x
    j = n_y

    alignment = []
    while s != START

        push!(alignment, s)

        new_i = s ∈ [MATCH, DELETE] ? (i-1) : i
        new_j = s ∈ [MATCH, INSERT] ? (j-1) : j

        lpS = A[START, s] + α[new_i, new_j, START] - α[i, j, s]
        lpM = A[MATCH, s] + α[new_i, new_j, MATCH] - α[i, j, s]
        lpD = A[DELETE, s] + α[new_i, new_j, DELETE] - α[i, j, s]
        lpI = A[INSERT, s] + α[new_i, new_j, INSERT] - α[i, j, s]

        i = new_i
        j = new_j
        s = rand(Categorical(normalize(exp.([lpS, lpM, lpD, lpI]), 1)))
    end
    return reverse(alignment)
end

function filled_alignment(alignment, X, Y)
    indicesX = findall(∈([MATCH, DELETE]), alignment)
    indicesY = findall(∈([MATCH, INSERT]), alignment)

    filled_alignment = Array{Union{Float64, Char}}(undef, 6, length(alignment))
    filled_alignment = fill!(filled_alignment, '-')
    filled_alignment[1, indicesX] = id_to_aa.(X[1, :])
    filled_alignment[2:3, indicesX] .= X[2:3, :]
    filled_alignment[4, indicesY] = id_to_aa.(Y[1, :])
    filled_alignment[5:6, indicesY] .= Y[2:3, :]

    return filled_alignment
end

function pair_align(chainX, chainY; t=0.1)
    sub = WAG_SubstitutionProcess

    helix = deg2rad.([-90.0, -30.0])
    diff = WrappedDiffusion(helix, 1.0, 1.0, 4.0, 1.0, 1.0)
    γ = 0.1
    diff = jumping(diff, γ)

    λ, μ, r = 0.0002, 0.00021, 0.9
    A, _ = TKF92TransitionMatrix(λ, μ, r, t)

    X = vcat(reshape(sequence(chainX), 1, :), angles(chainX))
    X[isnan.(X)] .= 0.0
    Y = vcat(reshape(sequence(chainY), 1, :), angles(chainY))
    Y[isnan.(Y)] .= 0.0


    println()
    println("Alignment generated by affine gap combinatorial model")
    scoremodel = AffineGapScoreModel(BLOSUM62, gap_open=-10, gap_extend=-1);
    aln = alignment(pairalign(GlobalAlignment(), LongAA(chainX, standardselector),
                    LongAA(chainY, standardselector), scoremodel))
    println(LongAA([x for (x, _) in aln]))
    println(LongAA([y for (_, y) in aln]))
    println()

    n_x = size(X, 2)
    n_y = size(Y, 2)
    states = 1:4

    α = OffsetArray{Real}(undef, -1:n_x, -1:n_y, states)
    forward!(α, X, Y, A, sub, diff, t)
    println("Alignments sampled from TKF92-WAG_WrappedDiff evol model")
    for _ in 1:10
        alignment = backward_sampling(α, A)

        filled = filled_alignment(alignment, X, Y)
        seqX, seqY = string(filled[1, :]...), string(filled[4, :]...)
        println(seqX)
        println(seqY)
        println()
    end
    return
end
