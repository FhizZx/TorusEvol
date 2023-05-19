using Random
using Distributions
import Base: rand

# ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄
# process acting over chains with unknown alignment
# hence the alignment is marginalised over using the forward algorithm
# struct ChainProcess
#     ξ::MixtureProductProcess # site level process
#     τ::TKF92 # alignment model
# end

#todo saturday
# # ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄
# # process acting over chains with known alignment
# struct AlignedChainProcess
#     ξ::MixtureProductProcess # site level process
#     τ::TKF92 # alignment model
# end

# ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄
# Chain and aligned chain distributions

#todo - saturday
# # __________________________________________________________________________________________
# # Stationary distribution of chains
# struct ChainStationaryDistribution
#     ξ::MixtureProductProcess # site level process
#     τ::TKF92 # alignment model
# end

# function Base.rand(d::ChainStationaryDistribution)
#     # sample length from τ lengh distribution
#     #
# end

# function logpdf(d::ChainStationaryDistribution, X::AbstractChain)

const ChainPair = Tuple{<:AbstractChain, <:AbstractChain}

# __________________________________________________________________________________________
# Distribution over two chains related by time t
struct ChainJointDistribution
    ξ::MixtureProductProcess # site level process
    τ::TKF92 # alignment model
    t::Real
    function ChainJointDistribution(ξ::MixtureProductProcess,
                                    τ::TKF92)
        @assert num_descendants(τ) ∈ [1, 2]
        if num_descendants(τ) == 1
            @assert τ.known_ancestor == true
            t = τ.ts[1]
        elseif num_descendants(τ) == 2
            @assert τ.known_ancestor == false
            t = sum(τ.ts)
        end
        new(ξ, τ, t)
    end
end

function get_α(τ::TKF92, chains::AbstractVector{<:AbstractChain})
    K = num_known_nodes(τ)
    @assert K == length(chains)
    lengths = num_sites.(chains) .+ 1
    α = Array{Float64}(undef, lengths..., num_states(τ))
    α .= -Inf
    return α
end

function get_α(τ::TKF92, (X, Y)::ChainPair)
    @assert num_known_nodes(τ) == 2
    α = Array{Float64}(undef, num_sites(X)+1, num_sites(Y)+1, num_states(τ))
    α .= -Inf
    return α
end

function get_B((X, Y)::ChainPair)
    B = Array{Float64}(undef, num_sites(X)+1, num_sites(Y)+1)
    B .= -Inf
    return B
end

function logpdf(d::ChainJointDistribution, (X, Y)::ChainPair)
    α = get_α(d.τ, [X, Y])
    logpdfα!(α, d, (X, Y))
end

function logpdfα!(α::AbstractArray{<:Real}, d::ChainJointDistribution,
                 (X, Y)::ChainPair)
    @assert all(size(α) .== (num_sites(X)+1, num_sites(Y)+1, num_states(d.τ)))
    B = fulljointlogpdf(d.ξ, d.t, X, Y)
    forward!(α, d.τ, B)
end

function logpdfαB!(α::AbstractArray{<:Real},
                   B::AbstractArray{<:Real},
                   d::ChainJointDistribution,
                   (X, Y)::ChainPair)
    @assert all(size(α) .== (num_sites(X)+1, num_sites(Y)+1, num_states(d.τ)))
    @assert all(size(B) .== (num_sites(X)+1, num_sites(Y)+1))
    fulljointlogpdf!(B, d.ξ, d.t, X, Y)
    forward!(α, d.τ, B)
end


function rand(d::ChainJointDistribution)
    ξ = d.ξ
    τ = d.τ
    t = d.t

    # sample pair alignment
    M_XY = Alignment(rand(AlignmentDistribution(τ)), τ)
    alignmentX = slice(M_XY, mask(M_XY, [[1], [0,1]]))
    XY_maskX = mask(alignmentX, [[1], [1]])
    X_maskX = mask(alignmentX, [[1], [0]])
    alignmentY = slice(M_XY, mask(M_XY, [[0,1], [1]]))
    XY_maskY = mask(alignmentY, [[1], [1]])
    Y_maskY = mask(alignmentY, [[0], [1]])

    N_X = length(alignmentX)
    N_Y = length(alignmentY)

    # Sample internal coordinates
    stat_featsX = randstat(ξ, count(X_maskX))
    stat_featsY = randstat(ξ, count(Y_maskY))
    @assert count(XY_maskX) == count(XY_maskY)
    match_featsX, match_featsY = randjoint(ξ, t, count(XY_maskX))

    C = num_coords(ξ)
    featsX = Vector{AbstractArray{Real}}(undef, 0)
    featsY = Vector{AbstractArray{Real}}(undef, 0)
    for c ∈ 1:C
        d = length(processes(ξ)[c, 1])
        x = Array{eltype(processes(ξ)[c, 1])}(undef, d, N_X); y = similar(x, d, N_Y)

        x[:, XY_maskX] .= match_featsX[c]
        x[:, X_maskX] .= stat_featsX[c]

        y[:, XY_maskY] .= match_featsY[c]
        y[:, Y_maskY] .= stat_featsY[c]

        push!(featsX, x); push!(featsY, y)
    end
    X = ObservedChain(featsX)
    Y = ObservedChain(featsY)
    (X, Y)
end

# __________________________________________________________________________________________
# Distribution over processes started at ancestor X in time t
struct ChainTransitionDistribution
    ξ::MixtureProductProcess # site level process
    τ::TKF92 # alignment model
    X::AbstractChain
    t::Real
    function ChainTransitionDistribution(ξ::MixtureProductProcess,
                                         τ::TKF92,
                                         X::AbstractChain)
        @assert num_descendants(τ) == 1
        @assert τ.known_ancestor == true
        t = τ.ts[1]
        new(ξ, τ, X, t)
    end
end

function logpdf(d::ChainTransitionDistribution,
                Y::AbstractChain)
    X = d.X
    α = Array{Float64}(undef, X.N + 1, Y.N + 1, num_states(d.τ))
    logpdfα!(α, d, Y)
end

function logpdfα!(α::AbstractArray{<:Real}, d::ChainTransitionDistribution,
                  Y::AbstractChain)
    X = d.X
    t = d.t
    @assert all(size(α) .== (num_sites(X)+1, num_sites(Y)+1, num_states(d.τ)))
    B = get_B((X, Y))
    fulltranslogpdf!(B, d.ξ, t, X, Y)
    forward!(α, d.τ, B)
end

function rand(d::ChainTransitionDistribution)
    ξ = d.ξ
    τ = d.τ
    t = d.t
    X = d.X

    # sample pair alignment
    N = num_sites(X)
    model = TKF92([t], τ.λ, τ.μ, τ.r; known_ancestor=false)
    M_X = Alignment(ones(Int, 1, N))
    B = zeros(Real, N+1)
    α = forward_anc(M_X, model, B)

    a = M_X; nn = 0
    while nn == 0
        a = backward_sampling_anc(α, model)
        nn = sequence_lengths(a)[1]
    end

    M_XY = Alignment(data(a)[[2,1], :])

    alignmentX = slice(M_XY, mask(M_XY, [[1], [0,1]]))
    XY_maskX = mask(alignmentX, [[1], [1]])

    alignmentY = slice(M_XY, mask(M_XY, [[0,1], [1]]))
    XY_maskY = mask(alignmentY, [[1], [1]])
    Y_maskY = mask(alignmentY, [[0], [1]])

    N_Y = length(alignmentY)

    # Sample internal coordinates
    stat_featsY = randstat(ξ, count(Y_maskY))
    @assert count(XY_maskX) == count(XY_maskY)

    match_featsX = data(slice(X, XY_maskX, :))
    match_featsY = randtrans(ξ, t, match_featsX)

    C = num_coords(ξ)
    featsY = Vector{AbstractArray{Real}}(undef, 0)
    for c ∈ 1:C
        d = length(processes(ξ)[c, 1])
        y = Array{eltype(processes(ξ)[c, 1])}(undef, d, N_Y)

        y[:, XY_maskY] .= match_featsY[c]
        y[:, Y_maskY] .= stat_featsY[c]

        push!(featsY, y)
    end
    Y = ObservedChain(featsY)
    Y
end
