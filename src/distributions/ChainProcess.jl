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


# __________________________________________________________________________________________
# Distribution over two chains related by time t
struct ChainJointDistribution
    ξ::MixtureProductProcess # site level process
    τ::TKF92 # alignment model
    t::Real
    function ChainJointDistribution(ξ::MixtureProductProcess,
                                    τ::TKF92)
        @assert num_descendants(τ) == 1
        @assert τ.known_ancestor == true
        t = τ.ts[1]
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

function logpdf(d::ChainJointDistribution, (X, Y)::Tuple{<:AbstractChain, <:AbstractChain})
    α = get_α(d.τ, [X, Y])
    logpdfα!(α, d, (X, Y))
end

function logpdfα!(α::AbstractArray{<:Real}, d::ChainJointDistribution,
                 (X, Y)::Tuple{<:AbstractChain, <:AbstractChain})
    B = fulljointlogpdf(d.ξ, d.t, X, Y)
    forward!(α, d.τ, B)
end


function rand(d::ChainJointDistribution)
    ξ = d.ξ
    τ = d.τ
    t = d.t

    # sample pair alignment
    #TODO - better max length
    M_XY = Alignment(rand(AlignmentDistribution(τ, 1000)))
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

#todo - tuesday
# # __________________________________________________________________________________________
# # Distribution over processes descending from ancestor X in time t
# struct ChainTransitionDistribution
#     ξ::MixtureProductProcess # site level process
#     τ::TKF92 # alignment model
#     X::AbstractChain
#     t::Real
# end

# function logpdf(d::ChainTransitionDistribution,
#                 Y::AbstractChain)
#     X = d.X
#     α = Array{Float64}(undef, X.N + 1, Y.N + 1, num_states(d.τ))
#     logpdf!(α, d, (X, Y))
# end

# function logpdf!(α::AbstractArray{<:Real}, d::JointChainDistribution,
#                  Y::AbstractChain)
#     B = fulltranslogpdf(d.ξ, t, X, Y)
#     forward!(α, d.τ, B)
# end
