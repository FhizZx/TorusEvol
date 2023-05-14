using Random
import Base: rand

# ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄
# process acting over chains with unknown alignment
# hence the alignment is marginalised over using the forward algorithm
struct ChainProcess
    ξ::MixtureProductProcess # site level process
    τ::TKF92 # alignment model
end

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
end

function logpdf(d::ChainJointDistribution, (X, Y)::Tuple{AbstractChain, AbstractChain})
    α = Array{Float64}(undef, X.N + 1, Y.N + 1, num_states(d.τ))
    logpdf!(α, d, (X, Y))
end

function logpdf!(α::AbstractArray{<:Real}, d::ChainJointDistribution,
                 (X, Y)::Tuple{AbstractChain, AbstractChain})
    B = fulljointlogpdf(d.ξ, t, X, Y)
    forward!(α, d.τ, B)
end

#todo - saturday
# function rand(d::ChainJointDistribution)
#     # M = # sample pair alignment
#     rand(AlignedChainJointDistribution(d.ξ, d.τ, d.t))
#     (X, Y)
# end

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
