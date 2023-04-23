using Distributions, DistributionsAD
using LinearAlgebra
using Memoization


struct ObservedData
    data::AbstractVector{AbstractArray}
    N::Integer
end

function ObservedData(feats...)
    data = collect(feats)
    N = size(data[1], 2)
    @assert all(N .== size.(data, Ref(2))) "Dimensions of feature vectors don't match"
    return ObservedData(data, N)
end

data(x::ObservedData) = x.data
num_sites(x::ObservedData) = x.N
num_coords(x::ObservedData) = length(data(x))

# __________________________________________________________________________________________

# struct MarginalisedData
#     domains::AbstractVector{AbstractArray}
#     logprobs::AbstractMatrix{<:Real, 3}
# end

# num_sites(x::MarginalisedData) = size(logprobs)
# num_coordinates(x::MarginalisedData) = length(data(x))
# num_regimes(x::MarginalisedData) =

# __________________________________________________________________________________________
struct MixtureProductProcess
    weights::AbstractVector{<:Real}
    processes::AbstractMatrix{<:AbstractProcess}
end

weights(m::MixtureProductProcess) = m.weights
processes(m::MixtureProductProcess) = m.processes
num_coords(m::MixtureProductProcess) = size(m.processes, 1)
num_regimes(m::MixtureProductProcess) = size(m.processes, 2)

function jointlogpdf!(r::AbstractMatrix{<:Real},
                      m::MixtureProductProcess, t::Real,
                      X::ObservedData,
                      Y::ObservedData)
    E = num_regimes(m)
    r_e = similar(r)

    xs = data(X)
    ys = data(Y)

    for e ∈ 1:E
        @views procs = processes(m)[:, e]
        r_e .= weights(m)[e] .* sum(jointlogpdf!.(Ref(r_e), procs, Ref(t), xs, ys))
        r .= logsumexp.(r, r_e)
    end

    return r
end

#todo implement jointlogpdf & statlogpdf for ancestor and descendant

function statlogpdf!(r::AbstractVector{<:Real},
                     m::MixtureProductProcess,
                     X::ObservedData)
    E = num_regimes(m)
    r_e = similar(r)

    xs = data(X)

    for e ∈ 1:E
        @views procs = processes(m)[:, e]
        r_e .= weights(m)[e] .* sum(statlogpdf!.(Ref(r_e), procs, xs))
        r .= logsumexp.(r, r_e)
    end

    return r
end

function randjoint(m::MixtureProductProcess, t::Real, N::Integer)
    sites = rand(Categorical(weights(m)), N)
    procs = processes(m)(:, sites)
    C = num_coords(m)
    featsX = []
    featsY = []
    for c ∈ 1:C
        @views x, y = randjoint.(procs[c, 1:N], Ref(t))
        push!(featsX, x); push!(featsY, y)
    end
    return ObservedData(featsX...), ObservedData(featsY...)
end
