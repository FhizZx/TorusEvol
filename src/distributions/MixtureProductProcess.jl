using Distributions, DistributionsAD
using LinearAlgebra
using Memoization



# __________________________________________________________________________________________
struct MixtureProductProcess
    weights::AbstractVector{<:Real}
    processes::AbstractMatrix{<:AbstractProcess}
end

function MixtureProductProcess(weights, processes)
    @assert length(weights) == size(processes, 2) "The number of regimes is not consistent"
    return MixtureProductProcess(weights, processes)
end

weights(m::MixtureProductProcess) = m.weights
processes(m::MixtureProductProcess) = m.processes
num_coords(m::MixtureProductProcess) = size(m.processes, 1)
num_regimes(m::MixtureProductProcess) = size(m.processes, 2)

function fulllogpdf!(r::AbstractMatrix{<:Real},
                     p::MixtureProductProcess, t::Real,
                     X::ObservedData,
                     Y::ObservedData)
    n = num_sites(X)
    m = num_sites(Y)
    @views jointlogpdf!(r[1:n, 1:m], p, t, X, Y)
    @views statlogpdf!(r[1:n, m+1], p, X)
    @views statlogpdf!(r[n+1, 1:m], p, Y)
    return r
end

function jointlogpdf!(r::AbstractMatrix{<:Real},
                      m::MixtureProductProcess, t::Real,
                      X::ObservedData,
                      Y::ObservedData)
    E = num_regimes(m)
    @assert num_coords(X) == num_coords(Y)
    @assert num_coords(X) == num_coords(m) string(num_coords(X)) * " " * string(num_coords(m))
    C = num_coords(X)
    workspace = similar(r)
    xs = data(X)
    ys = data(Y)

    # Compute the contribution of each regime to the final logpdf
    r_e = similar(workspace)
    for e ∈ 1:E
        w = weights(m)[e]
        r_e .= log(w)
        for c ∈ 1:C
            jointlogpdf!(workspace, processes(m)[c, e] , t, xs[c], ys[c])
            r_e .= r_e .+ workspace
        end
        r .= logaddexp.(r, r_e)
    end

    return r
end

#todo implement jointlogpdf & statlogpdf for ancestor and descendant

function statlogpdf!(r::AbstractVector{<:Real},
                     m::MixtureProductProcess,
                     X::ObservedData)
    E = num_regimes(m)
    C = num_coords(X)
    workspace = similar(r)

    xs = data(X)

    # Compute the contribution of each regime to the final logpdf
    r_e = similar(r)
    for e ∈ 1:E
        w = weights(m)[e]
        r_e .= log(w)
        for c ∈ 1:C
            statlogpdf!(workspace, processes(m)[c, e], xs[c])
            r_e .= r_e .+ workspace
        end
        r .= logaddexp.(r, r_e)
    end

    return r
end

function randstat(m::MixtureProductProcess, N::Integer)
    sites = rand(Categorical(weights(m)), N)
    dists = statdist.(processes(m)[:, sites])
    C = num_coords(m)
    featsX = []
    for c ∈ 1:C
        @views x = rand.(dists[c, 1:N])
        push!(featsX, x)
    end
    return ObservedData(featsX...)
end

function randjoint(m::MixtureProductProcess, t::Real, N::Integer)
    sites = rand(Categorical(weights(m)), N)
    procs = processes(m)[:, sites]
    C = num_coords(m)
    featsX = []
    featsY = []
    for c ∈ 1:C
        d = length(processes(m)[c, 1])
        x = Array{eltype(processes(m)[c, 1])}(undef, d, N); y = similar(x)
        for n ∈ 1:N
            xr, yr = randjoint(procs[c, n], t)
            x[:, n] .= xr; y[:, n] .= yr
        end
        push!(featsX, x); push!(featsY, y)
    end
    return ObservedData(featsX), ObservedData(featsY)
end
