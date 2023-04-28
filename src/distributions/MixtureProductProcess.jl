using Distributions, DistributionsAD
using LinearAlgebra
using Memoization



# __________________________________________________________________________________________
struct MixtureProductProcess
    weights::AbstractVector{<:Real}
    processes::AbstractMatrix{<:AbstractProcess}
    function MixtureProductProcess(weights, processes)
        @assert length(weights) == size(processes, 2) "The number of regimes is not consistent"
        C = size(processes, 1)
        E = size(processes, 2)
        for c ∈ 1:C
            l = length(processes[c, 1])
            #todo check that distributions in a coordinate operate over the same domain
            for e ∈ 2:E
                @assert length(processes[c, e]) == l "The processes in each coordinate should have the same length"
            end
        end
        new(weights, processes)
    end
end

weights(m::MixtureProductProcess) = m.weights
processes(m::MixtureProductProcess) = m.processes
num_coords(m::MixtureProductProcess) = size(m.processes, 1)
num_regimes(m::MixtureProductProcess) = size(m.processes, 2)

# function fulllogpdf(p::MixtureProductProcess, t::Real,
#                     X::ObservedData,
#                     Y::ObservedData)
#     r = Matrix{Real}(num_)
#     fulllogpdf!(r, p, t, X, Y)
#     return r
# end

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

function fulllogpdf(p::MixtureProductProcess, t::Real,
                    X::ObservedData,
                    Y::ObservedData)
    n = num_sites(X)
    m = num_sites(Y)
    r = zeros(num_sites(X)+1, num_sites(Y)+1)
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
    fill!(r, -Inf)
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
    fill!(r, -Inf)
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
    featsX = Vector{AbstractArray{Real}}(undef, 0)
    for c ∈ 1:C
        d = length(processes(m)[c, 1])
        x = Array{eltype(processes(m)[c, 1])}(undef, d, N)
        for n ∈ 1:N
            x[:, n] .= rand(dists[c, n])
        end
        push!(featsX, x)
    end
    return ObservedData(featsX)
end

function randjoint(m::MixtureProductProcess, t::Real, N::Integer)
    sites = rand(Categorical(weights(m)), N)
    procs = processes(m)[:, sites]
    C = num_coords(m)
    featsX = Vector{AbstractArray{Real}}(undef, 0)
    featsY = Vector{AbstractArray{Real}}(undef, 0)
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

show(io::IO, p::MixtureProductProcess) = print(io, "MixtureProductProcess(" *
                                                   "\nnum coords: " * string(num_coords(p)) *
                                                   "\nnum regimes: " * string(num_regimes(p)) *
                                                   "\nweights: " * string(weights(p)) *
                                                   "\nprocesses: " * string(processes(p)) *
                                                   "\n)")
