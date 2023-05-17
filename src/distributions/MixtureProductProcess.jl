using Distributions, DistributionsAD
using LinearAlgebra
using Memoization


# ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄
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
#                     X::ObservedChain,
#                     Y::ObservedChain)
#     r = Matrix{Real}(num_)
#     fulllogpdf!(r, p, t, X, Y)
#     return r
# end

function ProductProcess(processes...)
    return MixtureProductProcess([1.0], hcat(collect(processes)))
end

@memoize function fulltranslogpdf!(r::AbstractMatrix{<:Real},
                          p::MixtureProductProcess,
                          t::Real,
                          X::HiddenChain,
                          Y::ObservedChain)
    n = num_sites(X)
    m = num_sites(Y)
    r .= -Inf
    r[1:n+1, m+1] .= 0
    @views translogpdf!(r[1:n, 1:m], p, t, X, Y)
    @views statlogpdf!(r[n+1, 1:m], p, Y)
    return r
end

@memoize function fulljointlogpdf!(r::AbstractMatrix{<:Real},
                     p::MixtureProductProcess, t::Real,
                     X::ObservedChain,
                     Y::ObservedChain)
    n = num_sites(X)
    m = num_sites(Y)
    r .= -Inf
    r[n+1, m+1] = 0
    @views jointlogpdf!(r[1:n, 1:m], p, t, X, Y)
    @views statlogpdf!(r[1:n, m+1], p, X)
    @views statlogpdf!(r[n+1, 1:m], p, Y)
    return r
end

@memoize function fulljointlogpdf(p::MixtureProductProcess, t::Real,
                                  X::ObservedChain,
                                  Y::ObservedChain)
    n = num_sites(X)
    m = num_sites(Y)
    r = Array{Float64}(undef, num_sites(X)+1, num_sites(Y)+1)
    r .= -Inf
    r[n+1, m+1] = 0
    @views jointlogpdf!(r[1:n, 1:m], p, t, X, Y)
    @views statlogpdf!(r[1:n, m+1], p, X)
    @views statlogpdf!(r[n+1, 1:m], p, Y)
    return r
end

function logdotexp!(r, A::AbstractMatrix,
                       B::AbstractMatrix)
    N, M = size(r)
    for i ∈ 1:N, j ∈ 1:M
        @views r[i, j] = logsumexp(A[:, i] .+ B[:, j])
    end
    return r
end


function translogpdf!(r::AbstractMatrix{<:Real},
                          m::MixtureProductProcess,
                          t::Real,
                          X::HiddenChain,
                          Y::ObservedChain)
    N_X = num_sites(X)
    N_Y = num_sites(Y)
    E = num_regimes(m)
    C = num_coords(m)
    workspace = similar(r)
    workspace .= -Inf

    # Compute the contribution of each regime to the final logpdf
    r_e = similar(workspace)
    r_e .= -Inf

    ys = data(Y)


    for e ∈ 1:E
        w = weights(m)[e]
        r_e .= log(w)
        for c ∈ 1:C
            p = processes(m)[c, e]

            # Domain over which we marginalise
            Ω = data(domain(p))
            N_Ω = length(domain(p))
            A_Ω = area(domain(p))

            tape = similar(r, N_Ω, N_Y)
            tape .= -Inf
            translogpdf!(tape, p, t, Ω, ys[c])

            @views ρ_X = logprobs(X)[c][e, :, :]
            logdotexp!(workspace, ρ_X, tape)
            workspace .+= log(A_Ω) - log(N_Ω)

            r_e .= r_e .+ workspace
        end
        r .= logaddexp.(r, r_e)
    end

    return r

end

function jointlogpdf!(r::AbstractMatrix{<:Real},
                      m::MixtureProductProcess, t::Real,
                      X::ObservedChain,
                      Y::ObservedChain)
    E = num_regimes(m)
    @assert num_coords(X) == num_coords(Y)
    @assert num_coords(X) == num_coords(m) string(num_coords(X)) * " " * string(num_coords(m))
    C = num_coords(X)
    workspace = similar(r)
    workspace .= -Inf
    xs = data(X)
    ys = data(Y)

    # Compute the contribution of each regime to the final logpdf
    r_e = similar(workspace)
    r_e .= -Inf

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
                     X::ObservedChain)
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

# function logpdf(ξ::MixtureProductProcess, alignment::Alignment, X::ObservedChain, Y::ObservedChain)
#     res = Real[]
#     for v ∈ alignment
#         if v == [1, 1]


#     end
#     return res
# end

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
    return featsX
end

function randjoint(m::MixtureProductProcess, t::Real, N::Integer)
    regimes = rand(Categorical(weights(m)), N)
    procs = processes(m)[:, regimes]
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
    return featsX, featsY
end

show(io::IO, p::MixtureProductProcess) = print(io, "MixtureProductProcess(" *
                                                   "\nnum coords: " * string(num_coords(p)) *
                                                   "\nnum regimes: " * string(num_regimes(p)) *
                                                   "\nweights: " * string(weights(p)) *
                                                   "\nprocesses: " * string(processes(p)) *
                                                   "\n)")


function hiddenchain_from_alignment(Y::ObservedChain, Z::ObservedChain,
                                    t_Y::Real, t_Z::Real, M_XYZ::Alignment,
                                    ξ::MixtureProductProcess)
    C = num_coords(ξ)
    E = num_regimes(ξ)

    alignment = M_XYZ
    X_mask = mask(alignment, [[1], [0,1], [0,1]])
    alignmentX = slice(alignment, X_mask)
    Y_mask = mask(alignment, [[0,1], [1], [0,1]])
    alignmentY = slice(alignment, Y_mask)
    Z_mask = mask(alignment, [[0,1], [0,1], [1]])
    alignmentZ = slice(alignment, Z_mask)

    X_maskX = mask(alignmentX, [[1], [0], [0]])

    XY_maskX = mask(alignmentX, [[1], [1], [0]])
    XY_maskY = mask(alignmentY, [[1], [1], [0]])

    XZ_maskX = mask(alignmentX, [[1], [0], [1]])
    XZ_maskZ = mask(alignmentZ, [[1], [0], [1]])

    XYZ_maskX = mask(alignmentX, [[1], [1], [1]])
    XYZ_maskY = mask(alignmentY, [[1], [1], [1]])
    XYZ_maskZ = mask(alignmentZ, [[1], [1], [1]])

    N = length(alignmentX)
    logprobs = Array{Float64, 3}[]
    domains = Array{Real, 2}[]

    dataY = data(Y)
    dataZ = data(Z)

    for c ∈ 1:C
        Ω = data(domain(processes(ξ)[c, 1]))
        N_Ω = length(domain(processes(ξ)[c, 1]))
        push!(domains, Ω)
        lp_c = Array{Float64}(undef, E, N_Ω, N)
        lp_c .= -Inf
        push!(logprobs, lp_c)

        for e ∈ 1:E
            p = processes(ξ)[c, e]

            # [1, 0, 0] - p(x) = statdist
            logprobs[c][e, :, X_maskX] .= logpdf(statdist(p), Ω)

            # [1, 1, 0] - p(x | y)
            dataY110 = @view dataY[c][:, XY_maskY]
            dataX110 = @view logprobs[c][e, :, XY_maskX]
            translogpdf!(dataX110', p, t_Y, dataY110, Ω)

            # [1, 0, 1] - p(z | y)
            dataZ101 = @view dataZ[c][:, XZ_maskZ]
            dataX101 = @view logprobs[c][e, :, XZ_maskX]
            translogpdf!(dataX101', p, t_Z, dataZ101, Ω)

            # [1, 1, 1] - p(x | y,z) = p(x)p(y | x)p(z | x) / p(y, z)
            dataY111 = @view dataY[c][:, XYZ_maskY]
            dataZ111 = @view dataZ[c][:, XYZ_maskZ]
            dataX111 = @view logprobs[c][e, :, XYZ_maskX]
            tape = similar(dataX111)

            dataX111 .= logpdf(statdist(p), Ω)
            translogpdf!(tape, p, t_Y, Ω, dataY111); dataX111 .+= tape
            translogpdf!(tape, p, t_Z, Ω, dataZ111); dataX111 .+= tape
            dataX111 .-= logpdf(statdist(p), dataY111)'
            ts = transdist.(Ref(p), Ref(t_Y + t_Z), eachcol(dataY111))
            dataX111 .-= logpdf(ts, dataZ111)'
        end
    end

    return HiddenChain(domains, logprobs, N, E)
end
