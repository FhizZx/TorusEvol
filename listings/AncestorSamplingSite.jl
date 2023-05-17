
@model function ancestor_sampler(ξ::AbstractProcess, # evolutionary process
                                 descs::Vector{𝕊},   # observed descendants
                                 ts::Vector{Real}    # branch lengths
                                 ) where 𝕊           # state space
    D = length(descs)
    if D == 0                       # no descendant information
        anc ~ statdist(ξ); return   # sample from equilibrium
    end

    # Observe the first descendant
    descs[1] ~ statdist(ξ)

    # Sample ancestor from the first descendant
    anc ~ transdist(ξ, ts[1], descs[1])

    # Observe the rest of the descendants, given ancestor
    for i ∈ 2:D
        descs[i] ~ transdist(ξ, ts[i], anc)
    end

    return anc
end
