

@model function ancestor_chain_sampler(ξ, τ, descs, ts,τ)



    for i ∈ 1:N
        desc_mask = col[2:end] .== 1
        for c ∈ 1:C
            @submodel anc[c][:, i] = ancestor_sampler(ξ, descs[desc_mask], ts[desc_mask])
        end
    end
    D = length(descs)
    if D == 0
        anc ~ statdist(ξ); return
    end

    # Observe the first descendant
    descs[1] ~ statdist(ξ)

    # Sample ancestor from the first descendant
    anc ~ transdist(ξ, ts[1], descs[1])

    # Observe the rest of the descendants, given ancestor
    for i ∈ 2:D
        descs[i] ~ transdist(ξ, ts[i], anc)
    end
end
