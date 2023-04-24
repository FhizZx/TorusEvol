

function forward!(α::AbstractArray{<:Real}, model::TKF92,
                  emission_lps::AbstractArray{<:Real};
                  known_ancestor=false)
    D = num_descendants(model)

    α = fill!(α, -Inf)
    # Initial state
    α[START, ones(D)] = 0
    grid = Iterators.product(axes(α)[2:end]...)
    end_corner = 1 .+ size(α)

    state_ids = state_ids(model)
    values = known_ancestor ? values(model) : descendant_values(model)
    #A = known_ancestor ? transmat(model) :

    # Recursion
    for indices ∈ grid, s ∈ state_ids
        if all(indices .+ 1 .>= s)
            state = values[s]
            @. obs_ind = ifelse(state == 1, indices, end_corner)
            curr_ind = 1 .+ indices
            prev_ind = curr_ind .- state
            α[s, curr_ind...] = emission_lps[obs_ind...] +
                                logsumexp([A[:, s] .+ α[:, prev_ind...]])
        end
    end

    return α
end
