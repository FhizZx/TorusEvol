@model function posterior(D) # input data
    # weight = 1

    θ ~ prior()              # sample parameter
    # weight = p(θ)

    D ~ likelihood(θ)        # observe data
    # weight = p(θ) * p(D | θ)
    #        = p(θ, D)

    return θ
end
