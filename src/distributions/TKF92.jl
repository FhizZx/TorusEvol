using Memoization

const START=1
struct TKF92
    num_descendants::Integer
    t::Real
    λ::Real
    μ::Real
    r::Real
    A::AbstractMatrix{<:Real}
    end_transitions::AbstractVector{<:Real}
    dominos::AbstractMatrix{Integer}
end

function TKF92(ts::AbstractVector{<:Real}, λ::Real, μ::Real, r::Real)
    D = length(ts)

    # log ℙ[link survives for time tᵢ]
    lαs = lα.(ts)
    # log ℙ[link dies by time tᵢ]
    nlαs = log1mexp.(lαs)

    # log ℙ[surviving link gives at least one birth in time tᵢ]
    lβs = lβ.(ts)
    # log ℙ[surviving link give no birth in time tᵢ]
    nlβs = log1mexp.(lβs)

    # log ℙ[dying link gives at least one birth in time tᵢ]
    lγs = lγ.(ts)
    # log ℙ[dying link gives no birth in time tᵢ]
    nlγs = log1mexp.(lγs)

    # Elongation log probability
    lcont = log(λ) - log(μ)
    # END log probability
    lend = log1mexp(lcont)

    scenarios = digits.(2^D:(2^(D+1)-1), base=2); pop!.(scenarios)

    lpscenarios = Array{Real}(undef, length(scenarios))
    for i ∈ eachindex(scenarios)
        @views flags = scenarios[i]
        lpscenarios[i] = sum(ifelse.(flags .== 1, lαs, nlαs))

    for i ∈ eachindex(scenarios)
        @views flags = scenarios[i]
        # Birth probabilities in specific scenario
        @. inslps = ifelse(flags == 1, lβs, lγs)
        @. noinslps = ifelse(flags == 1, nlβs, nlγs)

        lps = push!(inslps, lend)
        nlps = push!(noinslps, lcont)
        run_states =

        A[id.(run_states), id.(run_states)] .= lp_ij .+ nlps'

        A[id.(run_states), ] .= A[id.(run_states), id(end)] .- lend .+ lcont  .+ A[]

    # Start transitions
    A[start, ] .= A[all, ]

    # Elongation probabilities
    # Ancestor states

    # Descendant states
    logaddexp(lr, log1mexp(lr) + lpbi)

end

transmat(model::TKF92) = model.A
end_transitions(model::TKF92) = model.end_transitions
# ancestor_extension_prob(model::TKF92) = model.r + p[back to myself]

# Includes start node but not end node
num_states(model::TKF92) = size(transmat(model), 1)
states(model::TKF92) = 1:num_states(model)
domino(model::TKF92, state::Integer) = model.dominos[:, state]




function TKF92TransitionMatrix(λ::Real, μ::Real, r::Real, t::Real; unitinserts::Bool=false)
    ltransMat = Array{Real}(undef, NUM_ALIGN_STATES, NUM_ALIGN_STATES)
    ltransMat = fill!(ltransMat, -Inf)

    # log(μ*β(t))
    lμβ = logμβ(λ, μ, t)
    lμ = log(μ)
    lλ = log(λ)
    lr = log(r)

    # lpxy = log probability of transitioning from state x to state y
    # S = start, E = end
    # b = birth decision area, e = elongation decision area
    # f = fragment death decision area (match or del)
    # g = dead fragment # of children decision area (0 or >0)
    # i = insertion, d = deletion, m = match -> the actual emissions


    # pSb = 1
    lpSb = 0
    # pbi = λ*β
    lpbi = lλ - lμ + lμβ
    # pbe = 1 - pbi
    lpbe = log1mexp(lpbi)

    # need to change back
    #pii = r + (1-r)*λ*β
    pii = r + (1-r) * exp(lpbi)
    lpii = unitinserts ? -Inf : logaddexp(lr, log1mexp(lr) + lpbi)
    #pie = 1 #pie = 1 - pii
    lpie = log1mexp(lpii)

    #pef = λ / μ
    lpef = lλ - lμ
    #peE = 1 - pef
    lpeE = log1mexp(lpef)
    #pfm = exp(-μ*t)
    lpfm = -μ*t
    #pfd = 1 - pfm = 1 - exp(-μ*t)
    lpfd = log1mexp(lpfm)
    #pmm = r
    lpmm = lr
    #pmb = 1 - r
    lpmb = log1mexp(lpmm)
    #pdd = r
    lpdd = lr
    #pdg = 1-r
    lpdg = lpmb
    #pgi = (1 - μ * β - exp(-μ*t)) / (1-exp(-μ*t))
    lpgi = log1mexp(logaddexp(lμβ, -μ*t)) - lpfd
    #pge = 1 - pgi
    lpge = lμβ - lpfd

    ltransMat[START, INSERT] = lpSb + lpbi
    ltransMat[START, MATCH] = lpSb + lpbe + lpef + lpfm
    ltransMat[START, DELETE] = lpSb + lpbe + lpef + lpfd
    ltransMat[START, END] = lpSb + lpbe + lpeE

    ltransMat[INSERT, INSERT] = lpii
    ltransMat[INSERT, MATCH] = lpie + lpef + lpfm
    ltransMat[INSERT, DELETE] = lpie + lpef + lpfd
    ltransMat[INSERT, END] = lpie + lpeE

    ltransMat[MATCH, INSERT] = lpmb + lpbi
    ltransMat[MATCH, MATCH] = lpmm + lpmb + lpbe + lpef + lpfm
    ltransMat[MATCH, DELETE] = lpmb + lpbe + lpef + lpfd
    ltransMat[MATCH, END] = lpmb + lpbe + lpeE

    ltransMat[DELETE, INSERT] = lpdg + lpgi
    ltransMat[DELETE, MATCH] = lpdg + lpge + lpef + lpfm
    ltransMat[DELETE, DELETE] = lpdd + lpdg + lpge + lpef + lpfd
    ltransMat[DELETE, END] = lpdg + lpge + lpeE

    return ltransMat, pii
end

#beta(λ, μ, t) = (1 - exp((λ - μ) * t)) / (μ - λ*exp((λ - μ) * t))

logμβ(λ, μ, t) = log1mexp((λ - μ) * t) - log1mexp((log(λ) - log(μ)) + (λ - μ) * t)
