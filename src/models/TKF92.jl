using Memoization
using OneHotArrays
using LogExpFunctions

const END_INDEX=1
const START_INDEX=2


function gen_scenarios(D::Integer)
    scenarios = digits.(2^D:(2^(D+1)-1), base=2); pop!.(scenarios)
    return scenarios
end
gen_start_state(D::Integer) = [zeros(Int, D+1);ones(Int, D)]
gen_end_state(D::Integer) = zeros(Int, 2*D+1)

gen_ancestor_state(values) = [1; values; values]
function gen_descendant_state(d::Integer, flags)
    D = length(flags)
    [0; collect(Int, onehot(d, 1:D)); flags]
end

function gen_align_states(D::Integer)
    scenarios = gen_scenarios(D)
    res = [gen_end_state(D), gen_start_state(D)]
    append!(res, gen_ancestor_state.(scenarios))
    for flags ∈ scenarios
        append!(res, gen_descendant_state.(1:D, Ref(flags)))
    end
    return res
end

function gen_descendant_state_ids(D)
    scenarios = gen_scenarios(D)
    n = 2 + 2^D
    descendant_ids = Dict()
    for flags ∈ scenarios
        descendant_ids[flags] = n .+ (1:D)
        n += D
    end
    return descendant_ids
end

const MAX_NUM_DESCENDANTS = 5
const align_states = [gen_align_states(D) for D ∈ 1:MAX_NUM_DESCENDANTS]
@views const align_state_values = [[v[1:(D+1)] for v ∈ align_states[D]] for D ∈ 1:MAX_NUM_DESCENDANTS]
@views const align_state_desc_values = [[v[2:(D+1)] for v ∈ align_states[D]] for D ∈ 1:MAX_NUM_DESCENDANTS]
@views const align_state_flags = [[v[(D+2):(2*D+1)] for v ∈ align_states[D]] for D ∈ 1:MAX_NUM_DESCENDANTS]
const align_state_ids = [Dict((s[i], i) for i ∈ eachindex(s)) for s ∈ align_states]
const descendant_state_ids = [gen_descendant_state_ids(D) for D ∈ 1:MAX_NUM_DESCENDANTS]
ancestor_state_ids(D) = 2 .+ (1:2^D)

# res[i, j] = sum(v[i:(j-1)]), i < j
function segment_sum(v::AbstractVector)
    n = length(v)
    res = fill(-Inf, n, n+1)
    for i ∈ 1:n
        res[i, i] = 0
        for j ∈ i:n
            res[i, j+1] = res[i, j] + v[j]
        end
    end
    return res
end

# log ℙ[link survives for time t]
#α(t, λ, μ) = exp(-μt)
function lα(t::Real, λ::Real, μ::Real)
    return -μ*t
end

# log ℙ[surviving link gives at least one birth in time t]
#β(λ, μ, t) = λ(1 - exp((λ - μ) * t)) / (μ - λ*exp((λ - μ) * t))
function lβ(t::Real, λ::Real, μ::Real)
    return log(λ) - log(μ) + log1mexp((λ - μ) * t) - log1mexp((log(λ) - log(μ)) + (λ - μ) * t)
end

# log ℙ[dying link gives at least one birth in time t]
#γ(λ, μ, t) = 1 - μ(1 - exp((λ - μ) * t)) / [(1 - exp(-μ*t)) * (μ - λ*exp((λ - μ) * t))]
function lγ(t::Real, λ::Real, μ::Real)
    #return log1mexp(μ - λ + lβ(t, λ, μ) + log1mexp(lα(t, λ, μ)))
    lμβ = log1mexp((λ - μ) * t) - log1mexp((log(λ) - log(μ)) + (λ - μ) * t)
    return log1mexp(logaddexp(lμβ, -μ*t)) - log1mexp(-μ*t)
end

# Generate the transition logprobability matrix for an EvolHMM with D = length(ts) number of
# descendants and one common ancestor, with the evolutionary time which has passed between
# the ancestor and descendant #i given by ts[i]
# λ gives the birth rate of a link, μ gives the death rate of a link and
# r gives the extension probability of a link (so that a link has geometrically distributed length)
function gen_full_trans_mat(ts::AbstractVector{<:Real}, λ::Real, μ::Real, r::Real)
    D = length(ts)
    num_states = length(align_states[D])
    A = fill(-Inf, num_states, num_states)

    # Link extension log probability
    lext = log(r)
    # Log probability that link is done extending
    lnoext = log1mexp(lext)

    # log ℙ[link survives for time tᵢ]
    lαs = lα.(ts, Ref(λ), Ref(μ))
    # log ℙ[link dies by time tᵢ]
    nlαs = log1mexp.(lαs)

    # log ℙ[surviving link gives at least one birth in time tᵢ]
    lβs = lβ.(ts, Ref(λ), Ref(μ))
    # log ℙ[surviving link give no birth in time tᵢ]
    nlβs = log1mexp.(lβs)

    # log ℙ[dying link gives at least one birth in time tᵢ]
    lγs = lγ.(ts, Ref(λ), Ref(μ))
    # log ℙ[dying link gives no birth in time tᵢ]
    nlγs = log1mexp.(lγs)

    # Elongation log probability
    lcont = log(λ) - log(μ)
    # END log probability
    lend = log1mexp(lcont)

    # An ancestor gets copied D times, then each copy either survives with probability α(tᵢ)
    # or it dies, independently for each copy. Consider all possible combinations of such events
    # These are the so-called scenarios.
    # There are 2ᴰ of them, and the insertion probabilities are dependent on the specific
    # scenario, hence we keep track of them in the EvolHMM states, keeping a boolean flag
    # for each copy: flags[i] == 1 iff the i'th copy survives

    scenarios = gen_scenarios(D)
    ancestor_ids = ancestor_state_ids(D)

    # Log probability of a link experiencing a certain scenario
    lpscenarios = Array{Real}(undef, length(scenarios))
    for i ∈ eachindex(scenarios)
        @views flags = scenarios[i]
        lpscenarios[i] = sum(ifelse.(flags .== 1, lαs, nlαs))
    end
    # "Simulate" a run for each scenario to compute the transition probabilities
    for i ∈ eachindex(scenarios)
        @views flags = scenarios[i]

        # First birth probabilities in the specific scenario
        inslps = @. ifelse(flags == 1, lβs, lγs)
        noinslps = @. ifelse(flags == 1, nlβs, nlγs)

        lps = push!(inslps, lend)
        nlps = push!(noinslps, lcont)

        anc_id = ancestor_ids[i]
        desc_ids = descendant_state_ids[D][flags]
        end_id = END_INDEX

        dummy_transitions = segment_sum(nlps)
        # From state #i, first need to leave and enter dummy state #i
        # i.e. you don't extend the link anymore and no more rebirths
        # with probability (1-r) the link doesn't get extended anymore
        # also, if in a descendant state, with probability (1-βᵢ), the link doesn't
        # give any more births.
        # Then, you travel through the dummy state chain from state #i to state #j
        # with probability given by the product of the dummy transitions computed before
        # Then finally, enter state #j with log probability lp[j] (hence why we take
        # the transpose of lp )
        lptodummy = lnoext .+ [0; nlβs]
        A[[anc_id; desc_ids], [desc_ids; end_id]] .=  lptodummy .+
                                                      dummy_transitions[:, 1:end-1] .+
                                                      lps'

        # We also need to handle link extensions and rebirths (i.e. self-transitions)
        # Note that the rebirth probability doesn't depend on the scenario

        for d ∈ 1:D
            d_id = desc_ids[d]
            A[d_id, d_id] = logaddexp(lext, lnoext + lβs[d])
        end

        # Finally, we arrive at the final dummy state, corresponding to the
        # scenario decision node, where we elongate the ancestor by a new link,
        # then see if it survives or dies on each of the D branches
        # For state i in the run, state j in the ancestors
        # the probability of getting from i to j is the probability of no extension/elongation
        # and travelling to dummy state i, then travelling to the last dummy state,
        # then deciding on the scenario with log probability lpscenarios[j]
        # (hence why we transpose lpscenarios)

        A[[anc_id; desc_ids], ancestor_ids] = lptodummy .+ dummy_transitions[:, end] .+ lpscenarios'
        A[anc_id, anc_id] = logaddexp(lext, A[anc_id, anc_id])
    end
    # The transitions from the start node are a special case, because the immortal link
    # cannot die, but also it doesn't extend. This corresponds to the scenario where all
    # copies survive, so we can reuse the probabilities we computed there, but need to
    # undo the inclusion of the extension probabilities
    all_surv_anc_id = align_state_ids[D][gen_ancestor_state(ones(D))]
    A[START_INDEX, :] .= A[all_surv_anc_id, :] .- lnoext
    # Also, cannot linger in the start state
    A[START_INDEX, START_INDEX] = -Inf
    # Also, need to remove the extension self-loop transition
    A[START_INDEX, all_surv_anc_id] = logsubexp(A[all_surv_anc_id, all_surv_anc_id],
                                                lext) - lnoext

    # We have now covered the start node transitions, for each scenario all the transitions
    # in a run between ancestor and descendant state and back to a different scenario
    # This covers all transitions in the HMM, as the end state doesn't have any

    # so we can safely return the full transition matrix

    # but first, normalize the row probabilities so that they sum up to 1
    #@views A[2:end, :] .-= logsumexp(A[2:end, :]; dims=2)
    return A
end

struct TKF92
    D::Integer
    ts::AbstractVector{<:Real}
    λ::Real
    μ::Real
    r::Real
    A::AbstractMatrix{<:Real}
end

function TKF92(ts::AbstractVector{<:Real}, λ::Real, μ::Real, r::Real)
    A = gen_full_trans_mat(ts, λ, μ, r)
    D = length(ts)
    return TKF92(D, ts,  λ, μ, r, A)
end



transmat(model::TKF92) = model.A


# Return the id of the state corresponding to the ancestor state for which
# no descendants survive
no_survivors_ancestor_id(model::TKF92) = align_state_ids[model.D][gen_ancestor_state(zeros(D))]

num_descendants(model::TKF92) = model.D
state_ids_dict(model::TKF92) = align_state_ids[model.D]
state_ids(model::TKF92) = 1:num_states(model)
proper_state_ids(model::TKF92) = @views state_ids(model)[3:end]
states(model::TKF92) = align_states[model.D]
num_states(model::TKF92) = length(align_states[model.D])
state_values(model::TKF92) = align_state_values[model.D]
descendant_values(model::TKF92) = align_state_desc_values[model.D]

show(io::IO, model::TKF92) = print(io, "TKF92(" *
                                  "\nnum descendants: " * string(num_descendants(model)) *
                                  "\nts: " * string(model.ts) *
                                  "\nλ: " * string(model.λ) *
                                  "\nμ: " * string(model.μ) *
                                  "\nr: " * string(model.r) *
                                  "\n)")
