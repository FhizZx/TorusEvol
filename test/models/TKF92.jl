using LogExpFunctions
using Distributions

@testset "TKF92 one descendant tests" begin
    λ = rand(Exponential(1.0))
    seq_length = rand(Exponential(0.01))
    μ = (seq_length+1) * λ
    r = rand(Uniform(0,1))
    t = rand(Exponential(1.0))



    model = TKF92([t], λ, μ, r)
    new_trans_mat = transmat(model)
    # scenarios = [[0], [1]]
    # states: 1END, 2START,
    # 3DELETE: 1desc(0)flag(0),
    # 4MATCH: 1desc(1)flag(1)
    # 5,6INSERTS: 0desc(1)flag(0), 0desc(1)flag(1)

    # need to sum up the probabilities of the 2 inserts,
    # since the states coincide in the old transmat
    my_mat = new_trans_mat[1:5, 1:5]
    my_mat[1:4, 5] .= logaddexp.(new_trans_mat[1:4, 5], new_trans_mat[1:4, 6])
    my_mat[5, 1:4] .= logaddexp.(new_trans_mat[5, 1:4], new_trans_mat[6, 1:4])
    my_mat[5, 5] = logsumexp(new_trans_mat[5:6, 5:6])

    old_mat = gen_pair_tkf92_transmat(t, λ, μ, r)

    @test my_mat ≈ old_mat atol=1e-7

    # Check that exp.(my_mat) is a transition probability matrix
    @test all(logsumexp(my_mat; dims=1) .≈ 0)

    #todo test edge cases for hyper parameter values (e.g. what happens when t is small, large)
end

# For correctness checks
#beta(λ, μ, t) = (1 - exp((λ - μ) * t)) / (μ - λ*exp((λ - μ) * t))
logμβ(λ, μ, t) = log1mexp((λ - μ) * t) - log1mexp((log(λ) - log(μ)) + (λ - μ) * t)
function gen_pair_tkf92_transmat(t::Real, λ::Real, μ::Real, r::Real)
    END=1; START=2; DELETE=3; MATCH=4; INSERT=5
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
