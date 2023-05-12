using LogExpFunctions
using Distributions
using TorusEvol


const NUM_PAIRHMM_TESTS = 44
@testset "PairDataHMM test $v" for v ∈ 1:NUM_PAIRHMM_TESTS
    Random.seed!(TEST_SEED+v)

    t = rand(Exponential(1))

    λ_a = rand(Exponential(0.01))
    seq_length = rand(Exponential(0.01))
    μ_a = (seq_length+1) * λ_a
    r_a = rand(Uniform(0,1))
    align_model = TKF92([t], λ_a, μ_a, r_a)

    E = rand(Geometric(0.2))+1
    proc_weights = rand(Dirichlet(E, 1.0))

    sub_procs = reshape(fill(WAG_SubstitutionProcess(), E), 1, E)

    μ_𝜙 = rand(E); μ_𝜓 = rand(E); σ_𝜙 = rand(E); σ_𝜓 = rand(E);
    α_𝜙 = rand(E); α_𝜓 = rand(E); α_cov = rand(E); γ = rand(E)
    for e ∈ 1:E
        μ_𝜙[e] = rand(Uniform(-π, π))
        μ_𝜓[e] = rand(Uniform(-π, π))
        σ_𝜙[e] = sqrt(rand(InverseGamma())) / π
        σ_𝜓[e] = sqrt(rand(InverseGamma())) / π
        α_𝜙[e] = sqrt(rand(InverseGamma())) / π
        α_𝜓[e] = sqrt(rand(InverseGamma())) / π
        α_corr = rand(Beta(3, 3))*2 - 1
        α_cov[e] = α_corr * (α_𝜙[e] * α_𝜓[e]) / 10
        γ[e] = rand(Gamma(10.0))
    end
    diff_procs = reshape(jumping.(WrappedDiffusion.(μ_𝜙, μ_𝜓, σ_𝜙, σ_𝜓, α_𝜙, α_𝜙, α_cov), γ), 1, E)

    ξ = MixtureProductProcess(proc_weights, vcat(sub_procs, diff_procs))


    N = rand(Geometric(0.01))+1
    M = rand(Geometric(0.01))+1
    X = randstat(ξ, N)
    Y = randstat(ξ, M)

    emission_lps = fulllogpdf(ξ, t, X, Y)

    pairdatahmm = PairDataHMM(align_model, num_sites(X), num_sites(Y))

    lp = logpdf(pairdatahmm, emission_lps)


    prop = rand(Uniform(0, 1))
    anc_align_model = TKF92([t*prop, t*(1-prop)], λ_a, μ_a, r_a; known_ancestor=false)
    triplehmm = PairDataHMM(anc_align_model, num_sites(X), num_sites(Y))

    rev_pair_hmm = PairDataHMM(align_model, num_sites(Y), num_sites(X))
    emission_lps_rev = fulllogpdf(ξ, t, Y, X)


    # Test the reversibility of the model:
    # marginalising the ancestor out is the same as having one descendant directly evolve
    #into a second descendant
    @test logpdf(pairdatahmm, emission_lps) ≈ logpdf(triplehmm, emission_lps) atol=1e-2 rtol=0.05

    # X evolving into Y is the same as Y evolving into X
    @test logpdf(pairdatahmm, emission_lps) ≈ logpdf(rev_pair_hmm, emission_lps_rev) atol=1e-2 rtol=0.005
end
