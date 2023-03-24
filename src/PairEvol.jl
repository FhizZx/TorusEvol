
@model function PairEvol(P_a::Protein, P_b::Protein)

    # Sample time
    t ~ Exponential(1)

    # Sample sequence parameters
    S = # exchangeability matrix
    Π = # stationary aminoacid distribution

    Φ = (S, Π) # substitution parameters
    seqmodel = SequenceModel(Φ...)


    # Sample structure parameters
    μ_ϕ ~ Uniform(-π, π)
    μ_ψ ~ Uniform(-π, π)
    σ_ϕ² ~ Gamma(1, 1)
    σ_ψ² ~ Gamma(1, 1)
    α_ϕ ~ Gamma(1, 1)
    α_ψ ~ Gamma(1, 1)
    α_corr ~ truncated(Normal(0, 1), -1, 1)

    Θ = ([μ_ϕ, μ_ψ], √σ_ϕ², √σ_ψ², α_ϕ, α_ψ, √(α_ϕ * α_ψ)*α_corr)
    structmodel = WrappedDiffusion(Θ...)


    # Make substitution model that incorporates both sequence and structure
    # submodel = ProductProcess(seqmodel, structmodel)


    # Sample indel parameters
    λ ~ Gamma(1, 1)
    μ ~ Gamma(1, 1)
    r ~ Uniform(0, 1)

    Λ = (λ, μ, r)
    indelmat = TKF92TransitionMatrix(Λ..., t)


    # Finally create the PairHMM model from the substitution and indel models
    pairmodel = PairHMM(indelmat, seqmodel, structmodel, t)

    Turing.@addlogprob! pairmodel((P_a, P_b))

    params = (Φ, Θ, Λ, t)
    return params
end
