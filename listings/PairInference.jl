
@model function pair_parameter_inference(pairs)
    # ______________________________________________________________________________
    # Step 1 - Sample prior parameters

    # Time parameter
    t ~ Exponential(1.0)
    # Alignment parameters
    @submodel prefix="τ" Λ = tkf92_prior()
    # Dihedral parameters
    @submodel prefix="Θ" Ξ = jwndiff_prior()
    # Check parameter validity
    if t ≤ 0 || any(isnan.(Ξ)) || any(isnan.(Λ))
        Turing.@addlogprob! -Inf; return # Reject sample
    end

    # ______________________________________________________________________________
    # Step 2 - Construct processes

    # Substitution Process
    S = WAG_SubstitutionProcess()
    # Dihedral Process
    Θ = JumpingWrappedDiffusion(Ξ...)
    # Joint sequence-structure site level process with one regime
    ξ = ProductProcess(S, Θ)

    # Alignment model
    τ = TKF92([t], Λ...)

    # Chain level model
    Γ = ChainJointDistribution(ξ, τ, t)

    # ______________________________________________________________________________
    # Step 3 - Observe each pair X, Y by proxy of their joint probability under Γ
    for (X, Y) ∈ pairs
        (X, Y) ~ Γ
    end

    return Γ
end
