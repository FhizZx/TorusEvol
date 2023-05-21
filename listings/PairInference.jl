
@model function pair_parameter_inference(pairs)
    N = length(pairs)
    # ______________________________________________________________________________
    # Step 1 - Sample prior parameters

    # Time parameter for each pair
    for i ∈ 1:D
        t[i] ~ Exponential(1.0)
    end
    # Alignment parameters
    @submodel prefix="τ" Λ = tkf92_prior()
    # Dihedral parameters
    @submodel prefix="Θ" Ξ = jwndiff_prior()
    # Check parameter validity
    if any(t .≤ 0) || any(isnan.(Ξ)) || any(isnan.(Λ))
        reject_sample()
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

    # ______________________________________________________________________________
    # Step 3 - Observe each pair X, Y by proxy of their joint probability
    for i ∈ 1:N
        X, Y = pairs[i]
        τ = TKF92([t[i]], Λ...)
        (X, Y) ~ ChainJointDistribution(ξ, τ)
    end

    return t, Λ, Ξ
end
