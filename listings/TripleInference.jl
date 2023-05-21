
@model function triple_alignment_sampler(Y, Z, W, Λ, ξ; max_N_X=200)
    # _____________________________________________________________________________
    # Step 1 - Sample prior parameters

    # Time parameters
    t_Y ~ Exponential(1.0)
    t_Z ~ Exponential(1.0)
    t_W ~ Exponential(1.0)

    # Check parameter validity
    if t_Y ≤ 0 || t_Z ≤ 0 || t_W ≤ 0
        reject_sample()
    end

    # _____________________________________________________________________________
    # Step 2 - Observe data and simultaneously construct alignment

    # First, observe Y and Z and sample a triple alignment of X, Y and Z
    τ_XYZ = TKF92([t_Y, t_Z], Λ...; known_ancestor=false)
    (Y, Z) ~ ChainJointDistribution(ξ, τ_XYZ)
    M_XYZ ~ ConditionedAlignmentDistribution(τ_XYZ, ξ, Y, Z)

    # Construct X, the hidden ancestor chain, given alignment M_XYZ and data Y, Z
    X = hiddenchain_from_alignment(Y, Z, t_Y, t_Z, M_XYZ, ξ)

    # Finally, observe W given X and sample alignment of X and W
    τ_XW = TKF92([t_W], Λ...; known_ancestor=true)
    W ~ ChainTransitionDistribution(ξ, τ_XW, X)
    M_XW ~ ConditionedAlignmentDistribution(τ_XW, ξ, X, W)

    M_XYZW = combine(M_XYZ, M_XW)
    M_YZW = subalignment(M_XYZW, [2, 3, 4])

    return t_Y, t_Z, t_W, M_YZW
end;
