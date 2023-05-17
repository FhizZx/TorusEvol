@model function trajectory_reconstruction(Y, Z, M_YZ, Γ)
    @submodel X, M_XYZ = ancestor_chain_sampler(Y, Z, M_YZ)
    left = trajectory_reconstruction(Y, X, t/2)
    right = trajectory_reconstruction(X, Z, t/2)

end
