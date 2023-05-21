
function trajectory_reconstruction(M_YZ::Alignment,
                                   Y::ObservedChain, Z::ObservedChain,
                                   ξ::MixtureProductProcess, t::Real, Λ; levels=1)
    # ______________________________________________________________________________
    # Base case
    if levels == 0
        return [Y, Z], M_YZ
    end

    # ______________________________________________________________________________
    # Recursion

    # 1. Sample X midpoint of Y and Z, as well as the
    #    alignment M_XYZ, given the alignment M_YZ
    X, M_XYZ = ancestor_sampling(M_YZ, Y, Z, t, ξ, Λ)

    # 2. Reconstruct trajectories recursively on each branch
    M_YX  = subalignment(M_XYZ, [2, 1])
    traj_YX, M_Y_toX = trajectory_reconstruction(M_YX, Y, X, ξ, t/2, Λ; levels-1)
    M_XZ = subalignment(M_XYZ, [1, 3])
    traj_XZ, M_X_toZ = trajectory_reconstruction(M_XZ, X, Z, ξ, t/2, Λ; levels-1)

    # Combine the trajectories and alignments
    traj = [traj_YX; traj_XZ[2:end]]
    M = glue(M_Y_toX, M_X_toZ)
    return traj, M
end
