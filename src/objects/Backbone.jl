# Much of this code was adapted from Michael Golden's code
using BioStructures
using DataStructures
using LinearAlgebra
using Rotations

const BioChain = BioStructures.Chain

ccmod(x) = rem(x,2π, RoundNearest)

function average_length(a, b)
    (a + b) / 2
end

function average_angle(a, b)
    if a > b
        tmp = a; a = b; b = tmp
    end
    d1 = b - a
    d2 = a + 2π - b
    if d1 < d2
        return a + d1/2
    else
        return ccmod(b + d2/2)
    end
end

aminoacids = "ARNDCQEGHILKMFPSTWYV"
id_to_aa(i) = aminoacids[i]
num_aa = length(aminoacids)

aminoacid_ids = Dict((aminoacids[i], i) for i ∈ eachindex(aminoacids))
aa_to_id(a) = aminoacid_ids[a]

one_to_three = Dict('A' => "ALA", 'C' => "CYS", 'D' => "ASP", 'E' => "GLU", 'F' => "PHE",
                    'G' => "GLY", 'H' => "HIS", 'I' => "ILE", 'L' => "LEU", 'K' => "LYS",
                    'M' => "MET", 'N' => "ASN", 'P' => "PRO", 'Q' => "GLN", 'R' => "ARG",
                    'S' => "SER", 'T' => "THR", 'W' => "TRP", 'V' => "VAL", 'Y' => "TYR")
one_to_three = DefaultDict("XXX", one_to_three)


# Empirical backbone bond angle distribution
# ideal_bond_angle[bond1-bond2, aminoacid] = mean, stddev
# ideal angle ~ WrappedNormal(mean, stddev) radians
ideal_bond_angle = Dict(("CA-C-N", 'G') => (116.4, 2.1), ("CA-C-N", 'P') => (116.9, 1.5),
                        ("C-N-CA", 'G') => (120.6, 1.7), ("C-N-CA", 'P') => (122.6, 5.0),
                        ("N-CA-C", 'G') => (112.5, 2.9), ("N-CA-C", 'P') => (111.8, 2.5))
for aa ∈ filter(∉("GP"), aminoacids)
    ideal_bond_angle["CA-C-N", aa] = (116.2, 2.0)
    ideal_bond_angle["C-N-CA", aa] = (121.7, 1.8)
    ideal_bond_angle["N-CA-C", aa] = (111.2, 2.8)
end
map!(x->deg2rad.(x), values(ideal_bond_angle))

# Empirical backbone bond length distribution
# ideal_bond_length[bond, aminoacid] = mean, stddev
# ideal angle ~ Normal(mean, stddev) Å
ideal_bond_length = Dict(("C-N", 'P') => (1.341, 0.016),
                         ("N-CA", 'G') => (1.451, 0.016), ("N-CA", 'P') => (1.466, 0.015),
                         ("CA-C", 'G') => (1.516, 0.018))
for aa ∈ filter(∉("P"), aminoacids)
    ideal_bond_length["C-N",  aa] = (1.329, 0.014)
end
for aa ∈ filter(∉("GP"), aminoacids)
    ideal_bond_length["N-CA", aa] = (1.458, 0.019)
end
for aa ∈ filter(∉("G"), aminoacids)
    ideal_bond_length["CA-C", aa] = (1.525, 0.021)
end

# idealised so that only trans peptide bonds are considered
# cis peptide bonds are rare and mostly occur in proline
ideal_omega_angle = rad2deg.((179.6, 5.9))

function backbone_bond_angles_and_lengths(chain::BioChain)
    residues = collectresidues(chain, standardselector)
    n = length(residues)
    bond_angles = Matrix{Float64}(undef, 3, n)
    bond_lengths = Matrix{Float64}(undef, 3, n)

    res = residues[1]
    atoms = [res["N"],res["CA"],res["C"]]
    for i ∈ 2:n
        res = residues[i]
        new_atoms = [res["N"],res["CA"],res["C"]]
        for j ∈ 1:3
            atoms[j] = new_atoms[j]
            bond_angles[j, i] = bondangle(atoms[j%3+1], atoms[(j+1)%3+1], atoms[j])
            bond_lengths[j, i] = distance(atoms[(j+1)%3+1], atoms[j])
        end
    end
    return bond_angles, bond_lengths
end


function Residue(name::AbstractString, number::Integer, ch::BioChain)
    return BioStructures.Residue(name, number, ' ', false, ch)
end

function Atom(serial::Int, name::String, coords::Vector{Float64},
              element::String, residue::StructuralElement)
    return BioStructures.Atom(serial, name, ' ', coords, 1.0, 13.0, element, "  ", residue)
end

function build_biochain_from_aminoacids_dihedrals(aminoacid_ids::AbstractArray{<:Integer},
                                                  dihedrals::AbstractMatrix{<:Real};
                                                  id="X")
    aminoacids = String(id_to_aa.(vec(aminoacid_ids)))
    N = length(aminoacids)

    @assert N == size(dihedrals, 2)
    torsion_angles = Matrix{Float64}(undef, 3, N)
    torsion_angles[1, 2:end] = dihedrals[2, 1:(end-1)]             # ψ
    torsion_angles[3, 2:end] = dihedrals[1, 2:end]                 # ϕ
    torsion_angles[2, :] .= rand(Normal(ideal_omega_angle...), N)  # ω

    bond_angles = Matrix{Float64}(undef, 3, N)
    for i in 1:N
        bond_angles[1, i] = rand(WrappedNormal(ideal_bond_angle["CA-C-N", aminoacids[i]]...))[1]
        bond_angles[2, i] = rand(WrappedNormal(ideal_bond_angle["C-N-CA", aminoacids[i]]...))[1]
        bond_angles[3, i] = rand(WrappedNormal(ideal_bond_angle["N-CA-C", aminoacids[i]]...))[1]
    end

    bond_lengths = Matrix{Float64}(undef, 3, N)
    for i in 1:N
        bond_lengths[1, i] = abs(rand(Normal(ideal_bond_length["C-N",  aminoacids[i]]...)))
        bond_lengths[2, i] = abs(rand(Normal(ideal_bond_length["N-CA", aminoacids[i]]...)))
        bond_lengths[3, i] = abs(rand(Normal(ideal_bond_length["CA-C", aminoacids[i]]...)))
    end

    return build_chain_from_internals(id, aminoacids, torsion_angles, bond_angles, bond_lengths)
end

function build_biochain_from_aminoacids_dihedrals_alignment(chain::BioChain,
                                                            M_XY::Alignment,
                                                            aminoacid_ids::AbstractArray{<:Integer},
                                                            dihedrals::AbstractMatrix{<:Real};
                                                            id="X")
    alignment = Alignment(data(M_XY))

    maskX = mask(alignment, [[1], [0,1]])
    maskY = mask(alignment, [[0,1], [1]])

    N = count(maskY)

    alignmentX = slice(alignment, maskX)
    alignmentY = slice(alignment, maskY)
    insert_maskY = mask(alignmentY, [[0], [1]])
    match_maskX = mask(alignmentX, [[1], [1]])
    match_maskY = mask(alignmentY, [[1], [1]])

    aminoacids = String(id_to_aa.(vec(aminoacid_ids)))
    @assert length(aminoacids) == N

    @assert size(dihedrals, 2) == N
    torsion_angles = Matrix{Float64}(undef, 3, N)
    torsion_angles[1, 2:end] = dihedrals[2, 1:(end-1)]             # ψ
    torsion_angles[3, 2:end] = dihedrals[1, 2:end]                 # ϕ
    # ω
    torsion_angles[2, insert_maskY] .= rand(Normal(ideal_omega_angle...), count(insert_maskY))
    torsion_angles[2, match_maskY] .= omegaangles(chain, standardselector)[match_maskX]

    bond_angles_X, bond_lengths_X = backbone_bond_angles_and_lengths(chain)

    #TODO write this more succintly
    bond_angles = Matrix{Real}(undef, 3, N)
    for i in collect(1:N)[insert_maskY]
        bond_angles[1, i] = rand(WrappedNormal(ideal_bond_angle["CA-C-N", aminoacids[i]]...))[1]
        bond_angles[2, i] = rand(WrappedNormal(ideal_bond_angle["C-N-CA", aminoacids[i]]...))[1]
        bond_angles[3, i] = rand(WrappedNormal(ideal_bond_angle["N-CA-C", aminoacids[i]]...))[1]
    end
    bond_angles[:, match_maskY] .= bond_angles_X[:, match_maskX]

    bond_lengths = Matrix{Real}(undef, 3, N)
    for i in collect(1:N)[insert_maskY]
        bond_lengths[1, i] = abs(rand(Normal(ideal_bond_length["C-N",  aminoacids[i]]...)))
        bond_lengths[2, i] = abs(rand(Normal(ideal_bond_length["N-CA", aminoacids[i]]...)))
        bond_lengths[3, i] = abs(rand(Normal(ideal_bond_length["CA-C", aminoacids[i]]...)))
    end
    bond_lengths[:, match_maskY] .= bond_lengths_X[:, match_maskX]

    return build_chain_from_internals(id, aminoacids, torsion_angles, bond_angles, bond_lengths)
end

function build_biochain_from_triple_alignment(chainY::BioChain,
                                              chainZ::BioChain,
                                              M_XYZ::Alignment,
                                              aminoacid_ids::AbstractArray{<:Integer},
                                              dihedrals::AbstractMatrix{<:Real};
                                              id="X")
    alignment = Alignment(data(M_XYZ))
    X_mask = mask(alignment, [[1], [0,1], [0,1]])
    alignmentX = slice(alignment, X_mask)
    Y_mask = mask(alignment, [[0,1], [1], [0,1]])
    alignmentY = slice(alignment, Y_mask)
    Z_mask = mask(alignment, [[0,1], [0,1], [1]])
    alignmentZ = slice(alignment, Z_mask)


    X_maskX = mask(alignmentX, [[1], [0], [0]])

    XY_maskX = mask(alignmentX, [[1], [1], [0]])
    XY_maskY = mask(alignmentY, [[1], [1], [0]])

    XZ_maskX = mask(alignmentX, [[1], [0], [1]])
    XZ_maskZ = mask(alignmentZ, [[1], [0], [1]])

    XYZ_maskX = mask(alignmentX, [[1], [1], [1]])
    XYZ_maskY = mask(alignmentY, [[1], [1], [1]])
    XYZ_maskZ = mask(alignmentZ, [[1], [1], [1]])

    N = count(X_mask)

    aminoacids = String(id_to_aa.(vec(aminoacid_ids)))
    @assert length(aminoacids) == N

    @assert size(dihedrals, 2) == N
    torsion_angles = Matrix{Float64}(undef, 3, N)
    torsion_angles[1, 2:end] = dihedrals[2, 1:(end-1)]             # ψ
    torsion_angles[3, 2:end] = dihedrals[1, 2:end]                 # ϕ
    # ω
    torsion_angles[2, X_maskX] .= rand(Normal(ideal_omega_angle...), count(X_maskX))
    torsion_angles[2, XY_maskX] .= omegaangles(chainY, standardselector)[XY_maskY]
    torsion_angles[2, XZ_maskX] .= omegaangles(chainZ, standardselector)[XZ_maskZ]
    torsion_angles[2, XYZ_maskX] .= average_angle.(omegaangles(chainY, standardselector)[XYZ_maskY],
                                                   omegaangles(chainZ, standardselector)[XYZ_maskZ])

    bond_angles_Y, bond_lengths_Y = backbone_bond_angles_and_lengths(chainY)
    bond_angles_Z, bond_lengths_Z = backbone_bond_angles_and_lengths(chainZ)

    #TODO write this more succintly
    bond_angles = Matrix{Real}(undef, 3, N)
    for i in collect(1:N)[X_maskX]
        bond_angles[1, i] = rand(WrappedNormal(ideal_bond_angle["CA-C-N", aminoacids[i]]...))[1]
        bond_angles[2, i] = rand(WrappedNormal(ideal_bond_angle["C-N-CA", aminoacids[i]]...))[1]
        bond_angles[3, i] = rand(WrappedNormal(ideal_bond_angle["N-CA-C", aminoacids[i]]...))[1]
    end
    bond_angles[:, XY_maskX] .= bond_angles_Y[:, XY_maskY]
    bond_angles[:, XZ_maskX] .= bond_angles_Z[:, XZ_maskZ]
    #bond_angles[:, XYZ_maskX] .= average_angle.(bond_angles_Y[:, XYZ_maskY], bond_angles_Z[:, XYZ_maskZ])
    bond_angles[:, XYZ_maskX] .= bond_angles_Y[:, XYZ_maskY]


    bond_lengths = Matrix{Real}(undef, 3, N)
    for i in collect(1:N)[X_maskX]
        bond_lengths[1, i] = abs(rand(Normal(ideal_bond_length["C-N",  aminoacids[i]]...)))
        bond_lengths[2, i] = abs(rand(Normal(ideal_bond_length["N-CA", aminoacids[i]]...)))
        bond_lengths[3, i] = abs(rand(Normal(ideal_bond_length["CA-C", aminoacids[i]]...)))
    end
    bond_lengths[:, XY_maskX] .= bond_lengths_Y[:, XY_maskY]
    bond_lengths[:, XZ_maskX] .= bond_lengths_Z[:, XZ_maskZ]
    #bond_lengths[:, XYZ_maskX] .= average_length.(bond_lengths_Y[:, XYZ_maskY],bond_lengths_Z[:, XYZ_maskZ])
    bond_lengths[:, XYZ_maskX] .= bond_lengths_Y[:, XYZ_maskY]

    return build_chain_from_internals(id, aminoacids, torsion_angles, bond_angles, bond_lengths)
end



function build_chain_from_internals(id::String, aminoacids::String,
                                    torsion_angles::Matrix{<:Real}, # ψ       ω       ϕ
                                    bond_angles::Matrix{<:Real},    # CA-C-N  C-N-CA  N-CA-C
                                    bond_lengths::Matrix{<:Real})   # C-N     N-CA    CA-C
    n = length(aminoacids)
    chain = Chain(id)
    chain.res_list = string.(1:n)

    res = Residue(one_to_three[aminoacids[1]], 1, chain)
    chain.residues["1"] = res

    res.atom_list = [" N  ", " CA ", " C  "]
    res.atoms[" N  "] = Atom(1, " N  ", [-4.308, -7.299, 1.075], "N", res)
    res.atoms[" CA "] = Atom(2, " CA ", [-3.695, -7.053, 2.378], "C", res)
    res.atoms[" C  "] = Atom(3, " C  ", [-4.351, -5.868, 3.097], "C", res)

    atom_coords = hcat([-4.308, -7.299, 1.075], [-3.695, -7.053, 2.378], [-4.351, -5.868, 3.097])

    for i ∈ 2:n
        res = Residue(one_to_three[aminoacids[i]], i, chain)
        res.atom_list =  [" N  ", " CA ", " C  "]
        chain.residues[string(i)] = res

        for j ∈ 1:3
            torsion = torsion_angles[j, i] + π
            angle = bond_angles[j, i] + π
            ab = atom_coords[:, j%3+1] - atom_coords[:, (j-1)%3+1]
            bc = atom_coords[:, (j+1)%3+1] - atom_coords[:, j%3+1]
            R = bond_lengths[j, i]
            D = Float64[R*cos(angle),
                        R*cos(torsion)*sin(angle),
                        R*sin(torsion)*sin(angle)]
            bc_hat = normalize(bc)
            normal = normalize(cross(ab, bc_hat))
            M = hcat(bc_hat, cross(normal, bc_hat), normal)
            atom_coords[:, (j-1)%3+1] = M*D + atom_coords[:, (j+1)%3+1]
        end

        res.atoms[" N  "] = Atom(4*i-3, " N  ", vec(atom_coords[:, 1]), "N", res)
        res.atoms[" CA "] = Atom(4*i-2, " CA ", vec(atom_coords[:, 2]), "C", res)
        res.atoms[" C  "] = Atom(4*i-1, " C  ", vec(atom_coords[:, 3]), "C", res)

        prevres = chain.residues[string(i-1)]
        push!(prevres.atom_list, " O  ")

        # a bit of a hacky way to compute the coords of O so that it is planar and roughly
        # in the right area
        C = prevres.atoms[" C  "].coords
        CA = prevres.atoms[" CA "].coords
        N = res.atoms[" N  "].coords

        a = CA - C
        b = N - C
        normal = normalize(cross(a, b))

        o_coords = 0.8 .* [(AngleAxis(-2π/3, normal...) * a).data...] + C
        prevres.atoms[" O  "] = Atom(4*i-4, " O  ", o_coords, "O", prevres)
    end
    return chain
end
