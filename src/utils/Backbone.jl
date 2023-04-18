# Much of this code was adapted from Michael Golden's code
using BioStructures
using DataStructures
using LinearAlgebra
using Rotations

one_to_three = Dict('A' => "ALA", 'C' => "CYS", 'D' => "ASP", 'E' => "GLU", 'F' => "PHE",
                    'G' => "GLY", 'H' => "HIS", 'I' => "ILE", 'L' => "LEU", 'K' => "LYS",
                    'M' => "MET", 'N' => "ASN", 'P' => "PRO", 'Q' => "GLN", 'R' => "ARG",
                    'S' => "SER", 'T' => "THR", 'W' => "TRP", 'V' => "VAL", 'Y' => "TYR")
one_to_three = DefaultDict("XXX", one_to_three)

# TODO find a source for these values
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

function backbone_bond_angles_and_lengths(chain::Chain)
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


function Residue(name::AbstractString, number::Integer, ch::Chain)
    return BioStructures.Residue(name, number, ' ', false, ch)
end

function Atom(serial::Int, name::String, coords::Vector{Float64},
              element::String, residue::StructuralElement)
    return BioStructures.Atom(serial, name, ' ', coords, 1.0, 13.0, element, "  ", residue)
end

function build_chain_from_alignment(chain::Chain, alignment, Y)
    indicesX = findall(∈([MATCH, DELETE]), alignment)
    indicesY = findall(∈([MATCH, INSERT]), alignment)

    n = length(indicesY)

    alignmentX = alignment[indicesX]
    alignmentY = alignment[indicesY]
    indicesIY = findall(==(INSERT), alignmentY)
    indicesMX = findall(==(MATCH), alignmentX)
    indicesMY = findall(==(MATCH), alignmentY)

    aminoacids = String(id_to_aa.(Y[1, :]))

    torsion_angles = Matrix{Real}(undef, 3, n)
    torsion_angles[1, 2:end] = Y[3, 1:(end-1)]      # ψ
    torsion_angles[3, 2:end] = Y[2, 2:end]          # ϕ
    torsion_angles[2, indicesIY] .= rand(Normal(ideal_omega_angle...), length(indicesIY))
    torsion_angles[2, indicesMY] .= omegaangles(chain, standardselector)[indicesMX]

    bond_angles_X, bond_lengths_X = backbone_bond_angles_and_lengths(chain)

    #TODO write this more succintly
    bond_angles = Matrix{Real}(undef, 3, n)
    for i in indicesIY
        bond_angles[1, i] = rand(WrappedNormal(ideal_bond_angle["CA-C-N", aminoacids[i]]...))[1]
        bond_angles[2, i] = rand(WrappedNormal(ideal_bond_angle["C-N-CA", aminoacids[i]]...))[1]
        bond_angles[3, i] = rand(WrappedNormal(ideal_bond_angle["N-CA-C", aminoacids[i]]...))[1]
    end
    bond_angles[:, indicesMY] .= bond_angles_X[:, indicesMX]

    bond_lengths = Matrix{Real}(undef, 3, n)
    for i in indicesIY
        bond_lengths[1, i] = abs(rand(Normal(ideal_bond_length["C-N",  aminoacids[i]]...)))
        bond_lengths[2, i] = abs(rand(Normal(ideal_bond_length["N-CA", aminoacids[i]]...)))
        bond_lengths[3, i] = abs(rand(Normal(ideal_bond_length["CA-C", aminoacids[i]]...)))
    end
    bond_lengths[:, indicesMY] .= bond_lengths_X[:, indicesMX]

    return build_chain_from_internals("Y", aminoacids, torsion_angles, bond_angles, bond_lengths)
end

function build_chain_from_internals(id::String, aminoacids::String,
                                    torsion_angles::Matrix{Real}, # ψ       ω       ϕ
                                    bond_angles::Matrix{Real},    # CA-C-N  C-N-CA  N-CA-C
                                    bond_lengths::Matrix{Real})   # C-N     N-CA    CA-C
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
