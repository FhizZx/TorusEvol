using BioSequences
using BioStructures

const DihedralAngle = AbstractVector{Real}

const Monomer = Tuple{AminoAcid, DihedralAngle}


# task 1 - protein

# step 1.1 - define protein struct
struct Protein :< Vector{Monomer}
    coords :: Matrix{Real}
end

aminoacids(p::Protein) = first.(p)

dihedrals(p::Protein) = second.(p)

function evolve(p::Protein)

# step 1.2 - read protein from pdb/ chain
# a protein is just a sequence of length N aminoacids
# N-2 x 2 dihedral angles (excluding 1st and last aminoacid)
# 3d coordinates of carbon-alpha, nitrogen, carbon on the backbone
# those are enough to compute 3d coordinates of carbon-alpha from just
# the dihedral angles
function proteinfompdb(pdb chain)

end

# step 1.3 - compute atoms from dihedral angles
function proteinfromdihedrals(dihedrals)
end

# step 1.4 - plot protein
# just plot sequence of N - Ca - C - etc.
# Ca has custom size and colour depending on a table
# render lines between atoms
function plot(protein )
end
