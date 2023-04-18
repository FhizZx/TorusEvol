using BioSequences
using BioStructures

aminoacids = "ARNDCQEGHILKMFPSTWYV-"
id_to_aa(i) = aminoacids[round(Int, i)]

aminoacid_ids = Dict((aminoacids[i], Float(i)) for i ∈ eachindex(aminoacids))
aa_to_id(a) = aminoacid_ids[a]

# Methods to extract internal coordinates data from BioStructures.Chain object
sequence(chain::Chain) = aa_to_id.(collect(string(LongAA(chain, standardselector))))
phi_angles(chain::Chain) = phiangles(chain, standardselector)
psi_angles(chain::Chain) = phiangles(chain, standardselector)
omega_angles(chain::Chain) = omegaangles(chain, standardselector)
calpha_coords(chain::Chain) = coordarray(chain, calphaselector)

# ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄
# Polypeptide

# Each column of data represents the internal coordinates of a residue
# The data in each row is specific by row_names
struct Polypeptide
    data::AbstractMatrix{<:Real}
    row_names::AbstractVector{String}
    chain::BioStructures.Chain
end

# Construct Polypeptide from BioStructures.Chain
function Polypeptide(chain::Chain; primary=true,
                     phi=true, psi=true, omega=false,
                     cartesian=false)
    rows = []
    row_names = []

    primary && (push!(rows, sequence(chain)); push!(row_names, "primary"))
    phi && (push!(rows, phi_angles(chain)); push!(row_names, "ϕ angles"))
    psi && (push!(rows, psi_angles(chain)); push!(row_names, "ψ angles"))
    omega && (push!(rows, omega_angles(chain)); push!(row_names, "ω angles"))
    cartesian && (push!(rows, calpha_coords(chain)); push!(row_names, "Cα x coords");
                  push!(row_names, "Cα y coords"); push!(row_names, "Cα z coords"))

    data = hcat(rows...)'

    return Polypeptide(data, row_names, chain)
end

num_residues(p::Polypeptide) = size(p.data, 2)
num_coords(p::Polypeptide) = size(p.data, 1)
data(p::Polypeptide) = p.data
chain(p::Polypeptide) = p.chain
