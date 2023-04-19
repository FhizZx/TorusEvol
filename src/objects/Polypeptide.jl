using BioSequences
using BioStructures
using Bio3DView

aminoacids = "ARNDCQEGHILKMFPSTWYV-"
id_to_aa(i) = aminoacids[round(Int, i)]

aminoacid_ids = Dict((aminoacids[i], Float64(i)) for i ∈ eachindex(aminoacids))
aa_to_id(a) = aminoacid_ids[a]

# Methods to extract internal coordinates data from BioStructures.Chain object
sequence(chain::Chain) = aa_to_id.(collect(string(LongAA(chain, standardselector))))
phi_angles(chain::Chain) = phiangles(chain, standardselector)
psi_angles(chain::Chain) = psiangles(chain, standardselector)
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

    primary && (push!(rows, sequence(chain)); push!(row_names, "aminoacid"))
    phi && (push!(rows, phi_angles(chain)); push!(row_names, "ϕ angle"))
    psi && (push!(rows, psi_angles(chain)); push!(row_names, "ψ angle"))
    omega && (push!(rows, omega_angles(chain)); push!(row_names, "ω angle"))
    cartesian && (push!(rows, calpha_coords(chain)); push!(row_names, "Cα x coord");
                  push!(row_names, "Cα y coord"); push!(row_names, "Cα z coord"))

    data = hcat(rows...)'

    return Polypeptide(data, row_names, chain)
end

function from_pdb(id::String, chain_id::String)
    struc = retrievepdb(id, dir="data/pdb")
    chain = struc[chain_id]
    return Polypeptide(chain)
end

Base.length(p::Polypeptide) = length(p.data)
num_residues(p::Polypeptide) = size(p.data, 2)
num_coords(p::Polypeptide) = size(p.data, 1)
data(p::Polypeptide) = p.data
chain(p::Polypeptide) = p.chain


# Render the backbone of a polypeptide using Bio3DView
function render(p::Polypeptide)
    viewstruc(chain(p))
end

# Superimpose the given Polypeptide chains onto one another and render them together
function render_aligned(ps...)
    chains = chain.(collect(ps))

    model = Model()
    new_chains = similar(chains)

    ref = chains[1]
    new_chains[1] = Chain("1", ref.res_list, ref.residues, model)

    for i ∈ eachindex(chains)[2:end]
        ch = chains[i]
        superimpose!(ch, ref)
        new_chains[i] = Chain(string(i), ch.res_list, ch.residues, model)
    end
    merge!(model.chains, Dict([(string(i), new_chains[i]) for i ∈ eachindex(new_chains)]))
    @info model

    style=Style("cartoon", Dict("opacity"=> 1.0, "color" => "cyan"))

    viewstruc(model)
end

Base.show(io::IO, p::Polypeptide) = print(io, "Polypeptide from chain " * chainid(chain(p)) *
                                              " of protein " * structurename(chain(p)) *
                                              " with " * string(num_residues(p)) * " residues" *
                                              "\nand internal coordinates given by: " * string(p.row_names))
