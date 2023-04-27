using BioSequences
using BioStructures
using Bio3DView

aminoacids = "ARNDCQEGHILKMFPSTWYV"
id_to_aa(i) = aminoacids[i]
num_aa = length(aminoacids)

aminoacid_ids = Dict((aminoacids[i], i) for i ∈ eachindex(aminoacids))
aa_to_id(a) = aminoacid_ids[a]

# Methods to extract internal coordinates data from BioStructures.Chain object
aa_sequence(chain::Chain) = reshape(aa_to_id.(collect(string(LongAA(chain, standardselector)))), 1, :)
phi_angles(chain::Chain) = reshape(phiangles(chain, standardselector), 1, :)
psi_angles(chain::Chain) = reshape(psiangles(chain, standardselector), 1, :)
ramachandran_angles(chain::Chain) = vcat(phi_angles(chain), psi_angles(chain))
omega_angles(chain::Chain) = reshape(omegaangles(chain, standardselector), 1, :)
calpha_coords(chain::Chain) = coordarray(chain, calphaselector)

# ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄
# Polypeptide

# Each column of data represents the internal coordinates of a residue
# The data in each row is specific by row_names
struct Polypeptide
    data::ObservedData
    row_names::AbstractVector{String}
    chain::BioStructures.Chain
end

# Construct Polypeptide from BioStructures.Chain
function Polypeptide(chain::Chain; primary=true, ramachandran=true,
                     omega=false, cartesian=false)
    feats = Vector{AbstractArray{Real}}(undef, 0)
    row_names = []

    primary && (push!(feats, aa_sequence(chain)); push!(row_names, "aminoacid"))
    ramachandran && (push!(feats, ramachandran_angles(chain)); push!(row_names, "ϕ, ψ angles"))
    omega && (push!(feats, omega_angles(chain)); push!(row_names, "ω angle"))
    cartesian && (push!(feats, calpha_coords(chain)); push!(row_names, "Cα coords"))

    return Polypeptide(ObservedData(feats), row_names, chain)
end

function from_pdb(id::String, chain_id::String)
    struc = retrievepdb(id, dir="data/pdb")
    chain = struc[chain_id]
    return Polypeptide(chain)
end

Base.length(p::Polypeptide) = length(p.data)
num_sites(p::Polypeptide) = num_sites(p.data)
num_coords(p::Polypeptide) = num_coords(p.data)
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
