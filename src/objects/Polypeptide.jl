using BioSequences
using BioStructures
using Bio3DView

const BioModel = BioStructures.Model

# Methods to extract internal coordinates data from BioStructures.Chain object
aa_sequence(chain::Chain) = reshape(aa_to_id.(collect(string(LongAA(chain, standardselector)))), 1, :)
phi_angles(chain::Chain) = reshape(phiangles(chain, standardselector), 1, :)
psi_angles(chain::Chain) = reshape(psiangles(chain, standardselector), 1, :)
function ramachandran_angles(chain::Chain)
    res = vcat(phi_angles(chain), psi_angles(chain))
    res[isnan.(res)] .= 0.0
    res
end
omega_angles(chain::Chain) = reshape(omegaangles(chain, standardselector), 1, :)
calpha_coords(chain::Chain) = coordarray(chain, calphaselector)

# ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄
# Polypeptide

# Each column of data represents the internal coordinates of a residue
# The data in each row is specific by row_names
struct Polypeptide
    data::ObservedChain
    row_names::AbstractVector{String}
    rows::Dict{String, Integer}
    chain::BioStructures.Chain
end

# Construct Polypeptide from BioStructures.Chain
function Polypeptide(chain::BioStructures.Chain; primary=true, ramachandran=true,
                     omega=false, cartesian=false)
    feats = Vector{AbstractArray{Real}}(undef, 0)
    row_names = []

    primary && (push!(feats, aa_sequence(chain)); push!(row_names, "aminoacid"))
    ramachandran && (push!(feats, ramachandran_angles(chain)); push!(row_names, "ramachadran angles"))
    omega && (push!(feats, omega_angles(chain)); push!(row_names, "ω angle"))
    cartesian && (push!(feats, calpha_coords(chain)); push!(row_names, "Cα coords"))

    rows = Dict([(row_names[i], i) for i ∈ eachindex(row_names)])

    return Polypeptide(ObservedChain(feats), row_names, rows, chain)
end

function from_observed_chain(X::ObservedChain)
    return from_primary_dihedrals(Int.(data(X)[1]), data(X)[2])
end

function from_aligned_polypeptide(p::Polypeptide, Y::ObservedChain, M_XY::Alignment; chain_id="X")
    aminoacid_ids = Int.(data(Y)[1])
    dihedrals = data(Y)[2]
    chain = build_biochain_from_aminoacids_dihedrals_alignment(p.chain, M_XY, aminoacid_ids,
                                                               dihedrals; id=chain_id)
    @assert all(aa_sequence(chain) .== aminoacid_ids)
    @assert all((isapprox.(ramachandran_angles(chain),dihedrals; atol=1e-5))[2:end-1])
    return Polypeptide(chain; primary=true, ramachandran=true)
end

function from_triple_alignment(p_Y::Polypeptide, p_Z::Polypeptide,
                               X::ObservedChain, M_XYZ::Alignment; chain_id="X")
    aminoacid_ids = Int.(data(X)[1])
    dihedrals = data(X)[2]
    chain = build_biochain_from_triple_alignment(p_Y.chain, p_Z.chain, M_XYZ,
                                                 aminoacid_ids, dihedrals; id="X")
    @assert all(aa_sequence(chain) .== aminoacid_ids)
    @assert all((isapprox.(ramachandran_angles(chain),dihedrals; atol=1e-5))[2:end-1])
    return Polypeptide(chain; primary=true, ramachandran=true)

end

function from_primary_dihedrals(aminoacid_ids::AbstractArray{<:Integer},
                                dihedrals::AbstractMatrix{<:Real};
                                chain_id="X")
    chain = build_biochain_from_aminoacids_dihedrals(aminoacid_ids, dihedrals; id=chain_id)
    @assert all(aa_sequence(chain) .== aminoacid_ids)
    @assert all((isapprox.(ramachandran_angles(chain),dihedrals; atol=1e-5))[2:end-1])
    return Polypeptide(chain; primary=true, ramachandran=true)
end

function to_file(p::Polypeptide, name::String)
    writepdb("output/pdb/" * name * ".pdb", p.chain)
end

function from_pdb(id::String, chain_id::String; primary=true, ramachandran=true,
                                                omega=false, cartesian=false)
    struc = retrievepdb(id, dir="data/pdb")
    chain = struc[chain_id]
    return Polypeptide(chain; primary=primary, ramachandran=ramachandran, omega=omega, cartesian=cartesian)
end

function from_file(name::String, chain_id::String; primary=true, ramachandran=true,
                   omega=false, cartesian=false)
    struc = read("data/pdb/" * name * ".pdb", PDB)
    chain = struc[chain_id]
    return Polypeptide(chain; primary=primary, ramachandran=ramachandran, omega=omega, cartesian=cartesian)
end

Base.length(p::Polypeptide) = length(p.data)
num_sites(p::Polypeptide) = num_sites(p.data)
num_coords(p::Polypeptide) = num_coords(p.data)
data(p::Polypeptide) = p.data
chain(p::Polypeptide) = p.chain

# Returns the data corresponding to a given coordinate
function get_coord(p::Polypeptide, coord_name::String)
    id = p.rows[coord_name]
    return data(p).data[id]
end

# Render the backbone of a polypeptide using Bio3DView
function render(p::Polypeptide)
    viewstruc(chain(p))
end

# Superimpose the given Polypeptide chains onto one another and render them together
function render(ps...; aligned=true)
    chains = chain.(collect(ps))

    model = BioModel()
    new_chains = similar(chains)

    ref = chains[1]
    new_chains[1] = Chain("1", ref.res_list, ref.residues, model)

    for i ∈ eachindex(chains)[2:end]
        ch = chains[i]
        if aligned
            superimpose!(ch, ref)
        end
        new_chains[i] = Chain(string(i), ch.res_list, ch.residues, model)
    end
    merge!(model.chains, Dict([(string(i), new_chains[i]) for i ∈ eachindex(new_chains)]))
    @info model

    style=Style("cartoon", Dict("opacity"=> 1.0, "color" => "cyan"))

    viewstruc(model)
end

Base.show(io::IO, p::Polypeptide) = print(io, "Polypeptide from chain " * chainid(chain(p)) *
                                              " of protein " * structurename(chain(p)) *
                                              " with " * string(num_sites(p)) * " sites" *
                                              "\nand internal coordinates given by: " * string(p.row_names))
