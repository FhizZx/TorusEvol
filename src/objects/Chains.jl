abstract type AbstractChain end

# Represents a sequential data object comprised of N sites
# For each site, there are C coordinates (e.g. aminoacid types, dihedral angles)
# the internal 'data' field stores an array for the contents of each coordinate
struct ObservedChain <: AbstractChain
    data::AbstractVector{<:AbstractArray}
    N::Integer
end

function ObservedChain(data::AbstractVector{<:AbstractArray{<:Real}})
    N = size(data[1], 2)
    @assert all(N .== size.(data, Ref(2))) "Dimensions of feature vectors don't match: " *string(size.(data, Ref(2)))
    return ObservedChain(data, N)
end

data(x::ObservedChain) = x.data                # C vectors of contents for N sites
num_sites(x::ObservedChain) = x.N              # N
num_coords(x::ObservedChain) = length(data(x)) # C

# Create new data object with a subset of the coordinates and sites
function slice(x::ObservedChain, site_inds, coord_inds)
    new_data = [v[:, site_inds] for v ∈ data(x)[coord_inds]]
    return ObservedChain(new_data)
end

# Print distribution parameters
show(io::IO, x::ObservedChain) = print(io, "ObservedChain(" *
                                          "\nnum sites: " * string(num_sites(x)) *
                                          "\nnum coords: " * string(num_coords(x)) *
                                          "\n)")

# __________________________________________________________________________________________

struct HiddenChain <: AbstractChain
    domains::AbstractVector{<:AbstractArray{<:Real, 2}}
    logprobs::AbstractVector{<:AbstractArray{<:Real, 3}}
    # size(logprobs)[c] = E x N_Ω_c x N
    N::Integer
    E::Integer
end

num_coords(x::HiddenChain) = length(x.logprobs)
num_sites(x::HiddenChain) = x.N
num_regimes(x::HiddenChain) = x.E
logprobs(X::HiddenChain) = X.logprobs
domains(X::HiddenChain) = X.domains
