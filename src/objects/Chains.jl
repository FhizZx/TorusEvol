abstract type AbstractChain end

# Represents a sequential data object comprised of N sites
# For each site, there are C coordinates (e.g. aminoacid types, dihedral angles)
# the internal 'data' field stores an array for the contents of each coordinate
struct ObservedChain
    data::AbstractVector{AbstractArray}
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

# struct MarginalisedData
#     domains::AbstractVector{AbstractArray}
#     logprobs::AbstractMatrix{<:Real, 3}
# end

# num_sites(x::MarginalisedData) = size(logprobs)
# num_coordinates(x::MarginalisedData) = length(data(x))
# num_regimes(x::MarginalisedData) =
