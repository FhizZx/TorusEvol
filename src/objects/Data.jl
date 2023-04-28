
# Represents a sequential data object comprised of N sites
# For each site, there are C coordinates (e.g. aminoacid types, dihedral angles)
# the internal 'data' field stores an array for the contents of each coordinate
struct ObservedData
    data::AbstractVector{AbstractArray}
    N::Integer
end

function ObservedData(data::AbstractVector{<:AbstractArray{<:Real}})
    N = size(data[1], 2)
    @assert all(N .== size.(data, Ref(2))) "Dimensions of feature vectors don't match: " *string(size.(data, Ref(2)))
    return ObservedData(data, N)
end

data(x::ObservedData) = x.data                # C vectors of contents for N sites
num_sites(x::ObservedData) = x.N              # N
num_coords(x::ObservedData) = length(data(x)) # C

# Create new data object with a subset of the coordinates and sites
function slice(x::ObservedData, site_inds, coord_inds)
    new_data = [v[:, site_inds] for v ∈ data(x)[coord_inds]]
    return ObservedData(new_data)
end

# Print distribution parameters
show(io::IO, x::ObservedData) = print(io, "ObservedData(" *
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
