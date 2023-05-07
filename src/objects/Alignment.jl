using Base

const Domino = AbstractVector{<:Integer}

struct Alignment <: AbstractVector{Domino}
    ids :: AbstractVector{Integer}
    row_indices :: Dict{Integer, Integer}
    data :: AbstractMatrix{Integer}
end

function Alignment(ids::Vector{<:Integer}, data::AbstractMatrix{<:Integer})
    row_indices = Dict([(ids[i],i) for i ∈ eachindex(ids)])
    # remove null columns
    nonzero_cols = (!).(iszero.(eachcol(data)))
    return Alignment(ids, row_indices, data[:, nonzero_cols])
end

function Alignment(data::AbstractMatrix{<:Integer})
    ids = vec(1:size(data, 1))
    return Alignment(ids, data)
end

# number of sequences is given by the number of rows
num_sequences(a::Alignment) = size(a.data, 1)

# length of alignment is given by the number of columns
size(a::Alignment) = (size(a.data, 2),)

IndexStyle(::Type{<:Alignment}) = IndexLinear()
getindex(a::Alignment, i::Int) = a.data[:, i]
@views setindex!(a::Alignment, v::Domino, i::Int) = a.data[:, i] .= v

row_index(a::Alignment, id::Int) = a.row_indices(id)

function subalignment(a::Alignment, ids::Vector[Int]) :: Alignment
    return Alignment(ids, a.data[a.row_indices.(ids), :])
end

#show(io::IO, a::Alignment) = print(io, "Alignment(" * join(eachrow(), '\n'))

# Combine two alignments using their common "parent" sequence to establish consensus
function combine(parent_id::Int, a1::Alignment, a2::Alignment) :: Alignment
    @assert intersect(ids(a1), ids(a2)) == [parent_id] "The alignments to be combined should only have the parent sequence in common"

    # Which columns in each alignment contain the parent residue
    contains1 = data(a1)[row_index(a1, parent_id), :]
    contains2 = data(a2)[row_index(a2, parent_id), :]
    @assert count(contains1) == count(contains2) "Length of parent in alignments to be combined is not consistent"
    # Number of residues of parent sequence
    parent_length = count(contains1)

    # The alignment matrices excluding the parent
    data1 = data(a1)[1:end .!= row_index(a1, parent_id), :]
    data2 = data(a2)[1:end .!= row_index(a2, parent_id), :]

    empty1 = zeros(length(a1))
    empty2 = zeros(length(a2))

    n = length(a1)
    m = length(a2)
    i = 1, j = 1
    columns = []

    # Match columns which have a parent residue in common
    for _ in 1:parent_length
        while !contains1[i]
            push!(columns, [0; data1[:, i]; empty2])
            i += 1
        end

        while !contains2[j]
            push!(columns, [0; empty1; data2[:, j]])
            j += 1
        end

        push![1; data1[:, i]; data2[:, j]]
    end

    # Add remaining columns from each alignment
    for _ in i:n
        push!(columns, [0; data1[:, i]; empty2])
    end
    for _ in j:m
        push!(columns, [0; empty1; data2[:, j]])
    end

    ids = [parent_id; filter(!=(parent_id), ids(a1)); filter(!=(parent_id), ids(a2))]

    return Alignment(ids, hcat(columns...))

end
