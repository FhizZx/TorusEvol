using Base
import Base: show, size, IndexStyle, getindex, setindex!

const Domino = AbstractVector{<:Integer}

struct Alignment <: AbstractVector{Domino}
    data::AbstractMatrix{Integer}
    ids::AbstractVector{Integer}
    row_indices::Dict{Integer, Integer}
end

function Alignment(data::AbstractMatrix{<:Integer}, ids::AbstractVector{<:Integer})
    row_indices = Dict([(ids[i],i) for i ∈ eachindex(ids)])
    # remove null columns
    nonzero_cols = (!).(iszero.(eachcol(data)))
    return Alignment(data[:, nonzero_cols], ids, row_indices)
end



function Alignment(data::AbstractMatrix{<:Integer})
    ids = collect(1:size(data, 1))
    return Alignment(data, ids)
end

ids(a::Alignment) = a.ids
data(a::Alignment) = a.data
# number of sequences is given by the number of rows
num_sequences(a::Alignment) = size(a.data, 1)

# length of alignment is given by the number of columns
Base.size(a::Alignment) = (size(a.data, 2),)

Base.IndexStyle(::Type{<:Alignment}) = IndexLinear()
Base.getindex(a::Alignment, i::Int) = a.data[:, i]
@views Base.setindex!(a::Alignment, v::Domino, i::Int) = a.data[:, i] .= v

row_index(a::Alignment, id::Int) = a.row_indices[id]

function mask(a::Alignment, allowed::AbstractArray{<:Domino})
    return BitVector([all(v .∈ allowed) for v ∈ a])
end

function slice(a::Alignment, mask::BitVector)
    Alignment(hcat(a[mask]...))
end

sequence_lengths(a::Alignment) = [count(x -> x==1, a.data[i, :]) for i ∈ 1:num_sequences(a)]

function subalignment(a::Alignment, ids::AbstractVector{<:Integer}) :: Alignment
    new_indices = getindex.(Ref(a.row_indices), ids)
    return Alignment(a.data[new_indices, :], ids)
end

function _char(m)
    v = Array{Char}(undef, size(m, 1), size(m, 2))
    v[m .== 1] .= '#'
    v[m .== 0] .= '-'
    v
end
function _nice_print(m; split_size = 75)
    i = 1
    num_rows = size(m, 1)
    L = size(m, 2)
    res = "\n"
    while i < L
        r = min(L, i+split_size-1)
        for j ∈ 1:num_rows
            row = m[j, i:r]
            for a ∈ row
                res = res * string(a)
            end
            res = res * '\n'
        end
        res = res * '\n'
        i += split_size
    end
    res
end

function show_filled_alignment(a::Alignment, contents...)
    contents = collect(contents)
    m = data(a)
    res = Array{String}(undef, size(m, 1), size(m, 2))
    for i ∈ 1:num_sequences(a)
        res[i, m[i, :] .== 1] .= vec(string.(contents[i]))
    end
    res[m .== 0] .= "-"
    print(_nice_print(res))
end


Base.show(io::IO, a::Alignment) = print(io, _nice_print(_char(a.data)) * '\n')

# Combine two alignments using their common "parent" sequence to establish consensus
function combine(parent_id::Int, a1::Alignment, a2::Alignment) :: Alignment
    @assert intersect(a1.ids, a2.ids) == [parent_id] "The alignments to be combined should only have the parent sequence in common"

    # Which columns in each alignment contain the parent residue
    contains1 = data(a1)[row_index(a1, parent_id), :] .== 1
    contains2 = data(a2)[row_index(a2, parent_id), :] .== 1
    @assert count(contains1) == count(contains2) "Length of parent in alignments to be combined is not consistent"
    # Number of residues of parent sequence
    parent_length = count(contains1)

    # The alignment matrices excluding the parent
    data1 = data(a1)[1:end .!= row_index(a1, parent_id), :]
    data2 = data(a2)[1:end .!= row_index(a2, parent_id), :]

    empty1 = zeros(Int, num_sequences(a1) - 1)
    empty2 = zeros(Int, num_sequences(a2) - 1)

    n = length(a1)
    @assert n == size(data1, 2)
    m = length(a2)
    @assert m == size(data2, 2)
    i = 1; j = 1
    columns = Domino[]

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
        # contains1[i] && contains2[j]

        push!(columns, [1; data1[:, i]; data2[:, j]])
        i += 1
        j += 1
    end

    # Add remaining columns from each alignment
    for k in i:n
        push!(columns, [0; data1[:, k]; empty2])
    end
    for k in j:m
        push!(columns, [0; empty1; data2[:, k]])
    end

    new_ids = [parent_id; filter(!=(parent_id), a1.ids); filter(!=(parent_id), a2.ids)]

    return Alignment(hcat(columns...), new_ids)

end


function glue(a1::Alignment, a2::Alignment) :: Alignment
    n = num_sequences(a1)
    m = num_sequences(a2)
    ids1 = [collect(2:n); 1]
    ids2 = 10000000 .+ collect(1:m); ids2[1] = 1
    x = Alignment(data(a1), ids1)
    y = Alignment(data(a2), ids2)
    M = combine(1, x, y)


    good_data = data(M)[[collect(2:n); 1; collect((n+1):(n+m-1))], :]
    return Alignment(good_data)
end
