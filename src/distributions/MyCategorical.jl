using Distributions

struct MyCategorical{V} <: ContinuousDistribution
    values::AbstractVector{V}
    value_ids::Dict{V, Integer}
    c::Categorical
    lp::AbstractVector{<:Real}
end

function MyCategorical(values::AbstractVector, lp::AbstractVector{<:Real})

    MyCategorical(Categorical(exp.(lp)))
end

logpdf(d::MyCategorical) = lp
