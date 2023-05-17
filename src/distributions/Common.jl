using Distributions

import Distributions: logpdf


# Fix for categorical logpdf
function domain(d::Categorical)
    K = ncategories(d)
    data = reshape(collect(1:K), 1, K)
    area = K
    return Domain(data, area)
end

function logpdf(ds::AbstractVector{<:Distribution}, X::AbstractVecOrMat)
    if size(X, 1) == 1
        return logpdf.(ds, vec(X))
    else
        return logpdf.(ds, eachcol(X))
    end
end

Distributions.logpdf(d::Categorical, x::AbstractArray{<:Real}) = vec(logpdf.(d, x))

Distributions.logpdf(d::Categorical) = vec(log.(probs(d)))

function Distributions.logpdf!(r::AbstractArray{<:Real}, d::Categorical, x::AbstractArray{<:Real})
    r .= logpdf(d, x)
    return r
end

function Distributions.logpdf(d::ContinuousMultivariateDistribution, X::AbstractMatrix{<: Real})
    r = Array{Float64}(undef, size(X, 2))
    r .= -Inf
    return logpdf!(r, d, X)
end
