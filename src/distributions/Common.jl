using Distributions

#TODO - implement these as recipes for Plots.jl

# Plot heatmap of the density over 𝕋²
function plotpdf(d::ContinuousDistribution; step=π/100)
    ticks = (-π):step:π
    if length(d) == 2
        grid = hcat([[j, i] for i in ticks, j in ticks]...)
        z = reshape(pdf(d, grid), length(ticks), :)
        heatmap(ticks, ticks, z, size=(400, 400), title="Density",
                xlabel="ϕ angles", ylabel="ψ angles")
    else
        throw("plotting not implemented for d != 2")
    end
end

# Scatter plot of samples from wn
function plotsamples(d::ContinuousDistribution, n_samples)
    samples = rand(d, n_samples)
    scatter(eachrow(samples)...,size=(400,400),
            title="Samples", label="", alpha=0.3)
end

# Fix for categorical logpdf
Distributions.logpdf(d::Categorical, x::AbstractArray{<:Real}) = logpdf.(d, x)

Distributions.logpdf(d::Categorical) = log.(probs(d))

function Distributions.logpdf!(r::AbstractArray{<:Real}, d::Categorical, x::AbstractArray{<:Real})
    r .= logpdf(d, x)
    return r
end
