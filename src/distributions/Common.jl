using Distributions

#TODO - implement these as recipes for Plots.jl

# Plot heatmap of the density over 𝕋²
function plotpdf(d::ContinuousDistribution; step=π/100)
    ticks = (-π):step:π
    if length(wn) == 2
        grid = hcat([[j, i] for i in ticks, j in ticks]...)
        z = reshape(pdf(wn, grid), length(ticks), :)
        heatmap(ticks, ticks, z, size=(400, 400), title="WN Density",
                xlabel="ϕ angles", ylabel="ψ angles")
    else
        throw("plotting not implemented for d != 2")
    end
end

# Scatter plot of samples from wn
function plotsamples(d::ContinuousDistribution, n_samples)
    samples = rand(wn, n_samples)
    scatter(eachrow(samples)...,size=(400,400),
            title="Samples", label="", alpha=0.3)
end
