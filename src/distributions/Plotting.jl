using Plots
using StatsPlots
#TODO - implement these as recipes for Plots.jl

# Plot heatmap of the density over 𝕋²
function plotpdf(d::ContinuousDistribution; step=π/100)
    ticks = (-π):step:π
    if length(d) == 2
        grid = hcat([[j, i] for i in ticks, j in ticks]...)
        z = reshape(pdf(d, grid), length(ticks), :)
        heatmap(ticks, ticks, z, size=(400, 400), title="Density",
                xlabel="ϕ angles", ylabel="ψ angles", legend=:none)
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

function anim_tpd_2D(𝚯, θ₀)
    times=[0.00001, 0.0001, 0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7,0.8,
           0.9, 1.0, 1.2, 1.4, 1.7, 2.0, 3.0, 4.0, 5.0, 6.0, 10.0, 15.0, 20.0,
           60.0, 100.0, 180.0]

    anim = @animate for t ∈ times
        println(t)
        plotpdf(transdist(𝚯, t, θ₀))
    end

    gif(anim, "tpd.gif", fps=5)
end

function anim_logtpd_2D(𝚯, θ₀)
    times=[0.0, 0.00001, 0.0001, 0.001, 0.01, 0.1]

    anim = @animate for t ∈ times
        println(t)
        plot_logtpd_2D(𝚯, t, θ₀)
    end

    gif(anim, "images/logtpd.gif", fps=1)
end


using Random

function diffuse(𝚯, θ₀; Δt=0.05, n_steps=250)
    θ = θ₀
    samples = zeros(length(θ), n_steps)
    anim = @animate for i ∈ 1:n_steps
        θ = rand(transdist(𝚯, Δt, θ))
        samples[:, i] = θ
        l = max(1, i - 2)
        scatter(samples[1, l:i], samples[2, l:i], size=(400,400), xlims=(-π, π), ylims=(-π, π), title="WN DIffusion, i = " * string(i), label="", alpha=1.0)
    end
    gif(anim, "images/diffusion.gif", fps=10)

end

function diffuse2(𝚯, θ₀; Δt=0.05, n_steps=250)
    θ = θ₀
    anim = @animate for i ∈ 1:n_steps
        t = transdist(𝚯, Δt, θ)
        θ = rand(t)
        plotpdf(t; step=π/5)
    end
    gif(anim, "images/diffusion.gif", fps=10)

end
