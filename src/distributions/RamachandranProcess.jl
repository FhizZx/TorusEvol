
# Jumping(WrappedDiffusion) over ϕ, ψ angles, which can also handle missing
# ϕ or ψ (as is the case when )
struct RamachandranProcess <: AbstractProcess{ContinuousMultivariateDistribution}
    𝜙𝜓_proc
    𝜙_proc
    ψ_proc
end

function RamachandranProcess(μ_𝜙::Real, μ_𝜓::Real,
                             σ_𝜙::Real, σ_𝜓::Real,
                             α_𝜙::Real, α_𝜓::Real, α_cov::Real,
                             γ::Real)
    𝜙𝜓_proc = Jumping(WrappedDiffusion(μ_𝜙, μ_𝜓, σ_𝜙, σ_𝜓, α_𝜙, α_𝜓, α_cov), γ)
    𝜙_proc = Jumping(WrappedDiffusion(μ_𝜙, σ_𝜙, α_𝜙), γ)
    𝜓_proc = Jumping(WrappedDiffusion(μ_𝜓, σ_𝜓, α_𝜓), γ)
    return RamachandranProcess(𝜙𝜓_proc, 𝜙_proc, 𝜓_proc)
end
