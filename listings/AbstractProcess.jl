
abstract type AbstractProcess{D <: Distribution} end

# The stationary distribution of the process
function statdist(::AbstractProcess{D})::D end

# The transition distribution at time t starting from x₀
function transdist(::AbstractProcess{D}, t::Real, x₀)::D end

# The state space 𝕊 of the process
function domain(::AbstractProcess{D})::Domain end
