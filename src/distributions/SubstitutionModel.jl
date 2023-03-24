using BioAlignments
using LinearAlgebra
using Distributions

# given some rate matrix Q, we can find the transition matrix P(t)
# by exponentiating Q

# i.e. P(t) = exp(t * Q)

# p(i, j) = π(i) * P(t)[i, j]

# p(i, _) = π(i)

# so the substitution model should implement likelihood Methods
# for i -> j as well as equilibrium frequencies
# given some specific substitution matrix? e.g. BLOSUM62

# use bioalignments where they already have implemented aa's etc.

# Q = S diag(Π) where S is symmetric exchangeability matrix,
#        Π is equilibrium frequencies

struct SubstitutionModel{T <: Real}
    statdist::Categorical
    Q::AbstractMatrix{T}
end

function SubstitutionModel(S::AbstractMatrix{T}, Π::AbstractVector{T}) where T <: Real
    S = Symmetric(S)
    S[diagind(S)] .= 0.0
    Q = S * Diagonal(Π)
    Q[diagind(Q)] .= sum(-Q, dims = 2)
    statdist = Categorical(Π)
    return SubstitutionModel(statdist, Q)
end

statdist(model::SubstitutionModel) = model.statdist

function transdist(model::SubstitutionModel, t::Real, aa::AminoAcid)
    P = exp(t * model.Q)
    return Categorical(vec(P[aa, :]))
end
