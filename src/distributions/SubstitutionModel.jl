using BioAlignments
using LinearAlgebra
using Distributions
using DelimitedFiles
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

aminoacids = "ARNDCQEGHILKMFPSTWYV-"

id_to_aa(i) = aminoacids[round(Int, i)]

aminoacid_ids = Dict((aminoacids[i], i) for i ∈ eachindex(aminoacids))

aa_to_id(a) = aminoacid_ids[a]

WAG_S = LowerTriangular(map((x) -> x=="" ? 0.0 : x,
                        readdlm("./data/params/WAG_S.csv"; skipblanks=false, dims=(20,20))))

WAG_Π = normalize!(vec(readdlm("./data/params/WAG_PI.csv")), 1)

function SubstitutionProcess(S::LowerTriangular, Π::AbstractVector)
    S = Symmetric(S, :L)
    S[diagind(S)] .= 0.0
    Q = S * Diagonal(Π)
    Q[diagind(Q)] .= sum(-Q, dims = 2)
    #TODO normalize rate of replacement so that the mean is 1
    return CTMC(Π, Q)
end

WAG_SubstitutionProcess = SubstitutionProcess(WAG_S, WAG_Π);
