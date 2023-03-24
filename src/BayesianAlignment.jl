
# Pairwise bayesian statistical alignment model

# Using Gibbs sampling for each parameter
# 1. sample time t ~ Exp(time rate)
# 2. sample indel params Λ = {λ - birth rate, μ - death rate, r - extension rate}
# 3. given Λ, sample an alignment 𝑀 using the TKF92 model
# 4. for each column in the alignment, compute the residue and torsional angle likelihoods
