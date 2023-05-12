using LinearAlgebra
using Distributions
using DelimitedFiles

const WAG_S = LowerTriangular(map((x) -> x=="" ? 0.0 : x,
                        readdlm("./data/params/WAG_S.csv"; skipblanks=false, dims=(20,20))))

const WAG_Π = normalize!(vec(readdlm("./data/params/WAG_PI.csv")), 1)

WAG_SubstitutionProcess(Π=WAG_Π) = SubstitutionProcess(WAG_S, Π);
