using BioAlignments
using LinearAlgebra
using Distributions
using DelimitedFiles

WAG_S = LowerTriangular(map((x) -> x=="" ? 0.0 : x,
                        readdlm("./data/params/WAG_S.csv"; skipblanks=false, dims=(20,20))))

WAG_Π = normalize!(vec(readdlm("./data/params/WAG_PI.csv")), 1)



WAG_SubstitutionProcess = SubstitutionProcess(WAG_S, WAG_Π);
