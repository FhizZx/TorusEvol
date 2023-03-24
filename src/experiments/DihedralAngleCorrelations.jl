using BioStructures
using Plots
using CSV

# globins = readdlm("data/globin_ids.csv", ',', String)
rat = readdlm("data/rat.csv", ',', String)

angles0(chain) = copy(transpose(hcat(filter(x -> !isnan(x), phiangles(chain)),
                     filter(x -> !isnan(x), psiangles(chain)))))

angles(chain) = copy(transpose(hcat(filter(x -> !isnan(x), phiangles(chain))[1:end-1],
                     filter(x -> !isnan(x), psiangles(chain))[2:end])))



function experiment(proteins)
    plane_angles = []
    res_angles = []
    for p in proteins
        try
            downloadpdb(p) do path
                struc = read(path, PDB)
                for chain in struc
                        push!(plane_angles, angles0(chain))
                        push!(res_angles, angles(chain))
                end
            end
        catch e
            print(".")
        end
    end

    plane_data = hcat(plane_angles...)
    res_data = hcat(res_angles...)

    display(histogram2d(eachrow(plane_data)...,size=(400,400),
            nbinsx=20, nbinsy=20,
            title="Plane Toroidal Angles Correlations", label="",
            xlabel="ϕ angles", ylabel="ψ angles"))

    display(histogram2d(eachrow(res_data)...,size=(400,400),
            nbinsx=20, nbinsy=20,
            title="Residue Toroidal Angles Correlations", label="",
            xlabel="ϕ angles", ylabel="ψ angles"))

end

experiment(rat[1:30])
