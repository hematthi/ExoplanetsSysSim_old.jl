using ExoplanetsSysSim
using HDF5, JLD

include(joinpath(Pkg.dir(),"ExoplanetsSysSim","examples","hsu_etal_2018", "christiansen_func.jl"))

#stellar_catalog_file_in = joinpath(Pkg.dir("ExoplanetsSysSim"), "data", "q1_q17_dr25_stellar.csv")
#stellar_catalog_file_out = replace(stellar_catalog_file_in,r".csv$",".jld")
#
#stellar_catalog = setup_star_table(ascii(stellar_catalog_file_in))


stellar_catalog_file_in = joinpath(Pkg.dir("ExoplanetsSysSim"), "data", "q1_q16_stellar.csv")
stellar_catalog_file_out = joinpath(Pkg.dir("ExoplanetsSysSim"), "data", "q1_q16_christiansen.jld")

stellar_catalog = setup_star_table_christiansen(ascii(stellar_catalog_file_in))

save(stellar_catalog_file_out,"stellar_catalog", stellar_catalog, "stellar_catalog_usable", ExoplanetsSysSim.StellarTable.usable)








 
