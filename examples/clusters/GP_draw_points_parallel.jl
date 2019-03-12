using Distributed

#addprocs(2) #number of additional processors

using SharedArrays
#@everywhere using DistributedArrays
#@everywhere using ParallelDataTransfer

@everywhere include("GP_functions.jl")
include("GP_emulator.jl")





# To use the optimized GP model to predict at a large number of points drawn from the prior:

#
# Keeping all points:
Random.seed!()

n_accept = 10000
mean_cut, std_cut = Inf, Inf
@time prior_draws_GP_table = predict_model_from_uniform_prior_until_accept_n_points_parallel(params_names, xtrain, ytrain, kernel_SE_ndims, hparams_best, ytrain_err, n_accept; max_mean=mean_cut, max_std=std_cut)

file_name = "GP_files/GP_emulator_points"*string(n_train)*"_meanf"*string(mean_f)*"_prior_draws"*string(n_accept)*"_mean_cut"*string(mean_cut)*"_std_cut"*string(std_cut)*".csv"
f = open(file_name, "w")
println(f, "# hparams: ", hparams_best)
CSV.write(f, prior_draws_GP_table; append=true)
close(f)
#
