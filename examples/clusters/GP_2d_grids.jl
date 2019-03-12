include("GP_emulator.jl")





# If we want to just make educated guesses for the hyperparameters:
#hparams_best = [5., 0.6, 0.9, 0.4, 0.5, 1., 0.015, 3., 3., 0.1, 0.05]
hparams_best = [1., 1.2, 1.8, 0.8, 1.0, 2.0, 0.03, 6., 6., 0.2, 0.1]





# To use the optimized GP model to predict on a series of 2d grids of points, with the other dimensions (model parameters) set to best-fit values:

grid_dims = 50
@time mean_2d_grids_stacked, std_2d_grids_stacked = predict_mean_std_on_2d_grids_all_combinations(params_names, reshape(xmin_guess, dims), xtrain, ytrain, kernel_SE_ndims, hparams_best, ytrain_err; grid_dims=grid_dims)

file_name = "GP_files/GP_emulator_points"*string(n_train)*"_meanf"*string(mean_f)*"_2d_grids_"*string(grid_dims)*"x"*string(grid_dims)*"_mean.csv"
f = open(file_name, "w")
println(f, "#xlower:"*string(xdata_lower))
println(f, "#xupper:"*string(xdata_upper))
CSV.write(f, mean_2d_grids_stacked; append=true)
close(f)

file_name = "GP_files/GP_emulator_points"*string(n_train)*"_meanf"*string(mean_f)*"_2d_grids_"*string(grid_dims)*"x"*string(grid_dims)*"_std.csv"
f = open(file_name, "w")
println(f, "#xlower:"*string(xdata_lower))
println(f, "#xupper:"*string(xdata_upper))
CSV.write(f, std_2d_grids_stacked; append=true)
close(f)
