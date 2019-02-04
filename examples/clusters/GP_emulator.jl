##### GP regression to fit to the results of the model optimization (fit GP model to the distance of the model compared to the Kepler data as a function of the model parameters)

using LinearAlgebra
using Statistics
using Plots
using Random
using DataFrames
using CSV

include("GP_functions.jl")

make_plots = false





# To load in the data:

data_table_original = CSV.read("GP_files/Active_params_distances_table_best100000_every10.txt", delim=" ", allowmissing=:none)
data_table_recomputed = CSV.read("GP_files/Active_params_distances_recomputed_table_best100000_every10.txt", delim=" ", allowmissing=:none)

params_names = names(data_table_recomputed)[1:10]
dists_names = names(data_table_recomputed)[11:23]

params_array_original = convert(Matrix, data_table_original[1:end, params_names])
dist_array_original = convert(Array, data_table_original[1:end, :dist_tot_weighted])

params_array = convert(Matrix, data_table_recomputed[1:end, params_names])
dist_array = convert(Array, data_table_recomputed[1:end, :dist_tot_weighted])
dims = length(params_names)





# To plot the data:

if make_plots
    fig = histogram([dist_array_original dist_array], fillalpha=0.5, xlabel="Distance", ylabel="Number of points", label=["Best distances during optimization", "Recomputed distances"])
    display(fig)

    fig2 = histogram(reshape(dist_array, (div(length(dist_array), 5), 5)), fillalpha=0.5, xlabel="Distance", ylabel="Number of points")
    display(fig2)
end




# To choose a subset of the data for training and cross-validation sets:

Random.seed!(1234) # If we want the same set of training and CV points every time

n_data = 4000

xdata = params_array[1:n_data, 1:end]
mean_f = 15.
ydata = dist_array[1:n_data] .- mean_f
ydata_err = 0.8 .*ones(n_data)

n_train = div(n_data, 2)
n_check = n_data - n_train
randperm_data = Random.randperm(n_data)
i_train, i_check = randperm_data[1:n_train], randperm_data[n_train+1:end]
xtrain, xcheck = xdata[i_train,1:dims], xdata[i_check,1:dims]
ytrain, ycheck = ydata[i_train], ydata[i_check]
ytrain_err, ycheck_err = ydata_err[i_train], ydata_err[i_check]





# To optimize the hyperparameters, plot the resulting GP model, and assess the fit of the model:
using Optim

#=
hparams_lower = [1e-1; 0.01*ones(dims)]
hparams_upper = [10.; 10*ones(dims)]
hparams_guess = [1.; ones(dims)]

Random.seed!()

#@time result1 = optimize(hparams -> -log_marginal_likelihood(xtrain, ytrain, kernel_SE_ndims, hparams; ydata_err=ytrain_err), hparams_guess, BFGS()) #unconstrained, no gradient

@time result2 = optimize(hparams -> -log_marginal_likelihood(xtrain, ytrain, kernel_SE_ndims, hparams; ydata_err=ytrain_err), hparams -> -gradient_log_marginal_likelihood(xtrain, ytrain, kernel_SE_ndims, hparams; ydata_err=ytrain_err), hparams_guess, BFGS(); inplace = false) #unconstrained, with gradient

#@time result3 = optimize(hparams -> -log_marginal_likelihood(xtrain, ytrain, kernel_SE_ndims, hparams; ydata_err=ytrain_err), hparams_lower, hparams_upper, hparams_guess, Fminbox(GradientDescent())) #constrained, no gradient

#@time result4 = optimize(hparams -> -log_marginal_likelihood(xtrain, ytrain, kernel_SE_ndims, hparams; ydata_err=ytrain_err), hparams -> -gradient_log_marginal_likelihood(xtrain, ytrain, kernel_SE_ndims, hparams; ydata_err=ytrain_err), hparams_lower, hparams_upper, hparams_guess, Fminbox(GradientDescent()); inplace = false) #constrained, with gradient

#println("Best (hparams, log_p): ", (Optim.minimizer(result1), -Optim.minimum(result1)))
println("Best (hparams, log_p): ", (Optim.minimizer(result2), -Optim.minimum(result2)))
#println("Best (hparams, log_p): ", (Optim.minimizer(result3), -Optim.minimum(result3)))
#println("Best (hparams, log_p): ", (Optim.minimizer(result4), -Optim.minimum(result4)))

result = result2
hparams_best = Optim.minimizer(result)
log_p_best = -Optim.minimum(result)
println("Best (hparams, log_p): ", (hparams_best, log_p_best))
=#

#hparams_best = [3.21594, -0.297917, -0.39717, 210.038, 48.4862, 1.78456, -201.024, 1.06141, 3.36472, 42.6757, -152.565] # 'n_data = 2000', 'mean_f = 15.'
#hparams_best = [1.10519, 0.234453, -0.477346, 0.135724, 0.378424, 1.73346, -0.0134949, 0.73254, 1.43884, 0.0956221, -0.0308756] # 'n_data = 2000', 'mean_f = 18.'
#hparams_best = [-1.76993, -0.272425, -0.507301, 0.327124, -0.383005, 1.45841, -0.023842, 1.16629, 3.04358, 0.137717, -0.0675467] # 'n_data = 2000', 'mean_f = 20.'
#hparams_best = [1718.25, 480.398, 857.24, 793.841, 1249.0, 4191.11, 0.465114, 4457.53, 5526.83, -21.5877, 1.61524] # 'n_data = 2000', 'mean_f = 50'

hparams_best = [25.6287, -5.27561, -1.40939, -10.0505, 2.40215, 5.89228, 1.08675, -3.03994, 12.2616, -13.3928, 3.42326] # 'n_data = 4000', 'mean_f = 15.'; OR
#hparams_best = [-1.68505, 7.56876e6, 1.16479, 1.46191e7, 1.07758, 7.2004e7, -6.35001e6, 3.76568e7, 7.90991e7, 1.02692, -0.108351] # 'n_data = 4000', 'mean_f = 15.'

# To predict at the training points:
mu_train, stdv_train, f_posterior_train = draw_from_posterior_given_kernel_and_data(xtrain, xtrain, ytrain, kernel_SE_ndims, hparams_best; ydata_err = ytrain_err)

# To predict at the checking points (cross-validation):
mu_cv, stdv_cv, f_posterior_cv = draw_from_posterior_given_kernel_and_data(xcheck, xtrain, ytrain, kernel_SE_ndims, hparams_best; ydata_err = ytrain_err)

# To plot histograms of the residuals of the mean prediction compared to the data:
ydiff_train = mu_train - ytrain
ydiff_cv = mu_cv - ycheck

if make_plots
    fig3 = histogram([ydiff_train ydiff_cv], fillalpha=0.5, xlabel="Mean prediction - Data", ylabel="Number of points", label=["Training", "Cross-validation"])

    fig4 = scatter([ytrain], [mu_train], markersize=1, xlabel="Data", ylabel="Mean prediction", label=["Training", "Cross-validation"])
    plot!(range(0, stop=maximum(ycheck), length=100), range(0, stop=maximum(ycheck), length=100), label="Perfect prediction")

    fig5 = scatter([ydiff_train], [stdv_train], markersize=1, xlabel="Mean prediction - Data", ylabel="Uncertainty of prediction", label=["Training", "Cross-validation"])

    fig6 = scatter([mu_train], [stdv_train], markersize=1, xlabel="Mean prediction", ylabel="Uncertainty of prediction", label=["Training", "Cross-validation"])

    fig3_6 = plot(fig3,fig4,fig5,fig6, layout=(2,2), guidefontsize=8, legend=true, legendfontsize=4)
    display(fig3_6)
end





# To run an optimizer using the best GP model in order to find the point that minimizes the function (i.e. GP mean prediction):

xdata_lower = minimum(xdata, dims=1)
xdata_upper = maximum(xdata, dims=1)
xmin_guess = Statistics.median(xdata, dims=1)
println("Median data point: ", xmin_guess)

L = compute_kernel_given_data(xtrain, kernel_SE_ndims, hparams_best; ydata_err=ytrain_err)

Random.seed!()

@time result1 = optimize(xpoint -> draw_from_posterior_given_precomputed_kernel_from_data(reshape(xpoint, (1,dims)), xtrain, ytrain, L, kernel_SE_ndims, hparams_best)[1][1], xmin_guess, BFGS()) # unconstrained, no gradient

@time result2 = optimize(xpoint -> draw_from_posterior_given_precomputed_kernel_from_data(reshape(xpoint, (1,dims)), xtrain, ytrain, L, kernel_SE_ndims, hparams_best)[1][1], xdata_lower, xdata_upper, xmin_guess, Fminbox(GradientDescent())) # constrained, no gradient

println("Best (xpoint, GP_mean): ", (Optim.minimizer(result1), Optim.minimum(result1)))
println("Best (xpoint, GP_mean): ", (Optim.minimizer(result2), Optim.minimum(result2)))





#=
# To use the optimized GP model to predict at a large number of points drawn from the prior:

n_draws = 1000

@time prior_draws_GP_table = predict_model_at_n_points_from_uniform_prior_fast(params_names, xtrain, ytrain, kernel_SE_ndims, hparams_best, ytrain_err, n_draws)
@time prior_draws_GP_table = predict_model_at_n_points_from_uniform_prior(params_names, xtrain, ytrain, kernel_SE_ndims, hparams_best, ytrain_err, n_draws)

file_name = "GP_files/GP_emulator_points"*string(n_train)*"_meanf"*string(mean_f)*"_prior_draws"*string(n_draws)*".csv"
CSV.write(file_name, prior_draws_GP_table)
=#