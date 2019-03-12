##### GP regression to fit to the results of the model optimization (fit GP model to the distance of the model compared to the Kepler data as a function of the model parameters)

include("GP_functions.jl")

using Plots
using Optim

make_plots = true





# To load in the data:

#data_path = "GP_files"
data_path = "/Users/Matthias/Documents/GradSchool/Eric_Ford_Research/ACI/Model_Optimization/Julia_v0.7/Kepler_catalog_optimization/q1q17_dr25_gaia_fgk_stars80006/Clustered_P_R/f_high_incl_low_incl_mmr/Fit_rate_mult_P_Pratios_D_Dratios_dur_durratios_mmr/Some11_params_KSweightedrms/lc_lp_0p5_5_alphaP_-2_1_alphaR1_R2_-6_0_ecc_0_0p1_incl_inclmmr_0_90_sigmaR_0_0p5_sigmaP_0_0p3/Fixed_Rbreak3_Ncrit8/targs400030_maxincl0_maxiters5000/sigma_i_greater_sigma_i_mmr"
#data_path = "/Users/Matthias/Documents/GradSchool/Eric_Ford_Research/ACI/Model_Optimization/Julia_v0.7/Kepler_catalog_optimization/q1q17_dr25_gaia_fgk_stars80006/Clustered_P/f_high_incl_low_incl_mmr/Fit_rate_mult_P_Pratios_D_Dratios_dur_durratios_mmr/Some10_params_KSweightedrms/Fixed_Rbreak3_Ncrit8/lc_lp_0p5_5_alphaP_-2_1_alphaR1_R2_-6_0_ecc_0_0p1_incl_inclmmr_0_90_sigmaP_0_0p3/targs400030_maxincl0_maxiters5000/sigma_i_greater_sigma_i_mmr"
#data_path = "/Users/Matthias/Documents/GradSchool/Eric_Ford_Research/ACI/Model_Optimization/Julia_v0.7/Kepler_catalog_optimization/q1q17_dr25_gaia_fgk_stars80006/Non_Clustered/f_high_incl_low_incl_mmr/Fit_rate_mult_P_Pratios_D_Dratios_dur_durratios_mmr/Some8_params_KSweightedrms/Fixed_Rbreak3_Ncrit8/lc_1_10_alphaP_-2_1_alphaR1_R2_-6_0_ecc_0_0p1_incl_inclmmr_0_90/targs400030_maxincl0_maxiters5000/sigma_i_greater_sigma_i_mmr"

data_table_original = CSV.read(joinpath(data_path,"Active_params_distances_table_best100000_every10.txt"), delim=" ", allowmissing=:none)
data_table_recomputed = CSV.read(joinpath(data_path,"Active_params_distances_recomputed_table_best100000_every10.txt"), delim=" ", allowmissing=:none)
#data_table_recomputed = CSV.read("GP_files/Active_params_distances_table_best388529_every1.txt", delim=" ", allowmissing=:none)[1:388000,:]

params_names = names(data_table_recomputed)[1:11]
dists_names = names(data_table_recomputed)[12:end]

params_array_original = convert(Matrix, data_table_original[1:end, params_names])
dist_array_original = convert(Array, data_table_original[1:end, :dist_tot_weighted])

params_array = convert(Matrix, data_table_recomputed[1:end, params_names])
dist_array = convert(Array, data_table_recomputed[1:end, :dist_tot_weighted])
dims = length(params_names)





# To plot the data:

if make_plots
    fig = histogram([dist_array_original dist_array], fillalpha=0.5, xlabel="Distance", ylabel="Number of points", label=["Best distances during optimization", "Recomputed distances"])
    display(fig)

    fig1 = histogram(reshape(dist_array, (div(length(dist_array), 5), 5)), fillalpha=0.5, xlabel="Distance", ylabel="Number of points")
    display(fig1)
end




# To choose a subset of the data for training and cross-validation sets:

Random.seed!(1234) # If we want the same set of training and CV points every time

n_data = 4000

#i_data = Random.randperm(sum(dist_array .< Inf))[1:n_data]
i_data = 1:n_data
xdata = params_array[i_data, 1:end]
mean_f = 22.
ydata = dist_array[i_data] .- mean_f
ydata_err = 0.8 .*ones(n_data)

n_train = div(n_data, 2)
xtrain, xcheck, ytrain, ycheck, ytrain_err, ycheck_err = split_data_training_cv(xdata, ydata, ydata_err; n_train=n_train)

if make_plots
    fig2 = histogram(ydata, fillalpha=0.5, xlabel="Distance - mean_f", ylabel="Number of points")
    display(fig2)
end





# To optimize the hyperparameters, plot the resulting GP model, and assess the fit of the model:

hparams_guess = [1.; ones(dims)]
#hparams_best, log_p_best = optimize_hparams_with_MLE(hparams_guess, xtrain, ytrain, kernel_SE_ndims; ydata_err=ytrain_err)

# Clustered_P_R:
hparams_best = [8.03106, -0.2036, 0.344798, 0.415796, -0.484306, -0.757346, 1.81563, -950.991, -36.9642, 0.851163, -0.159388, 395.641] # Clustered_P_R with 'n_data = 2000', 'mean_f = 30.'
#hparams_best = [10., 0.3, 1.2, 1.5, 1.2, 1.8, 3., 0.03, 60., 1., 0.15, 0.1]

# Clustered_P:
#hparams_best = [10., 0.3, 0.8, 1., 0.5, 0.8, 2., 0.03, 60., 1., 0.1]

# Non_Clustered:
#hparams_best = 4 .*[1.25, 0.05, 0.25, 0.1, 0.8, 1.5, 0.01, 60., 0.5]






# To predict at the training points:
mu_train, stdv_train, f_posterior_train = draw_from_posterior_given_kernel_and_data(xtrain, xtrain, ytrain, kernel_SE_ndims, hparams_best; ydata_err = ytrain_err)

# To predict at the checking points (cross-validation):
mu_cv, stdv_cv, f_posterior_cv = draw_from_posterior_given_kernel_and_data(xcheck, xtrain, ytrain, kernel_SE_ndims, hparams_best; ydata_err = ytrain_err)

# To plot histograms of the residuals of the mean prediction compared to the data:
ydiff_train = mu_train - ytrain
ydiff_cv = mu_cv - ycheck

if make_plots
    fig3 = histogram([ydiff_train ydiff_cv], fillalpha=0.5, xlabel="Mean prediction - Data", ylabel="Number of points", label=["Training", "Cross-validation"])

    fig4 = scatter([ytrain ycheck], [mu_train mu_cv], markersize=1, xlabel="Data", ylabel="Mean prediction", label=["Training", "Cross-validation"])
    plot!(range(0, stop=maximum(ycheck), length=100), range(0, stop=maximum(ycheck), length=100), label="Perfect prediction")

    fig5 = scatter([ydiff_train], [stdv_train], markersize=1, xlabel="Mean prediction - Data", ylabel="Uncertainty of prediction", label=["Training", "Cross-validation"])

    fig6 = scatter([mu_train], [stdv_train], markersize=1, xlabel="Mean prediction", ylabel="Uncertainty of prediction", label=["Training", "Cross-validation"])

    fig3_6 = plot(fig3,fig4,fig5,fig6, layout=(2,2), guidefontsize=8, legend=true, legendfontsize=4)
    display(fig3_6)
end
