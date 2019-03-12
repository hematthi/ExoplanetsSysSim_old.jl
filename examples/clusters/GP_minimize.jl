include("GP_emulator.jl")





# If we want to just make educated guesses for the hyperparameters:
#hparams_best = [5., 0.6, 0.9, 0.4, 0.5, 1., 0.015, 3., 3., 0.1, 0.05]
hparams_best = [1., 1.2, 1.8, 0.8, 1.0, 2.0, 0.03, 6., 6., 0.2, 0.1]





"""
Optimize the parameters of the model (i.e. find the point that minimizes the function) by using a dataset and a GP model (i.e. a given kernel and set of hyperparameters) to minimize the GP mean prediction.
"""
function optimize_model_GP_mean(xmin_guess::Array{Float64,2}, xdata::Array{Float64,2}, ydata::Vector{Float64}, L::Transpose{Float64,UpperTriangular{Float64,Array{Float64,2}}}, kernel::Function, hparams::Vector{Float64}; ydata_err::Vector{Float64}=zeros(length(ydata)))
    @assert size(xdata)[1] == length(ydata) == length(ydata_err)
    dims = size(xmin_guess,2)

    xdata_lower = minimum(xdata, dims=1)
    xdata_upper = maximum(xdata, dims=1)

    @time result1 = optimize(xpoint -> draw_from_posterior_given_precomputed_kernel_from_data(reshape(xpoint, (1,dims)), xdata, ydata, L, kernel, hparams)[1][1], xmin_guess, BFGS()) # unconstrained, no gradient

    #@time result2 = optimize(xpoint -> draw_from_posterior_given_precomputed_kernel_from_data(reshape(xpoint, (1,dims)), xtrain, ytrain, L, kernel_SE_ndims, hparams_best)[1][1], xdata_lower, xdata_upper, xmin_guess, Fminbox(GradientDescent())) # constrained, no gradient

    println("Best (xpoint, GP_mean): ", (Optim.minimizer(result1), Optim.minimum(result1)))
    #println("Best (xpoint, GP_mean): ", (Optim.minimizer(result2), Optim.minimum(result2)))
    result = result1
    xmin, ymin = Optim.minimizer(result), Optim.minimum(result)
    return xmin, ymin
end

"""
Optimize the parameters of the model repeatedly (i.e. find the point that minimizes the function) by using a GP model (i.e. a given kernel and set of hyperparameters) and different iterations of the training data to minimize the GP mean prediction.
"""
function optimize_model_GP_mean_multiple_datasets(iterations::Int64, n_train::Int64, params_names::Array{Symbol,1}, xmin_guess::Array{Float64,2}, xdata_all::Array{Float64,2}, ydata_all::Vector{Float64}, kernel::Function, hparams::Vector{Float64}; ydata_err_all::Vector{Float64}=zeros(length(ydata_all)))
    @assert size(xdata_all)[1] == length(ydata_all) == length(ydata_err_all)
    @assert n_train < length(ydata_all)
    dims = length(xmin_guess)

    min_points_table = Array{Float64,2}(undef, iterations, dims+1)
    @time for i in 1:iterations
        i_train = Random.randperm(length(ydata_all))[1:n_train]
        xtrain = xdata_all[i_train,:]
        ytrain = ydata_all[i_train]
        ytrain_err = ydata_err_all[i_train]
        L = compute_kernel_given_data(xtrain, kernel, hparams; ydata_err=ytrain_err)
        min_points_table[i,1:dims], min_points_table[i,dims+1] = optimize_model_GP_mean(xmin_guess, xtrain, ytrain, L, kernel, hparams; ydata_err=ytrain_err)
    end

    return DataFrame(min_points_table, [params_names; :ymin])
end





# To run an optimizer using the best GP model in order to find the point that minimizes the function (i.e. GP mean prediction):

xmin_guess = Statistics.median(xdata, dims=1)
println("Median data point: ", xmin_guess)

# To optimize for the minimum once using a dataset:
L = compute_kernel_given_data(xtrain, kernel_SE_ndims, hparams_best; ydata_err=ytrain_err)
xmin, ymin = optimize_model_GP_mean(xmin_guess, xtrain, ytrain, L, kernel_SE_ndims, hparams_best; ydata_err=ytrain_err)

# To optimize for the minimum multiple times using random iterations of the dataset:
#
iterations = 10
n_train = 2000
min_table = optimize_model_GP_mean_multiple_datasets(iterations, n_train, params_names, xmin_guess, params_array, dist_array.-mean_f, kernel_SE_ndims, hparams_best; ydata_err_all=0.8 .*ones(length(dist_array)))
#
