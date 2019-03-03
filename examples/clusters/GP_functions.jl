##### To define various GP kernels:

function kernel_SE_1d(xrows::Vector{Float64}, xcols::Vector{Float64}, hparams::Vector{Float64})
    # This function computes the squared exponential kernel in one dimension
    # Assumes the first element in 'hparams' is the amplitude and the second element is the length scale
    sigma_f, lscale = hparams
    sqdist = xrows.^2 .+ transpose(xcols.^2) .- 2*(xrows .* transpose(xcols))
    return (sigma_f^2).*exp.((-1/(2*lscale^2)) .* sqdist)
end

function kernel_SE_ndims(xpoints1::Array{Float64,2}, xpoints2::Array{Float64,2}, hparams::Vector{Float64})
    # This function computes the squared exponential kernel in any number of dimensions, allowing for a different length scale for each dimension (i.e. it is a product of multiple 1d SE kernels)
    # Assumes the first element in 'hparams' is the amplitude and the rest are the length scales
    sigma_f, lscales = hparams[1], hparams[2:end]
    @assert length(lscales) == size(xpoints1)[2] == size(xpoints2)[2]

    sqdist_normed = zeros(size(xpoints1)[1], size(xpoints2)[1])
    for i in 1:length(lscales)
        sqdist_normed += (xpoints1[:,i].^2 .+ transpose(xpoints2[:,i].^2) .- 2*(xpoints1[:,i] .* transpose(xpoints2[:,i])))/(lscales[i]^2)
    end
    return (sigma_f^2).*exp.(-0.5 .* sqdist_normed)
end





##### To define functions that make random draws from a given kernel (with and without data):

function draw_from_prior_given_kernel(xpoints::Array{Float64,2}, kernel::Function, hparams::Vector{Float64}; diag_noise::Float64=1e-5, draws::Integer=1)
    # This function returns a number of draws from a GP with a given kernel, at the points provided
    K_ss = kernel(xpoints, xpoints, hparams)
    L = transpose(cholesky(K_ss + diag_noise*I).U)
    f_prior = L * randn(size(xpoints)[1], draws)

    # Returns a 2-d array where each column is a single draw from the kernel at 'xpoints':
    return f_prior
end

function draw_from_posterior_given_kernel_and_data(xpoints::Array{Float64,2}, xdata::Array{Float64,2}, ydata::Vector{Float64}, kernel::Function, hparams::Vector{Float64}; ydata_err::Vector{Float64}=zeros(length(ydata)), diag_noise::Float64=1e-5, draws::Integer=1)
    # This function computes the GP model given a kernel and a set of data, by computing the mean and standard deviation of the prediction at the points provided, and also making a number of draws from the GP model
    @assert size(xdata)[1] == length(ydata) == length(ydata_err)

    K_ss = kernel(xpoints, xpoints, hparams)
    K_s = kernel(xdata, xpoints, hparams)
    K = kernel(xdata, xdata, hparams)
    var_I = zeros(size(K))
    var_I[diagind(var_I)] = ydata_err.^2
    L = transpose(cholesky(K + var_I).U)

    # To compute the mean at our test points 'xpoints':
    Lk = L \ K_s
    mu = transpose(Lk) * (L \ ydata)

    # To compute the std at our test points 'xpoints':
    s2 = diag(K_ss) .- transpose(sum(Lk.^2, dims=1))
    stdv = sqrt.(s2)

    # To draw samples from the posterior at our test points 'xpoints':
    L = transpose(cholesky(K_ss - (transpose(Lk) * Lk) + diag_noise*I).U) #NOTE: is it OK to add a diagonal term here to help avoid PosDefException error?
    f_posterior = mu .+ (L * randn(size(xpoints)[1], draws))

    # Returns the mean, std, and draws from the posterior at 'xpoints':
    return mu, stdv, f_posterior
end

function compute_kernel_given_data(xdata::Array{Float64,2}, kernel::Function, hparams::Vector{Float64}; ydata_err::Vector{Float64}=zeros(length(ydata)))
    @assert size(xdata)[1] == length(ydata_err)

    K = kernel(xdata, xdata, hparams)
    var_I = zeros(size(K))
    var_I[diagind(var_I)] = ydata_err.^2
    L = transpose(cholesky(K + var_I).U)
    return L
end

function draw_from_posterior_given_precomputed_kernel_from_data(xpoints::Array{Float64,2}, xdata::Array{Float64,2}, ydata::Vector{Float64}, L::Transpose{Float64,UpperTriangular{Float64,Array{Float64,2}}}, kernel::Function, hparams::Vector{Float64}; diag_noise::Float64=1e-5, draws::Integer=1)
    # This function does the same thing as "draw_from_posterior_given_kernel_and_data", but takes a precomputed "L" (cholesky of kernel computed from data) to save time
    @assert size(xdata)[1] == length(ydata)

    K_ss = kernel(xpoints, xpoints, hparams)
    K_s = kernel(xdata, xpoints, hparams)

    # To compute the mean at our test points 'xpoints':
    Lk = L \ K_s
    mu = transpose(Lk) * (L \ ydata)

    # To compute the std at our test points 'xpoints':
    s2 = diag(K_ss) .- transpose(sum(Lk.^2, dims=1))
    stdv = sqrt.(s2)

    # To draw samples from the posterior at our test points 'xpoints':
    L = transpose(cholesky(K_ss - (transpose(Lk) * Lk) + diag_noise*I).U) #NOTE: is it OK to add a diagonal term here to help avoid PosDefException error?
    f_posterior = mu .+ (L * randn(size(xpoints)[1], draws))

    # Returns the mean, std, and draws from the posterior at 'xpoints':
    return mu, stdv, f_posterior
end

function predict_mean_std_on_2d_grid(i::Int64, j::Int64, xi_axis::Vector{Float64}, xj_axis::Vector{Float64}, xmin::Vector{Float64}, xdata::Array{Float64,2}, ydata::Vector{Float64}, L::Transpose{Float64,UpperTriangular{Float64,Array{Float64,2}}}, kernel::Function, hparams::Vector{Float64})
    # This function uses a GP model (i.e. a given kernel and set of hyperparameters), an estimate for the best-fit point 'xmin', and a set of data points, to compute the mean and standard deviation along a 2d grid of parameters for the i-th and j-th parameters (holding all the other parameters fixed at 'xmin')
    @assert size(xdata)[2] == length(xmin)
    @assert size(xdata)[1] == length(ydata)
    dims = length(xmin)
    @assert 1 <= i <= dims
    @assert 1 <= j <= dims

    mean_2d_grid = Array{Float64,2}(undef, length(xi_axis), length(xj_axis))
    std_2d_grid = Array{Float64,2}(undef, length(xi_axis), length(xj_axis))
    for (k,xi) in enumerate(xi_axis)
        for (l,xj) in enumerate(xj_axis)
            xpoint = copy(xmin)
            xpoint[[i,j]] = [xi, xj]
            mean, std, draw = [draw_from_posterior_given_precomputed_kernel_from_data(reshape(xpoint, (1,dims)), xdata, ydata, L, kernel, hparams)[x][1] for x in 1:3]
            mean_2d_grid[k,l] = mean
            std_2d_grid[k,l] = std
        end
    end

    return mean_2d_grid, std_2d_grid
end

function predict_mean_std_on_2d_grids_all_combinations(params_names::Array{Symbol,1}, xmin::Vector{Float64}, xdata::Array{Float64,2}, ydata::Vector{Float64}, kernel::Function, hparams::Vector{Float64}, ydata_err::Vector{Float64}; grid_dims::Int64=20)
    # This function uses a GP model (i.e. a given kernel and set of hyperparameters), an estimate for the best-fit point 'xmin', and a set of data points, to compute the mean and standard deviation along 2d grids of parameters for all possible combinations of the parameters (holding all the other parameters fixed at 'xmin')
    @assert size(xdata)[2] == length(xmin) == length(params_names)
    @assert size(xdata)[1] == length(ydata) == length(ydata_err)
    dims = length(params_names)

    xtrain_mins, xtrain_maxs = findmin(xdata, dims=1)[1], findmax(xdata, dims=1)[1]

    L = compute_kernel_given_data(xdata, kernel, hparams; ydata_err=ydata_err)

    mean_2d_grids = Array{Float64,3}(undef, grid_dims, grid_dims, sum(1:dims-1))
    std_2d_grids = Array{Float64,3}(undef, grid_dims, grid_dims, sum(1:dims-1))
    grid_count = 1 # To index how many 2d grids are computed
    for i in 1:dims
        for j in 1:i-1
            xi_axis = convert(Vector{Float64}, range(xtrain_mins[i], stop=xtrain_maxs[i], length=grid_dims))
            xj_axis = convert(Vector{Float64}, range(xtrain_mins[j], stop=xtrain_maxs[j], length=grid_dims))
            mean_2d_grids[:,:,grid_count], std_2d_grids[:,:,grid_count] = predict_mean_std_on_2d_grid(i, j, xi_axis, xj_axis, xmin, xdata, ydata, L, kernel, hparams)
            grid_count += 1
        end
    end

    return DataFrame(transpose(reshape(mean_2d_grids, (grid_dims, grid_dims*sum(1:dims-1))))), DataFrame(transpose(reshape(std_2d_grids, (grid_dims, grid_dims*sum(1:dims-1)))))
end

function predict_model_at_n_points_from_uniform_prior(params_names::Array{Symbol,1}, xdata::Array{Float64,2}, ydata::Vector{Float64}, kernel::Function, hparams::Vector{Float64}, ydata_err::Vector{Float64}, n_draws::Int64)
    # This function uses a GP model (i.e. a given kernel and set of hyperparameters) and a set of data points, to compute the mean, standard deviation, and a draw from the posterior at each point drawn from a uniform prior for 'n_draws' points
    @assert size(xdata)[2] == length(params_names)
    @assert size(xdata)[1] == length(ydata) == length(ydata_err)
    dims = length(params_names)

    xtrain_mins, xtrain_maxs = findmin(xdata, dims=1)[1], findmax(xdata, dims=1)[1]
    prior_bounds = [(xtrain_mins[i], xtrain_maxs[i]) for i in 1:dims] # To prevent predicting "outside" of the n-dim box of training points

    L = compute_kernel_given_data(xdata, kernel, hparams; ydata_err=ydata_err)

    prior_draws_GP_table = Array{Float64,2}(undef, n_draws, dims+3)
    for i in 1:n_draws
        prior_draws_GP_table[i, 1:dims] = map(j -> prior_bounds[j][1] + (prior_bounds[j][2] - prior_bounds[j][1])*rand(), 1:dims)
        GP_result = draw_from_posterior_given_precomputed_kernel_from_data(reshape(prior_draws_GP_table[i, 1:dims], (1,dims)), xdata, ydata, L, kernel, hparams)
        prior_draws_GP_table[i, dims+1:dims+3] = [GP_result[j][1] for j in 1:3]
    end

    return DataFrame(prior_draws_GP_table, [params_names; [:GP_mean, :GP_std, :GP_posterior_draw]])
end

function predict_model_from_uniform_prior_until_accept_n_points(params_names::Array{Symbol,1}, xdata::Array{Float64,2}, ydata::Vector{Float64}, kernel::Function, hparams::Vector{Float64}, ydata_err::Vector{Float64}, n_accept::Int64; max_mean::Float64=Inf, max_std::Float64=Inf)
    # This function uses a GP model (i.e. a given kernel and set of hyperparameters) and a set of data points, to compute the mean, standard deviation, and a draw from the posterior at points drawn from a uniform prior until 'n_accept' points passing the criteria for mean and std are accepted
    @assert size(xdata)[2] == length(params_names)
    @assert size(xdata)[1] == length(ydata) == length(ydata_err)
    dims = length(params_names)

    xtrain_mins, xtrain_maxs = findmin(xdata, dims=1)[1], findmax(xdata, dims=1)[1]
    prior_bounds = [(xtrain_mins[i], xtrain_maxs[i]) for i in 1:dims] # To prevent predicting "outside" of the n-dim box of training points

    L = compute_kernel_given_data(xdata, kernel, hparams; ydata_err=ydata_err)

    prior_draws_GP_table = Array{Float64,2}(undef, n_accept, dims+3)
    count_accepted = 0
    count_draws = 0
    while count_accepted < n_accept
        prior_draw = map(j -> prior_bounds[j][1] + (prior_bounds[j][2] - prior_bounds[j][1])*rand(), 1:dims)
        GP_result = draw_from_posterior_given_precomputed_kernel_from_data(reshape(prior_draw, (1,dims)), xdata, ydata, L, kernel, hparams)
        count_draws += 1
        if GP_result[1][1] < max_mean && GP_result[2][1] < max_std
            count_accepted += 1
            prior_draws_GP_table[count_accepted, 1:end] = [prior_draw; [GP_result[j][1] for j in 1:3]]
            if count_accepted == 10 # To give an idea of what percentage of points are being accepted
                println("First 10 points accepted after ", count_draws, " draws...")
            end
        end
    end
    println("Accepted ", count_accepted, " points after ", count_draws, " draws.")

    return DataFrame(prior_draws_GP_table, [params_names; [:GP_mean, :GP_std, :GP_posterior_draw]])
end





##### To define a function for calculating the log-marginal likelihood given the data, a kernel and its hyperparameters:

function log_marginal_likelihood(xdata::Array{Float64,2}, ydata::Vector{Float64}, kernel::Function, hparams::Vector{Float64}; ydata_err::Vector{Float64}=zeros(length(ydata)))
    # This function computes the log-marginal likelihood of the GP model given the data, a kernel, and a set of hyperparameter values
    @assert size(xdata)[1] == length(ydata) == length(ydata_err)

    K_f = kernel(xdata, xdata, hparams)
    var_I = zeros(size(K_f))
    var_I[diagind(var_I)] = ydata_err.^2
    K_y = K_f + var_I
    L = transpose(cholesky(K_y).U)

    return -(1/2)*transpose(L \ ydata)*(L \ ydata) - logdet(transpose(L)) - (length(ydata)/2)*log(2*pi) #Note: det(L) == det(transpose(L)) in general
end





##### To define functions for calculating the partial derivatives of various kernels with respect to the hyperparameters, as well as the gradient for the log-marginal likelihood as a function of the hyperparameters:

function dK_dsigma_kernel_SE_ndims(xpoints1::Array{Float64,2}, xpoints2::Array{Float64,2}, hparams::Vector{Float64})
    # This function computes the matrix of partial derivatives of the squared exponential kernel with respect to the amplitude (sigma_f) hyperparameter
    # Assumes the first element in 'hparams' is the amplitude and the rest are the length scales
    sigma_f, lscales = hparams[1], hparams[2:end]
    @assert length(lscales) == size(xpoints1)[2] == size(xpoints2)[2]

    return (2/sigma_f).*kernel_SE_ndims(xpoints1, xpoints2, hparams)
end

function dK_dlj_kernel_SE_ndims(xpoints1::Array{Float64,2}, xpoints2::Array{Float64,2}, hparams::Vector{Float64}, j::Integer)
    # This function computes the matrix of partial derivatives of the squared exponential kernel with respect to the j-th dimension length-scale (lscales[j]) hyperparameter
    # Assumes the first element in 'hparams' is the amplitude and the rest are the length scales
    sigma_f, lscales = hparams[1], hparams[2:end]
    @assert length(lscales) == size(xpoints1)[2] == size(xpoints2)[2]
    @assert 1 <= j <= length(lscales)

    K = kernel_SE_ndims(xpoints1, xpoints2, hparams)
    r2_lj3 = (xpoints1[:,j].^2 .+ transpose(xpoints2[:,j].^2) .- 2*(xpoints1[:,j] .* transpose(xpoints2[:,j])))/(lscales[j]^3)
    return r2_lj3 .* K #this is an element-wise product of the two matrices!
end

function gradient_log_marginal_likelihood(xdata::Array{Float64,2}, ydata::Vector{Float64}, kernel::Function, hparams::Vector{Float64}; ydata_err::Vector{Float64}=zeros(length(ydata)))
    # This function computes the gradient of the log-marginal likelihood as a function of the hyperparameters, given the data, a kernel, its partial derivatives wrt the hyperparameters, and a set of hyperparameter values
    @assert size(xdata)[1] == length(ydata) == length(ydata_err)

    K_f = kernel(xdata, xdata, hparams)
    var_I = zeros(size(K_f))
    var_I[diagind(var_I)] = ydata_err.^2
    K_y = K_f + var_I
    L = transpose(cholesky(K_y).U)
    K_y_inv = transpose(L) \ (L \ I) #inverse of K_y using cholesky decomposition

    matrix = (K_y_inv*ydata)*transpose(K_y_inv*ydata) - K_y_inv
    if kernel == kernel_SE_ndims
        # Assumes the first element in 'hparams' is the amplitude and the rest are the length scales
        sigma_f, lscales = hparams[1], hparams[2:end]
        return [0.5 .* tr(matrix * dK_dsigma_kernel_SE_ndims(xdata, xdata, hparams)); map(i -> 0.5 .* tr(matrix * dK_dlj_kernel_SE_ndims(xdata, xdata, hparams, i)), 1:length(lscales))]
    else
        println("The partial derivatives for this kernel are unknown.")
        return NaN
    end
end