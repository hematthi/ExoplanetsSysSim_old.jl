#Pkg.add("ParallelDataTransfer")

addprocs(9) #number of additional processors

@everywhere using ParallelDataTransfer
@everywhere using ExoplanetsSysSim

import DataArrays.skipmissing

@everywhere include("clusters.jl")
@everywhere include("planetary_catalog.jl")

@everywhere sim_param = setup_sim_param_model()





##### To define functions for calculating the distances:

@everywhere function calc_distance(ss1::ExoplanetsSysSim.CatalogSummaryStatistics, ss2::ExoplanetsSysSim.CatalogSummaryStatistics, all_dist::Bool=false, save_dist::Bool=true)
    #This function calculates the total KS distance between two summary statistics (simulated observed catalogs).
    #If 'all_dist=true', the function outputs the individual distances in the distance function.
    #If 'save_dist=true', the function also saves the distances (individual and total) to a file (assuming file 'f' is open for writing).

    global cos_factor

    M_cat_obs1 = ones(Int64,0) #array to be filled with the number of transiting planets in each simulated system for ss1
    M_cat_obs2 = ones(Int64,0) #array to be filled with the number of transiting planets in each simulated system for ss2
    for k in 1:get_int(sim_param,"max_tranets_in_sys")
        append!(M_cat_obs1, k*ones(Int64, ss1.stat["num n-tranet systems"][k]))
        append!(M_cat_obs2, k*ones(Int64, ss2.stat["num n-tranet systems"][k]))
    end

    #To handle empty arrays:
    if sum(M_cat_obs1 .>= 2) < 2 || sum(M_cat_obs2 .>= 2) < 2 #need at least 2 multi-systems per catalog in order to be able to compute AD distances for distributions of ratios of observables
        println("Not enough observed multi-planet systems in one of the catalogs to compute the AD distance.")
        d = ones(Int64,8)*1e6

        println("Distances: ", d, [sum(d)])
        if save_dist
            println(f, "Dist_KS: ", d, [sum(d)])
            println(f, "Dist_AD: ", d, [sum(d)])
        end

        if all_dist
            return d
        else
            return sum(d)
        end
    end

    #To compute the KS distances:
    d_KS = Array{Float64}(8)
    d_KS[1] = abs(ss1.stat["num_tranets"]/(ss1.stat["num targets"]/cos_factor) - ss2.stat["num_tranets"]/(ss2.stat["num targets"]))
    d_KS[2] = ksstats_ints(M_cat_obs1, M_cat_obs2)[5]
    d_KS[3] = ksstats(ss1.stat["P list"], ss2.stat["P list"])[5]
    d_KS[4] = ksstats(ss1.stat["period_ratio_list"], ss2.stat["period_ratio_list"])[5]
    d_KS[5] = ksstats(ss1.stat["duration list"], ss2.stat["duration list"])[5]
    d_KS[6] = ksstats(ss1.stat["duration_ratio_list"], ss2.stat["duration_ratio_list"])[5]
    d_KS[7] = ksstats(ss1.stat["depth list"], ss2.stat["depth list"])[5]
    d_KS[8] = ksstats(ss1.stat["radius_ratio_list"], ss2.stat["radius_ratio_list"])[5]

    #To compute the AD distances:
    d_AD = Array{Float64}(8)
    d_AD[1] = abs(ss1.stat["num_tranets"]/(ss1.stat["num targets"]/cos_factor) - ss2.stat["num_tranets"]/(ss2.stat["num targets"]))
    d_AD[2] = ksstats_ints(M_cat_obs1, M_cat_obs2)[5]
    d_AD[3] = ADstats(ss1.stat["P list"], ss2.stat["P list"])
    d_AD[4] = ADstats(ss1.stat["period_ratio_list"], ss2.stat["period_ratio_list"])
    d_AD[5] = ADstats(ss1.stat["duration list"], ss2.stat["duration list"])
    d_AD[6] = ADstats(ss1.stat["duration_ratio_list"], ss2.stat["duration_ratio_list"])
    d_AD[7] = ADstats(ss1.stat["depth list"], ss2.stat["depth list"])
    d_AD[8] = ADstats(ss1.stat["radius_ratio_list"], ss2.stat["radius_ratio_list"])

    #To print and/or write the distances to file:
    println("KS Distances: ", d_KS, [sum(d_KS)])
    println("AD Distances: ", d_AD, [sum(d_AD)])
    if save_dist
        println(f, "Dist_KS: ", d_KS, [sum(d_KS)])
        println(f, "Dist_AD: ", d_AD, [sum(d_AD)])
    end

    #To return the distances or total distance:
    if all_dist
        return d_KS
        #return d_AD
    else
        return sum(d_KS)
        #return sum(d_AD)
    end
end

@everywhere function calc_distance_Kepler(ss1::ExoplanetsSysSim.CatalogSummaryStatistics, all_dist::Bool=false, save_dist::Bool=true)
    #This function calculates the total KS distance between a population generated by our model and the actual Kepler population (which must be loaded in).
    #If 'all_dist=true', the function outputs the individual distances in the distance function.
    #If 'save_dist=true', the function also saves the distances (individual and total) to a file (assuming file 'f' is open for writing).

    global cos_factor

    M_cat_obs = ones(Int64,0) #array to be filled with the number of transiting planets in each simulated system
    for k in 1:get_int(sim_param,"max_tranets_in_sys")
        append!(M_cat_obs, k*ones(Int64, ss1.stat["num n-tranet systems"][k]))
    end

    if sum(M_cat_obs .>= 2) < 2 #need at least 2 observed multi-systems in order to be able to compute AD distances for distributions of ratios of observables
        println("Not enough observed multi-planet systems in the simulated catalog.")
        d = ones(Int64,8)*1e6

        println("Distances: ", d, [sum(d)])
        if save_dist
            println(f, "Dist_KS: ", d, [sum(d)])
            println(f, "Dist_AD: ", d, [sum(d)])
        end

        if all_dist
            return d
        else
            return sum(d)
        end
    end

    #To compute the KS distances:
    d_KS = Array{Float64}(8)
    d_KS[1] = abs(ss1.stat["num_tranets"]/(ss1.stat["num targets"]/cos_factor) - length(P_confirmed)/N_Kepler_targets)
    d_KS[2] = ksstats_ints(M_cat_obs, M_confirmed)[5]
    d_KS[3] = ksstats(ss1.stat["P list"], P_confirmed)[5]
    d_KS[4] = ksstats(ss1.stat["period_ratio_list"], R_confirmed)[5]
    d_KS[5] = ksstats(ss1.stat["duration list"].*24, t_D_confirmed)[5] #transit durations in simulations are in days, while in the Kepler catalog are in hours
    d_KS[6] = ksstats(ss1.stat["duration_ratio_list"], xi_confirmed)[5]
    d_KS[7] = ksstats(ss1.stat["depth list"], D_confirmed)[5]
    d_KS[8] = ksstats(ss1.stat["radius_ratio_list"].^2, D_ratio_confirmed)[5] #simulations save radius ratios while we computed transit duration ratios from the Kepler catalog

    #To compute the AD distances:
    d_AD = Array{Float64}(8)
    d_AD[1] = abs(ss1.stat["num_tranets"]/(ss1.stat["num targets"]/cos_factor) - length(P_confirmed)/N_Kepler_targets)
    d_AD[2] = ksstats_ints(M_cat_obs, M_confirmed)[5]
    d_AD[3] = ADstats(ss1.stat["P list"], P_confirmed)
    d_AD[4] = ADstats(ss1.stat["period_ratio_list"], R_confirmed)
    d_AD[5] = ADstats(ss1.stat["duration list"].*24, t_D_confirmed) #transit durations in simulations are in days, while in the Kepler catalog are in hours
    d_AD[6] = ADstats(ss1.stat["duration_ratio_list"], xi_confirmed)
    d_AD[7] = ADstats(ss1.stat["depth list"], D_confirmed)
    d_AD[8] = ADstats(ss1.stat["radius_ratio_list"].^2, D_ratio_confirmed) #simulations save radius ratios while we computed transit duration ratios from the Kepler catalog

    #To print and/or write the distances to file:
    println("KS Distances: ", d_KS, [sum(d_KS)])
    println("AD Distances: ", d_AD, [sum(d_AD)])
    if save_dist
        println(f, "Dist_KS: ", d_KS, [sum(d_KS)])
        println(f, "Dist_AD: ", d_AD, [sum(d_AD)])
    end

    #To return the distances or total distance:
    if all_dist
        return d_KS
        #return d_AD
    else
        return sum(d_KS)
        #return sum(d_AD)
    end
end


@everywhere function target_function(active_param::Vector, all_dist::Bool=false, save_dist::Bool=true)
    #This function takes in the values of the active model parameters, generates a simulated observed catalog, and computes the distance function compared to a reference simulated catalog.
    #If 'all_dist=true', the function outputs the individual distances in the distance function.
    #If 'save_dist=true', the function also saves the distances (individual and total) to a file (assuming file 'f' is open for writing).

    println("Active parameter values:", active_param)
    println(f, "Active_params: ", active_param) #if we also want to write the params to file

    global sim_param, summary_stat_ref
    sim_param_here = deepcopy(sim_param)
    ExoplanetsSysSim.update_sim_param_from_vector!(active_param,sim_param_here)
    cat_phys = generate_kepler_physical_catalog(sim_param_here)
    cat_phys_cut = ExoplanetsSysSim.generate_obs_targets(cat_phys,sim_param_here)
    cat_obs = observe_kepler_targets_single_obs(cat_phys_cut,sim_param_here)
    summary_stat = calc_summary_stats_model(cat_obs,sim_param_here)

    dist = calc_distance(summary_stat, summary_stat_ref, all_dist, save_dist)
    return dist
end

@everywhere function target_function_Kepler(active_param::Vector, all_dist::Bool=false, save_dist::Bool=true)
    #This function takes in the values of the active model parameters, generates a simulated observed catalog, and computes the distance function compared to the actual Kepler population.
    #If 'all_dist=true', the function outputs the individual distances in the distance function.
    #If 'save_dist=true', the function also saves the distances (individual and total) to a file (assuming file 'f' is open for writing).

    println("Active parameter values: ", active_param)
    if save_dist
        println(f, "Active_params: ", active_param) #to write the params to file
    end

    global sim_param
    sim_param_here = deepcopy(sim_param)
    ExoplanetsSysSim.update_sim_param_from_vector!(active_param,sim_param_here)
    cat_phys = generate_kepler_physical_catalog(sim_param_here)
    cat_phys_cut = ExoplanetsSysSim.generate_obs_targets(cat_phys,sim_param_here)
    cat_obs = observe_kepler_targets_single_obs(cat_phys_cut,sim_param_here)
    summary_stat = calc_summary_stats_model(cat_obs,sim_param_here)

    dist = calc_distance_Kepler(summary_stat, all_dist, save_dist)
    return dist
end

@everywhere function target_function_weighted(active_param::Vector, all_dist::Bool=false, save_dist::Bool=true)
    #This function takes in the values of the active model parameters, generates a simulated observed catalog, and computes the weighted distance function (assuming an array of weights 'weights' is a global variable) compared to a reference simulated catalog.
    #If 'all_dist=true', the function outputs the individual distances (weighted) in the distance function.
    #If 'save_dist=true', the function also saves the distances (unweighted and weighted, individual and total) to a file (assuming file 'f' is open for writing).

    global weights

    dist = target_function(active_param, true, save_dist)
    weighted_dist = dist./weights
    #used_dist = weighted_dist[1:6] #choose a subset of the distances to pass into the optimizer

    println("Weighted distances: ", weighted_dist, [sum(weighted_dist)])
    if save_dist
        println(f, "Dist_weighted: ", weighted_dist, [sum(weighted_dist)])
    end

    if all_dist
        return weighted_dist
    else
        return sum(weighted_dist)
        #return sum(used_dist)
    end
end

@everywhere function target_function_Kepler_weighted(active_param::Vector, all_dist::Bool=false, save_dist::Bool=true)
    #This function takes in the values of the active model parameters, generates a simulated observed catalog, and computes the weighted distance function (assuming an array of weights 'weights' is a global variable) compared to the actual Kepler population.
    #If 'all_dist=true', the function outputs the individual distances (weighted) in the distance function.
    #If 'save_dist=true', the function also saves the distances (unweighted and weighted, individual and total) to a file (assuming file 'f' is open for writing).

    global weights

    dist = target_function_Kepler(active_param, true, save_dist)
    weighted_dist = dist./weights
    #used_dist = weighted_dist[1:6] #choose a subset of the distances to pass into the optimizer

    println("Weighted distances: ", weighted_dist, [sum(weighted_dist)])
    if save_dist
        println(f, "Dist_weighted: ", weighted_dist, [sum(weighted_dist)])
    end

    if all_dist
        return weighted_dist
    else
        return sum(weighted_dist)
        #return sum(used_dist)
    end
end





##### To start saving the model iterations in the optimization into a file:

@everywhere add_param_fixed(sim_param,"num_targets_sim_pass_one",78005) #9)   # For "observed" data, use a realistic number of targets (after any cuts you want to perform)

model_name = "Clustered_P_R_broken_R_simulated_optimization" #"Clustered_P_R_broken_R_simulated_optimization"
optimization_number = "_random"*ARGS[1] #if want to run on the cluster with random initial active parameters: "_random"*ARGS[1]
max_evals = 3000
file_name = model_name*optimization_number*"_targs"*string(get_int(sim_param,"num_targets_sim_pass_one"))*"_evals"*string(max_evals)

sendto(workers(), max_evals=max_evals, file_name=file_name)

@everywhere f = open(file_name*"_worker"*string(myid())*".txt", "w")
println(f, "# All initial parameters (for estimating stochastic noise in a perfect model):")
write_model_params(f, sim_param)





##### To run the same model multiple times to see how it compares to a simulated catalog with the same parameters:

srand(1234) #to have the same reference catalog and 20 simulated catalogs for calculating the weights

#To generate a simulated catalog to fit to:
cat_phys = generate_kepler_physical_catalog(sim_param)
cat_phys_cut = ExoplanetsSysSim.generate_obs_targets(cat_phys,sim_param)
cat_obs = observe_kepler_targets_single_obs(cat_phys_cut,sim_param)
summary_stat_ref = calc_summary_stats_model(cat_obs,sim_param)

@passobj 1 workers() summary_stat_ref #to send the 'summary_stat_ref' object to all workers

#To simulate more observed planets for the subsequent model generations:
@everywhere add_param_fixed(sim_param,"max_incl_sys",80.0) #degrees; 0 (deg) for isotropic system inclinations; set closer to 90 (deg) for more transiting systems
@everywhere const max_incl_sys = get_real(sim_param,"max_incl_sys")
@everywhere cos_factor = cos(max_incl_sys*pi/180) #factor to divide the number of targets in simulation by to get the actual number of targets needed (with an isotropic distribution of system inclinations) to produce as many transiting systems for a single observer

tic()
active_param_true = make_vector_of_sim_param(sim_param)
println("# True values: ", active_param_true)
println(f, "# Format: Dist: [distances][total distance]")

num_eval = 20 #20
dists_true = zeros(num_eval,8)
for i in 1:num_eval
    dists_true[i,:] = target_function(active_param_true, true)
end
mean_dists = transpose(mean(dists_true,1))[:,] #array of mean distances for each individual distance
mean_dist = mean(sum(dists_true, 2)) #mean total distance
#std_dists = transpose(std(dists_true, 1))[:,] #array of std distances for each individual distance
rms_dists = transpose(sqrt.(mean(dists_true.^2, 1)))[:,] #array of rms (std around 0)  distances for each individual distance
std_dist = std(sum(dists_true, 2)) #std of total distance
rms_dist = sqrt(mean(sum(dists_true, 2).^2)) #rms (std around 0) of total distance; should be similar to the mean total distance

weights = rms_dists #to use the array 'rms_dists' as the weights for the individual distances
sendto(workers(), weights=weights) #to send the weights to all workers
weighted_dists_true = zeros(num_eval,8)
for i in 1:num_eval
    weighted_dists_true[i,:] = dists_true[i,:]./rms_dists
end
mean_weighted_dists = transpose(mean(weighted_dists_true,1))[:,] #array of mean weighted distances for each individual distance
mean_weighted_dist = mean(sum(weighted_dists_true,2)) #mean weighted total distance
std_weighted_dist = std(sum(weighted_dists_true,2)) #std of weighted total distance

println("Mean dists: ", mean_dists)
println("Rms dists: ", rms_dists)
println("Mean weighted dists: ", mean_weighted_dists)
println("Distance using true values: ", mean_dist, " +/- ", std_dist)
println("Weighted distance using true values: ", mean_weighted_dist, " +/- ", std_weighted_dist)
println(f, "Mean: ", mean_dists, [mean_dist])
println(f, "Rms: ", rms_dists, [rms_dist])
println(f, "Mean weighted dists: ", mean_weighted_dists, [mean_weighted_dist])
println(f, "# Distance using true values (default parameter values): ", mean_dist, " +/- ", std_dist)
println(f, "# Weighted distance using true values (default parameter values): ", mean_weighted_dist, " +/- ", std_weighted_dist)
t_elapsed = toc()
println(f, "# elapsed time: ", t_elapsed, " seconds")
println(f, "#")





##### To draw the initial values of the active parameters randomly within a search range:

active_param_keys = ["break_radius", "log_rate_clusters", "log_rate_planets_per_cluster", "mr_power_index", "num_mutual_hill_radii", "power_law_P", "power_law_r1", "power_law_r2", "sigma_hk", "sigma_incl", "sigma_incl_near_mmr", "sigma_log_radius_in_cluster", "sigma_logperiod_per_pl_in_cluster"]
    #["break_radius", "log_rate_clusters", "log_rate_planets_per_cluster", "mr_power_index", "num_mutual_hill_radii", "power_law_P", "power_law_r1", "power_law_r2", "sigma_hk", "sigma_incl", "sigma_incl_near_mmr", "sigma_log_radius_in_cluster", "sigma_logperiod_per_pl_in_cluster"]
active_params_box = [(0.5*ExoplanetsSysSim.earth_radius, 10.*ExoplanetsSysSim.earth_radius), (log(1.), log(5.)), (log(1.), log(5.)), (1., 4.), (3., 20.), (-0.5, 1.5), (-6., 0.), (-6., 0.), (0., 0.1), (0., 5.), (0., 5.), (0.1, 1.0), (0., 0.3)] #search ranges for all of the active parameters
    #[(0.5*ExoplanetsSysSim.earth_radius, 10.*ExoplanetsSysSim.earth_radius), (log(1.), log(5.)), (log(1.), log(5.)), (1., 4.), (3., 20.), (-0.5, 1.5), (-6., 0.), (-6., 0.), (0., 0.1), (0., 5.), (0., 5.), (0.1, 1.0), (0., 0.3)] #search ranges for all of the active parameters

#To randomly draw (uniformly) a value for each active model parameter within its search range:

srand() #to have a random set of initial parameters and optimization run

for (i,param_key) in enumerate(active_param_keys)
    active_param_draw = active_params_box[i][1] + (active_params_box[i][2] - active_params_box[i][1])*rand(1)
    add_param_active(sim_param,param_key,active_param_draw[1])
end
active_param_start = make_vector_of_sim_param(sim_param)

PopSize = length(active_param_true)*4 #length(active_param_true)*4

println("# Active parameters: ", make_vector_of_active_param_keys(sim_param))
println(f, "# Active parameters: ", make_vector_of_active_param_keys(sim_param))
println(f, "# Starting active parameter values: ", active_param_start)
println(f, "# Optimization active parameters search bounds: ", active_params_box)
println(f, "# Method: dxnes")
println(f, "# PopulationSize: ", PopSize)
println(f, "# Format: Active_params: [active parameter values]")
println(f, "# Format: Dist: [distances][total distance]")
println(f, "# Format: Dist_weighted: [weighted distances][total weighted distance]")
println(f, "# Distances used: Dist_KS (all, weighted)") #Edit this line to specify which distances were actually used in the optimizer!

target_function_Kepler_weighted(active_param_start) #to simulate the model once with the drawn parameters and compare to the Kepler population before starting the optimization





##### To use automated optimization routines to optimize the active model parameters:

# Pkg.add("BlackBoxOptim")       # only need to do these once
# Pkg.checkout("BlackBoxOptim")  # needed to get the lastest version
using BlackBoxOptim              # see https://github.com/robertfeldt/BlackBoxOptim.jl for documentation

tic()
opt_result = bboptimize(target_function_Kepler_weighted; SearchRange = active_params_box, NumDimensions = length(active_param_true), Method = :dxnes, PopulationSize = PopSize, MaxFuncEvals = max_evals, TargetFitness = mean_weighted_dist, FitnessTolerance = std_weighted_dist, TraceMode = :verbose, Workers = workers())
t_elapsed = toc()

println(f, "# best_candidate: ", best_candidate(opt_result))
println(f, "# best_fitness: ", best_fitness(opt_result))
println(f, "# elapsed time: ", t_elapsed, " seconds")
@everywhere close(f)


