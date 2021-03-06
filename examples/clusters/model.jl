if !isdefined(:ExoplanetsSysSim) using ExoplanetsSysSim end

# Code for generating clustered planetary systems (once stable can move to src/planetary_system.jl)

function calc_hill_sphere(a::Float64, mu::Float64)
  a*(mu/3)^(1//3)
end

function calc_mutual_hill_radii{StarT<:StarAbstract}(ps::PlanetarySystem{StarT},pl1::Int64, pl2::Int64)
  mu = (ps.planet[pl1].mass+ps.planet[pl2].mass)/ps.star.mass
  a = 0.5*(ps.orbit[pl1].a+ps.orbit[pl2].a)
  calc_hill_sphere(a,mu)
end

function test_stability_circular(P::Vector{Float64},mass::Vector{Float64},star_mass::Float64, sim_param::SimParam)
   @assert length(P) == length(mass)
   const min_num_mutual_hill_radii = get_real(sim_param,"num_mutual_hill_radii")
   #println("# test_stability_circ: mass= ",mass," star_mass=",star_mass)
   #println("# test_stability_circ: mu= ",mass/star_mass, "; P= ",P)
   #print(  "# (aratio,Delta_H)= ")
   found_instability = false
   order = sortperm(P)
   a2 = semimajor_axis(P[order[1]],star_mass)
   for pl in 1:(length(P)-1)
       a1 = a2   # semimajor_axis(P[order[pl]],star_mass)
       a2 = semimajor_axis(P[order[pl+1]],star_mass)
       a = 0.5*(a1+a2)
       mu = (mass[order[pl]]+mass[order[pl+1]])/star_mass
       mutual_hill_radius = calc_hill_sphere(a,mu)
       #print("(",a2/a1,", ",(a2-a1)/mutual_hill_radius,"), ")
       if ! ( a2-a1  >= min_num_mutual_hill_radii*mutual_hill_radius )
          found_instability = true
          break
       end
   end # loop over neighboring planet pairs within cluster
   #println("# test_stability_circ = ",!found_instability)
   return !found_instability
end

function test_stability(P::Vector{Float64},mass::Vector{Float64},star_mass::Float64, sim_param::SimParam; ecc::Vector{Float64} = zeros(length(P)))
   @assert length(P) == length(mass) == length(ecc)
   const min_num_mutual_hill_radii = get_real(sim_param,"num_mutual_hill_radii")
   found_instability = false
   order = sortperm(P)
   a2 = semimajor_axis(P[order[1]],star_mass)
   for pl in 1:(length(P)-1)
       a1 = a2   # semimajor_axis(P[order[pl]],star_mass)
       a2 = semimajor_axis(P[order[pl+1]],star_mass)
       a = 0.5*(a1+a2)
       mu = (mass[order[pl]]+mass[order[pl+1]])/star_mass
       mutual_hill_radius = calc_hill_sphere(a,mu)
       e1 = ecc[order[pl]]
       e2 = ecc[order[pl+1]]
       if ! ( a2*(1-e2)-a1*(1+e1)  >= min_num_mutual_hill_radii*mutual_hill_radius )
          found_instability = true
          break
       end
   end # loop over neighboring planet pairs within cluster
   return !found_instability
end

function calc_if_near_resonance(P::Vector{Float64})
   @assert issorted(P)   # TODO: OPT: Could remove once know it is safe
   const resonance_width = 0.05   # TODO: FEATURE Make a model parameter?  Would need sim_param
   const resonance_width_factor = 1+resonance_width
   const period_ratios_to_check = [ 2.0, 1.5, 4/3, 5/4 ]
   result = falses(length(P))
   if length(P) >= 2
      for i in 1:(length(P)-1)
          for period_ratio in period_ratios_to_check
              if P[i]*period_ratio <= P[i+1] <= P[i]*period_ratio*resonance_width_factor
                 result[i] = true
                 result[i+1] = true
                 break
              end # if near resonance
          end # period_ratio
      end # planets
   end # at least two planets
   return result
end

function generate_planet_periods_sizes_masses_eccs_in_cluster( star::StarAbstract, sim_param::SimParam; n::Int64 = 1 )  # TODO: IMPORTANT: Make this function work and test before using for science
   @assert n>=1
   const generate_planet_mass_from_radius = get_function(sim_param,"generate_planet_mass_from_radius")
   const generate_sizes = get_function(sim_param,"generate_sizes")
   const generate_e_omega = get_function(sim_param,"generate_e_omega")
   const sigma_ecc::Float64 = haskey(sim_param,"sigma_hk") ? get_real(sim_param,"sigma_hk") : 0.0

   if n==1
      R = generate_sizes(star,sim_param)[1]
      mass = map(r->generate_planet_mass_from_radius(r,sim_param),R)
      P = [1.0] # generate_periods_power_law(star,sim_param)
      eccentricity::Float64,  omega_e::Float64 = generate_e_omega(sigma_ecc)::Tuple{Float64,Float64}
      ecc, omega = [eccentricity], [omega_e]
      return P, R, mass, ecc, omega
   end

   # If reach here, then at least 2 planets in cluster

   mean_R = generate_sizes(star,sim_param)[1]
   const sigma_log_radius_in_cluster = get_real(sim_param,"sigma_log_radius_in_cluster")
   #println("# mean_R = ",mean_R," sigma_log_radius_in_cluster= ",sigma_log_radius_in_cluster)

   const min_radius::Float64 = get_real(sim_param,"min_radius")
   const max_radius::Float64 = get_real(sim_param,"max_radius")
   #Rdist = LogNormal(log(mean_R),sigma_log_radius_in_cluster)
   Rdist = Truncated(LogNormal(log(mean_R),sigma_log_radius_in_cluster), min_radius, max_radius) # if we want clustered planet sizes
   R = rand(Rdist,n)
   #R = generate_sizes(star,sim_param, num_pl=n) # if want non-clustered planet sizes instead

   #println("# Rp = ", R)
   mass = map(r->generate_planet_mass_from_radius(r,sim_param),R)
   #println("# mass = ", mass)

   local ecc, omega
   ecc::Array{Float64,1} = Array{Float64}(n)
   omega::Array{Float64,1} = Array{Float64}(n)
   for i in 1:n
        (ecc[i],  omega[i]) = generate_e_omega(sigma_ecc)::Tuple{Float64,Float64}
   end

   const sigma_logperiod_per_pl_in_cluster = get_real(sim_param,"sigma_logperiod_per_pl_in_cluster")
   const min_period = get_real(sim_param,"min_period")
   const max_period = get_real(sim_param,"max_period")
   log_mean_P = 0.0 # log(generate_periods_power_law(star,sim_param))
   # Note: Currently, drawing all periods within a cluster at once and either keeping or rejecting the whole cluster
   #       Should we instead draw periods one at a time?
   max_period_ratio = max_period/min_period
   #Pdist = LogNormal(log_mean_P,sigma_logperiod_per_pl_in_cluster*n)
   Pdist = Truncated(LogNormal(log_mean_P,sigma_logperiod_per_pl_in_cluster*n), 1/sqrt(max_period_ratio), sqrt(max_period_ratio)) #Truncated unscaled period distribution to ensure that the cluster can fit in the period range [min_period, max_period] after scaling by a period scale
   local P
   found_good_periods = false
   attempts = 0
   max_attempts = 100  # Note: Should this be a parameter?
   while !found_good_periods && attempts<max_attempts
      attempts += 1
      P = rand(Pdist,n)
      if test_stability(P, mass, star.mass, sim_param; ecc=ecc)
        found_good_periods = true
     end
   end # while trying to draw periods

   #println("attempts for cluster: ", attempts)

   if !found_good_periods
      #println("# Warning: Did not find a good set of periods, sizes and masses for one cluster.")
      return fill(NaN,n), R, mass, ecc, omega  # Return NaNs for periods to indicate failed
   end
   return P, R, mass, ecc, omega    # Note can also return earlier if only one planet in cluster or if fail to generate a good set of values
end

function generate_num_planets_in_cluster_poisson(s::Star, sim_param::SimParam)
  const lambda::Float64 = exp(get_real(sim_param,"log_rate_planets_per_cluster"))
  const max_planets_in_cluster::Int64 = get_int(sim_param,"max_planets_in_cluster")
  ExoplanetsSysSim.generate_num_planets_poisson(lambda,max_planets_in_cluster,min_planets=1)
end

function generate_num_clusters_poisson(s::Star, sim_param::SimParam)
  const lambda::Float64 = exp(get_real(sim_param,"log_rate_clusters"))
  const max_clusters_in_sys::Int64 = get_int(sim_param,"max_clusters_in_sys")
  ExoplanetsSysSim.generate_num_planets_poisson(lambda,max_clusters_in_sys)
end

# This version generates clustered planetary systems, adaptation of python code from Matthias He
function generate_planetary_system_clustered(star::StarAbstract, sim_param::SimParam; verbose::Bool = false)  # TODO: Make this function work and test before using for science
  # load functions to use for drawing parameters
  const generate_num_clusters = get_function(sim_param,"generate_num_clusters")
  const generate_num_planets_in_cluster = get_function(sim_param,"generate_num_planets_in_cluster")
  const power_law_P = get_real(sim_param,"power_law_P")
  const min_period = get_real(sim_param,"min_period")
  const max_period = get_real(sim_param,"max_period")

  # To check if we have any giant stars in our catalog
  if star.radius > 5
      log_g = log10((6.67408e-8*star.mass*1.989e33)/((star.radius*6.955e10)^2))
      println("R_star: ", star.radius, "; log(g): ", log_g)
  end

  # Generate a set of periods, planet radii, and planet masses.
  attempt_system = 0
  max_attempts_system = 1 #Note: currently this should not matter; each system is always attempted just once
  local num_pl, Plist, Rlist, masslist, ecclist, omegalist
  valid_system = false
  while !valid_system && attempt_system < max_attempts_system

     # First, generate number of clusters (to attempt) and planets (to attempt) in each cluster
     num_clusters = generate_num_clusters(star,sim_param)::Int64
     #num_clusters = 1 # If we want to test singly-clustered model
     num_pl_in_cluster = map(x->generate_num_planets_in_cluster(star, sim_param)::Int64, 1:num_clusters)
     #num_pl_in_cluster = ones(Int64, num_clusters) ##### If we want a non-clustered model, setting each cluster to have 1 planet is equivalent to having no clusters
     num_pl = sum(num_pl_in_cluster)

     #println("num_clusters: ", num_clusters, " ; num_pl_in_clusters", num_pl_in_cluster)

     if( num_pl==0 )
       return PlanetarySystem(star)
     end
     Plist::Array{Float64,1} = Array{Float64}(num_pl)
     Rlist::Array{Float64,1} = Array{Float64}(num_pl)
     masslist::Array{Float64,1} = Array{Float64}(num_pl)
     ecclist::Array{Float64,1} = Array{Float64}(num_pl)
     omegalist::Array{Float64,1} = Array{Float64}(num_pl)
     @assert num_pl_in_cluster[1] >= 1
     pl_start = 1
     pl_stop = 0
     for c in 1:num_clusters
         pl_stop  += num_pl_in_cluster[c]
         Plist_tmp::Array{Float64,1}, Rlist[pl_start:pl_stop], masslist[pl_start:pl_stop], ecclist[pl_start:pl_stop], omegalist[pl_start:pl_stop] = generate_planet_periods_sizes_masses_eccs_in_cluster(star, sim_param, n = num_pl_in_cluster[c])
         valid_cluster = !any(isnan.(Plist_tmp)) #should this be looking for nans in Plist or Plist_tmp? Was Plist but I think it should be Plist_tmp!
         valid_period_scale = false
         attempt_period_scale = 0
         max_attempts_period_scale = 100
         while !valid_period_scale && attempt_period_scale<max_attempts_period_scale && valid_cluster
             attempt_period_scale += 1

             #period_scale::Array{Float64,1} = ExoplanetsSysSim.generate_periods_power_law(star,sim_param)
             period_scale::Array{Float64,1} = power_law_P!=-1.0 ? ExoplanetsSysSim.draw_power_law(power_law_P, min_period/min(Plist_tmp...), max_period/max(Plist_tmp...), 1) : exp(log(min_period/min(Plist_tmp...)).+rand()*log((max_period/max(Plist_tmp...))/(min_period/min(Plist_tmp...))))
             #Note: this ensures that the minimum and maximum periods will be in the range [min_period, max_period]
             #Warning: not sure about the behaviour when min_period/min(Plist_tmp...) > max_period/max(Plist_tmp...) (i.e. when the cluster cannot fit in the given range)?
             #TODO OPT: could draw period_scale more efficiently by computing the allowed regions in [min_period, max_period] given the previous cluster draws

             Plist[pl_start:pl_stop] = Plist_tmp .* period_scale

             #Note: Checking whether the periods are in the range [min_period, max_period] should be redundant now:
             periods_in_range = true
             for p in Plist_tmp .* period_scale # To check if the periods are still in the range [min_period, max_period] after scaling by period_scale
                if (p < min_period) || (p > max_period)
                    periods_in_range = false
                    println("Pscale = ", period_scale, " , P = ", p, " ; Pmax/Pmin = ", max(Plist_tmp...)/min(Plist_tmp...))
                end
             end

             if test_stability(Plist[1:pl_stop], masslist[1:pl_stop], star.mass, sim_param; ecc=ecclist[1:pl_stop]) && periods_in_range
                valid_period_scale = true
             end
         end  # while !valid_period_scale...

         #if attempt_period_scale > 1
             #println("attempts_period_scale: ", attempt_period_scale)
         #end

         if !valid_period_scale
            Plist[pl_start:pl_stop] = NaN
         end
         pl_start += num_pl_in_cluster[c]
     end # for c in 1:num_clusters
     isnanPlist::Array{Bool,1} = isnan.(Plist::Array{Float64,1})
     if any(isnanPlist)  # If any loop failed to generate valid planets, it should set a NaN in the period list
     keep::Array{Bool,1} = .!(isnanPlist)  # Currently, keeping clusters that could be fit, rather than throwing out entirely and starting from scratch.  Is this a good idea?  Matthias tried the other approach in his python code.
        num_pl = sum(keep)
        Plist = Plist[keep]
        Rlist = Rlist[keep]
        masslist = masslist[keep]
        ecclist = ecclist[keep]
        omegalist = omegalist[keep]
     end
     valid_system = false #this was set to true but I think it should be false?

#println("P: ", Plist)
#println("R: ", Rlist)
#println("M: ", masslist)
#println("ecc: ", ecclist)
#println("omega: ", omegalist)

     #Note: This would be for drawing each cluster separately and then accepting or rejecting the whole lot. By testing for stability before adding each cluster, this last test should be unnecessary.
     if length(Plist) > 0
         if test_stability(Plist, masslist, star.mass, sim_param; ecc=ecclist)
            valid_system = true
         else
            println("Warning: re-attempting system because it fails stability test even though its clusters each pass the test.")
            #Note: this should never happen because we check for stability before adding each cluster, and unstable additions are set to NaN and then discarded
         end
     else
         valid_system = true #this else statement is to allow for systems with no planets to pass
     end

     attempt_system += 1
  end # while !valid_system...

  if attempt_system > 1
    println("attempt_system: ", attempt_system)
  end

  # To print out periods, radii, and masses (for troubleshooting):
  #=
  i_sort = sortperm(Plist)
  Plist_sorted = sort(Plist)
  if length(Plist) > 1
      ratio_list = Plist_sorted[2:end]./Plist_sorted[1:end-1]
      if minimum(ratio_list) < 1.1
         println("P: ", Plist_sorted)
         println(Rlist[i_sort])
         println(masslist[i_sort])
         println(ecclist[i_sort])
         println(omegalist[i_sort])
      end
  end
  =#

  # Now assign orbits with given periods, sizes, and masses.
###const generate_e_omega = get_function(sim_param,"generate_e_omega")
###const sigma_ecc::Float64 = haskey(sim_param,"sigma_hk") ? get_real(sim_param,"sigma_hk") : 0.0
  const sigma_incl = deg2rad(get_real(sim_param,"sigma_incl"))
  const sigma_incl_near_mmr = deg2rad(get_real(sim_param,"sigma_incl_near_mmr"))
  const max_incl_sys = get_real(sim_param,"max_incl_sys")
    pl = Array{Planet}(num_pl)
    orbit = Array{Orbit}(num_pl)
    incl_sys = acos(cos(max_incl_sys*pi/180)*rand()) #acos(rand()) for isotropic distribution of system inclinations; acos(cos(X*pi/180)*rand()) gives angles from X (deg) to 90 (deg)
    idx = sortperm(Plist)       # TODO OPT: Check to see if sorting is significant time sink.  If so, could reduce redundant sortperm
    is_near_resonance = calc_if_near_resonance(Plist[idx])
    for i in 1:num_pl
      # if verbose   println("i=",i," idx=",idx," Plist=",Plist[idx] );     end
      #=
      if haskey(sim_param,"sigma_hk_one") && haskey(sim_param,"sigma_hk_multi")
         sigma_ecc = num_pl == 1 ? get_real(sim_param,"sigma_hk_one") : get_real(sim_param,"sigma_hk_multi")
      end
      =#
###(ecc,  omega) = generate_e_omega(sigma_ecc)::Tuple{Float64,Float64}  # WARNING: Not testing for stability after eccentricites drawn.

      sigma_incl_use = is_near_resonance[i] ? sigma_incl_near_mmr : sigma_incl
      incl_mut = sigma_incl_use*sqrt(randn()^2+randn()^2) # rand(Distributions.Rayleigh(sigma_incl))
      asc_node = 2pi*rand()
      mean_anom = 2pi*rand()
      incl =  incl_mut!=zero(incl_mut) ? acos( cos(incl_sys)*cos(incl_mut) + sin(incl_sys)*sin(incl_mut)*cos(asc_node) ) : incl_sys
      orbit[i] = Orbit(Plist[idx[i]], ecclist[idx[i]], incl, omegalist[idx[i]], asc_node, mean_anom)
      pl[i] = Planet( Rlist[idx[i]],  masslist[idx[i]] )
    end # for i in 1:num_pl

  return PlanetarySystem(star,pl,orbit)
end

## test_generate_planetary_system
function test_generate_planetary_system_clustered() # TODO: Update to test to not reply on generate_kepler_physical_catalog
  sim_param = setup_sim_param_model()
  cat_phys = generate_kepler_physical_catalog(sim_param)
end

