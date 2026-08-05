# Load required libraries
library(R2jags)
library(coda)
library(ggplot2)
library(dplyr)
library(tidyr)
library(purrr)
library(loo)

# Set seed for reproducibility
set.seed(123)

# ==============================================================================
# SIMULATION STUDY: 1000 REPETITIONS (WITHOUT MEASUREMENT ERROR IN DGP,
# BUT INFERENCE INCORRECTLY ASSUMES MEASUREMENT ERROR)
# ==============================================================================

n_sims <- 1000  # Number of simulations
N <- 500        # Time points
K <- 6          # Number of pollutants

# True parameters (extended to 6 pollutants)
beta_true <- c(0.1, -0.2, 0.3, -0.15, 0.25, -0.1)
alpha_true <- -12
gamma_true <- -0.01
rho_true <- c(0.7, 0.65, 0.8, 0.75, 0.68, 0.72)
sigma_proc_true <- c(0.5, 0.6, 0.4, 0.55, 0.45, 0.5)
# NO measurement error in DGP (sigma_me_true = 0)

# Storage arrays for results
beta_storage <- array(NA, dim = c(n_sims, K))
rho_storage <- array(NA, dim = c(n_sims, K))
sigma_proc_storage <- array(NA, dim = c(n_sims, K))
sigma_me_storage <- array(NA, dim = c(n_sims, K))  # For measurement error SD
alpha_storage <- rep(NA, n_sims)
gamma_storage <- rep(NA, n_sims)

# Storage for CI coverage
beta_ci_lower <- array(NA, dim = c(n_sims, K))
beta_ci_upper <- array(NA, dim = c(n_sims, K))
rho_ci_lower <- array(NA, dim = c(n_sims, K))
rho_ci_upper <- array(NA, dim = c(n_sims, K))
sigma_me_ci_lower <- array(NA, dim = c(n_sims, K))
sigma_me_ci_upper <- array(NA, dim = c(n_sims, K))

# Storage for WAIC values
waic_storage <- rep(NA, n_sims)

# Progress tracking
cat("Starting", n_sims, "simulations (NO measurement error in DGP, but INFERENCE assumes ME)...\n")
start_time <- Sys.time()

# ==============================================================================
# JAGS MODEL: INCORRECTLY SPECIFIED (ASSUMES MEASUREMENT ERROR)
# Treats observed data as error-prone measurements of latent states
# ==============================================================================
misspecified_model <- "
model {
  # State-space model with measurement error
  for (k in 1:K) {
    # AR(1) process for latent states
    Z[1,k] ~ dnorm(0, tau_proc[k] * (1 - rho[k]^2))
    for (t in 2:N) {
      Z[t,k] ~ dnorm(rho[k] * Z[t-1,k], tau_proc[k])
    }
    
    # Measurement error model (incorrectly assuming error exists)
    for (t in 1:N) {
      Y_obs[t,k] ~ dnorm(Z[t,k], tau_me[k])
    }
    
    # Priors
    rho_raw[k] ~ dbeta(15, 5)
    rho[k] <- 2 * rho_raw[k] - 1
    tau_proc[k] ~ dgamma(2, 1)
    sigma_proc[k] <- 1/sqrt(tau_proc[k])
    
    # Prior for measurement error SD (will be estimated >0 even though true=0)
    sigma_me[k] ~ dunif(0, 5)
    tau_me[k] <- 1/(sigma_me[k]^2)
    
    beta[k] ~ dnorm(0, 0.01)
  }
  
  # Outcome model - uses latent states
  for (t in 1:N) {
    Y_out[t] ~ dpois(lambda[t])
    log(lambda[t]) <- alpha + inprod(beta[1:K], Z[t,1:K]) + 
                      gamma * temperature[t] + log(population[t])
    
    # Log-likelihood for WAIC calculation
    log_lik[t] <- logdensity.pois(Y_out[t], lambda[t])
  }
  
  # Priors for outcome model
  alpha ~ dnorm(0, 0.01)
  gamma ~ dnorm(0, 0.01)
}
"

# Run simulations
for(sim in 1:n_sims) {
  if(sim %% 10 == 0) cat("Completed simulation", sim, "of", n_sims, "\n")
  
  # Generate temperature data (fixed across simulations for consistency)
  set.seed(100 + sim)
  temperature <- 15 + 5 * sin(2 * pi * (1:N) / 52) + rnorm(N, 0, 1)
  population <- rep(100000, N)
  
  # ============================================================================
  # DATA GENERATING PROCESS (WITHOUT MEASUREMENT ERROR)
  # ============================================================================
  
  # Generate AR(1) latent states (these are directly observed - NO measurement error)
  Z_true <- matrix(NA, nrow = N, ncol = K)
  for(k in 1:K) {
    Z_true[1, k] <- rnorm(1, 0, sigma_proc_true[k] / sqrt(1 - rho_true[k]^2))
    for(t in 2:N) {
      Z_true[t, k] <- rho_true[k] * Z_true[t-1, k] + 
        rnorm(1, 0, sigma_proc_true[k])
    }
  }
  
  # NO measurement error: observed = true latent
  Y_obs <- Z_true  # Direct observation without error
  
  # Generate outcomes using TRUE latent pollutants
  log_lambda <- alpha_true + Z_true %*% beta_true + 
    gamma_true * temperature + log(population)
  Y_out <- rpois(N, exp(log_lambda))
  
  # ============================================================================
  # INFERENCE: MISSPECIFIED MODEL (assumes measurement error exists)
  # The model incorrectly treats Y_obs as error-prone measurements
  # ============================================================================
  
  # Prepare data for JAGS
  data_list <- list(
    N = N, K = K,
    Y_obs = Y_obs,  # Treated as error-prone measurements (incorrect)
    Y_out = Y_out,
    temperature = temperature,
    population = population
  )
  
  # Initial values for measurement error parameters (starting positive)
  init_func <- function() {
    list(
      sigma_me = rep(0.5, K),  # Start with positive values
      rho_raw = rep(0.8, K),
      tau_proc = rep(1, K),
      beta = rep(0, K),
      alpha = -10,
      gamma = 0
    )
  }
  
  # Fit JAGS model (reduced iterations for speed)
  tryCatch({
    fit <- jags(data_list, 
                inits = init_func,
                parameters.to.save = c("beta", "alpha", "gamma", "rho", 
                                       "sigma_proc", "sigma_me", "log_lik"),
                model.file = textConnection(misspecified_model),
                n.chains = 2,
                n.iter = 8000,  # Increased iterations due to more complex model
                n.burnin = 2000,
                n.thin = 6,
                quiet = TRUE)
    
    # Extract results
    results <- fit$BUGSoutput$summary
    
    # Store point estimates (mean)
    for(k in 1:K) {
      beta_storage[sim, k] <- results[paste0("beta[", k, "]"), "mean"]
      rho_storage[sim, k] <- results[paste0("rho[", k, "]"), "mean"]
      sigma_proc_storage[sim, k] <- results[paste0("sigma_proc[", k, "]"), "mean"]
      sigma_me_storage[sim, k] <- results[paste0("sigma_me[", k, "]"), "mean"]
      
      # Store CI bounds
      beta_ci_lower[sim, k] <- results[paste0("beta[", k, "]"), "2.5%"]
      beta_ci_upper[sim, k] <- results[paste0("beta[", k, "]"), "97.5%"]
      rho_ci_lower[sim, k] <- results[paste0("rho[", k, "]"), "2.5%"]
      rho_ci_upper[sim, k] <- results[paste0("rho[", k, "]"), "97.5%"]
      sigma_me_ci_lower[sim, k] <- results[paste0("sigma_me[", k, "]"), "2.5%"]
      sigma_me_ci_upper[sim, k] <- results[paste0("sigma_me[", k, "]"), "97.5%"]
    }
    
    alpha_storage[sim] <- results["alpha", "mean"]
    gamma_storage[sim] <- results["gamma", "mean"]
    
    # Calculate WAIC from log-likelihood
    log_lik_samples <- fit$BUGSoutput$sims.list$log_lik
    
    lppd <- sum(log(colMeans(exp(log_lik_samples))))
    p_waic <- sum(apply(log_lik_samples, 2, var))
    waic_storage[sim] <- -2 * (lppd - p_waic)
    
  }, error = function(e) {
    cat("Error in simulation", sim, ":", e$message, "\n")
  })
}

end_time <- Sys.time()
cat("\nCompleted in", difftime(end_time, start_time, units = "mins"), "minutes\n")

# ==============================================================================
# RESULTS ANALYSIS
# ==============================================================================

# Remove any failed simulations
valid_sims <- complete.cases(beta_storage[,1])
n_valid <- sum(valid_sims)
cat("\nValid simulations:", n_valid, "of", n_sims, "\n")

# Function to compute summary statistics
compute_summary <- function(estimates, true_values, param_name) {
  bias <- colMeans(estimates, na.rm = TRUE) - true_values
  mse <- colMeans((estimates - matrix(true_values, nrow = nrow(estimates), 
                                      ncol = length(true_values), byrow = TRUE))^2, 
                  na.rm = TRUE)
  rmse <- sqrt(mse)
  
  sd_est <- apply(estimates, 2, sd, na.rm = TRUE)
  
  results <- data.frame(
    Parameter = param_name,
    Pollutant = 1:length(true_values),
    True = true_values,
    Mean_Est = colMeans(estimates, na.rm = TRUE),
    Bias = bias,
    Empirical_SD = sd_est,
    RMSE = rmse,
    MSE = mse
  )
  return(results)
}

# Beta results (likely biased due to over-specification)
beta_summary <- compute_summary(beta_storage[valid_sims, ], beta_true, "beta")
cat("\n=== BETA ESTIMATES (Misspecified model - assuming ME when none exists) ===\n")
print(beta_summary)

# Rho results
rho_summary <- compute_summary(rho_storage[valid_sims, ], rho_true, "rho")
cat("\n=== RHO ESTIMATES ===\n")
print(rho_summary)

# Measurement error results (should be near 0 but likely overestimated)
sigma_me_true <- rep(0, K)
sigma_me_summary <- compute_summary(sigma_me_storage[valid_sims, ], sigma_me_true, "sigma_me")
cat("\n=== MEASUREMENT ERROR SD ESTIMATES (True = 0) ===\n")
print(sigma_me_summary)

# Function to compute coverage
compute_coverage <- function(estimates, ci_lower, ci_upper, true_values) {
  coverage <- matrix(NA, nrow = nrow(estimates), ncol = ncol(estimates))
  for(j in 1:ncol(estimates)) {
    coverage[,j] <- (ci_lower[,j] <= true_values[j]) & (ci_upper[,j] >= true_values[j])
  }
  coverage_percent <- colMeans(coverage, na.rm = TRUE) * 100
  return(list(
    coverage_percent = coverage_percent,
    n_covered = colSums(coverage, na.rm = TRUE),
    n_total = colSums(!is.na(coverage))
  ))
}

# Coverage probabilities (likely lower than 95% due to over-specification)
beta_coverage_results <- compute_coverage(beta_storage[valid_sims, ], 
                                          beta_ci_lower[valid_sims, ], 
                                          beta_ci_upper[valid_sims, ], 
                                          beta_true)
rho_coverage_results <- compute_coverage(rho_storage[valid_sims, ], 
                                         rho_ci_lower[valid_sims, ], 
                                         rho_ci_upper[valid_sims, ], 
                                         rho_true)

cat("\n=== CONFIDENCE INTERVAL COVERAGE (95% CI) ===\n")
cat("\nBeta Coverage (Misspecified model):\n")
cat("----------------------------------------\n")
for(k in 1:K) {
  cat(sprintf("  β%-2d: %5.1f%% (%3d/%3d intervals)\n", 
              k, 
              beta_coverage_results$coverage_percent[k],
              beta_coverage_results$n_covered[k],
              beta_coverage_results$n_total[k]))
}
cat(sprintf("\n  Average Beta Coverage: %.1f%% (should be ~95%%, but likely lower)\n", 
            mean(beta_coverage_results$coverage_percent)))

cat("\nRho Coverage:\n")
cat("----------------------------------------\n")
for(k in 1:K) {
  cat(sprintf("  ρ%-2d: %5.1f%% (%3d/%3d intervals)\n", 
              k, 
              rho_coverage_results$coverage_percent[k],
              rho_coverage_results$n_covered[k],
              rho_coverage_results$n_total[k]))
}
cat(sprintf("\n  Average Rho Coverage: %.1f%%\n", 
            mean(rho_coverage_results$coverage_percent)))

