# Load required libraries
library(R2jags)
library(coda)
library(ggplot2)
library(dplyr)
library(tidyr)
library(purrr)
library(loo)  # For WAIC calculations

# Set seed for reproducibility
set.seed(123)

# ==============================================================================
# SIMULATION STUDY: 1000 REPETITIONS
# ==============================================================================

# Parameters
n_sims <- 1000  # Changed to 1000 for full simulation
N <- 500        # Time points
K <- 6          # Number of pollutants

# True parameters (extended to 6 pollutants)
beta_true <- c(0.1, -0.2, 0.3, -0.15, 0.25, -0.1)
alpha_true <- -12
gamma_true <- -0.01
rho_true <- c(0.7, 0.65, 0.8, 0.75, 0.68, 0.72)
sigma_proc_true <- c(0.5, 0.6, 0.4, 0.55, 0.45, 0.5)
sigma_me_true <- 0.3

# Storage arrays for results
beta_storage <- array(NA, dim = c(n_sims, K))
rho_storage <- array(NA, dim = c(n_sims, K))
sigma_proc_storage <- array(NA, dim = c(n_sims, K))
sigma_me_storage <- rep(NA, n_sims)
alpha_storage <- rep(NA, n_sims)
gamma_storage <- rep(NA, n_sims)

# Storage for CI coverage
beta_ci_lower <- array(NA, dim = c(n_sims, K))
beta_ci_upper <- array(NA, dim = c(n_sims, K))
rho_ci_lower <- array(NA, dim = c(n_sims, K))
rho_ci_upper <- array(NA, dim = c(n_sims, K))

# Storage for WAIC values
waic_storage <- rep(NA, n_sims)

# Progress tracking
cat("Starting", n_sims, "simulations...\n")
start_time <- Sys.time()

# JAGS model (modified to compute log-likelihood for WAIC)
ar_model <- "
model {
  # AR(1) latent process
  for (k in 1:K) {
    # Initial state (stationary)
    Z[1,k] ~ dnorm(0, tau_proc[k] * (1 - rho[k]^2))
    
    # AR(1) evolution
    for (t in 2:N) {
      Z[t,k] ~ dnorm(rho[k] * Z[t-1,k], tau_proc[k])
    }
  }
  
  # Measurement model
  for (t in 1:N) {
    for (k in 1:K) {
      Y_obs[t,k] ~ dnorm(Z[t,k], tau_me)
    }
  }
  
  # Outcome model
  for (t in 1:N) {
    Y_out[t] ~ dpois(lambda[t])
    log(lambda[t]) <- alpha + inprod(beta[1:K], Z[t,1:K]) + 
                      gamma * temperature[t] + log(population[t])
    
    # Log-likelihood for WAIC calculation
    log_lik[t] <- logdensity.pois(Y_out[t], lambda[t])
  }
  
  # Priors
  for (k in 1:K) {
    rho_raw[k] ~ dbeta(15, 5)
    rho[k] <- 2 * rho_raw[k] - 1
    tau_proc[k] ~ dgamma(2, 1)
    sigma_proc[k] <- 1/sqrt(tau_proc[k])
    beta[k] ~ dnorm(0, 0.01)
  }
  
  tau_me ~ dgamma(10, 1)
  sigma_me <- 1/sqrt(tau_me)
  alpha ~ dnorm(0, 0.01)
  gamma ~ dnorm(0, 0.01)
}
"

# Run simulations
for(sim in 1:n_sims) {
  if(sim %% 50 == 0) cat("Completed simulation", sim, "of", n_sims, "\n")
  
  # Generate temperature data (fixed across simulations for consistency)
  set.seed(100 + sim)
  temperature <- 15 + 5 * sin(2 * pi * (1:N) / 52) + rnorm(N, 0, 1)
  population <- rep(100000, N)
  
  # Generate AR(1) latent states
  Z_true <- matrix(NA, nrow = N, ncol = K)
  for(k in 1:K) {
    Z_true[1, k] <- rnorm(1, 0, sigma_proc_true[k] / sqrt(1 - rho_true[k]^2))
    for(t in 2:N) {
      Z_true[t, k] <- rho_true[k] * Z_true[t-1, k] + 
        rnorm(1, 0, sigma_proc_true[k])
    }
  }
  
  # Add measurement error
  Y_obs <- Z_true + rnorm(N * K, 0, sigma_me_true)
  
  # Generate outcomes
  log_lambda <- alpha_true + Z_true %*% beta_true + 
    gamma_true * temperature + log(population)
  Y_out <- rpois(N, exp(log_lambda))
  
  # Prepare data for JAGS
  data_list <- list(
    N = N, K = K,
    Y_obs = Y_obs,
    Y_out = Y_out,
    temperature = temperature,
    population = population
  )
  
  # Fit JAGS model (reduced iterations for speed)
  tryCatch({
    fit <- jags(data_list, 
                parameters.to.save = c("beta", "alpha", "gamma", "rho", 
                                       "sigma_proc", "sigma_me", "log_lik"),
                model.file = textConnection(ar_model),
                n.chains = 2,        # Reduced chains for speed
                n.iter = 5000,       # Reduced iterations
                n.burnin = 1000,
                n.thin = 4,
                quiet = TRUE)
    
    # Extract results
    results <- fit$BUGSoutput$summary
    
    # Store point estimates (mean)
    for(k in 1:K) {
      beta_storage[sim, k] <- results[paste0("beta[", k, "]"), "mean"]
      rho_storage[sim, k] <- results[paste0("rho[", k, "]"), "mean"]
      sigma_proc_storage[sim, k] <- results[paste0("sigma_proc[", k, "]"), "mean"]
      
      # Store CI bounds
      beta_ci_lower[sim, k] <- results[paste0("beta[", k, "]"), "2.5%"]
      beta_ci_upper[sim, k] <- results[paste0("beta[", k, "]"), "97.5%"]
      rho_ci_lower[sim, k] <- results[paste0("rho[", k, "]"), "2.5%"]
      rho_ci_upper[sim, k] <- results[paste0("rho[", k, "]"), "97.5%"]
    }
    
    alpha_storage[sim] <- results["alpha", "mean"]
    gamma_storage[sim] <- results["gamma", "mean"]
    sigma_me_storage[sim] <- results["sigma_me", "mean"]
    
    # Calculate WAIC from log-likelihood
    # Extract log-likelihood samples
    log_lik_samples <- fit$BUGSoutput$sims.list$log_lik
    
    # Compute WAIC
    lppd <- sum(log(colMeans(exp(log_lik_samples))))  # log pointwise predictive density
    p_waic <- sum(apply(log_lik_samples, 2, var))      # effective number of parameters
    waic_storage[sim] <- -2 * (lppd - p_waic)          # WAIC
    
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
  
  # Standard deviation of estimates
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


# Beta results
boxplot(alpha_storage)
boxplot(gamma_storage)

# Beta results
beta_summary <- compute_summary(beta_storage[valid_sims, ], beta_true, "beta")
cat("\n=== BETA ESTIMATES ===\n")
print(beta_summary)

# Rho results
rho_summary <- compute_summary(rho_storage[valid_sims, ], rho_true, "rho")
cat("\n=== RHO ESTIMATES ===\n")
print(rho_summary)

# Function to compute coverage with percentages
compute_coverage <- function(estimates, ci_lower, ci_upper, true_values) {
  coverage <- matrix(NA, nrow = nrow(estimates), ncol = ncol(estimates))
  for(j in 1:ncol(estimates)) {
    coverage[,j] <- (ci_lower[,j] <= true_values[j]) & (ci_upper[,j] >= true_values[j])
  }
  coverage_percent <- colMeans(coverage, na.rm = TRUE) * 100
  return(list(
    coverage_matrix = coverage,
    coverage_percent = coverage_percent,
    n_covered = colSums(coverage, na.rm = TRUE),
    n_total = colSums(!is.na(coverage))
  ))
}

# Coverage probabilities
beta_coverage_results <- compute_coverage(beta_storage[valid_sims, ], 
                                          beta_ci_lower[valid_sims, ], 
                                          beta_ci_upper[valid_sims, ], 
                                          beta_true)
rho_coverage_results <- compute_coverage(rho_storage[valid_sims, ], 
                                         rho_ci_lower[valid_sims, ], 
                                         rho_ci_upper[valid_sims, ], 
                                         rho_true)

cat("\n=== CONFIDENCE INTERVAL COVERAGE (95% CI) ===\n")
cat("\nBeta Coverage:\n")
cat("----------------------------------------\n")
for(k in 1:K) {
  cat(sprintf("  β%-2d: %5.1f%% (%3d/%3d intervals contained true value)\n", 
              k, 
              beta_coverage_results$coverage_percent[k],
              beta_coverage_results$n_covered[k],
              beta_coverage_results$n_total[k]))
}
cat(sprintf("\n  Average Beta Coverage: %.1f%%\n", 
            mean(beta_coverage_results$coverage_percent)))

cat("\nRho Coverage:\n")
cat("----------------------------------------\n")
for(k in 1:K) {
  cat(sprintf("  ρ%-2d: %5.1f%% (%3d/%3d intervals contained true value)\n", 
              k, 
              rho_coverage_results$coverage_percent[k],
              rho_coverage_results$n_covered[k],
              rho_coverage_results$n_total[k]))
}
cat(sprintf("\n  Average Rho Coverage: %.1f%%\n", 
            mean(rho_coverage_results$coverage_percent)))
