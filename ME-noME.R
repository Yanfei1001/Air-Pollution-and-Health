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
# SIMULATION STUDY: 1000 REPETITIONS (WITH MEASUREMENT ERROR IN DGP,
# BUT INFERENCE IGNORES MEASUREMENT ERROR)
# ==============================================================================

# Parameters
n_sims <- 1000  # Change to 1000 for full simulation
N <- 500        # Time points
K <- 6          # Number of pollutants

# True parameters (extended to 6 pollutants)
beta_true <- c(0.1, -0.2, 0.3, -0.15, 0.25, -0.1)
alpha_true <- -12
gamma_true <- -0.01
rho_true <- c(0.7, 0.65, 0.8, 0.75, 0.68, 0.72)
sigma_proc_true <- c(0.5, 0.6, 0.4, 0.55, 0.45, 0.5)
sigma_me_true <- 0.3  # Measurement error standard deviation

# Storage arrays for results
beta_storage <- array(NA, dim = c(n_sims, K))
rho_storage <- array(NA, dim = c(n_sims, K))
sigma_proc_storage <- array(NA, dim = c(n_sims, K))
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

# ==============================================================================
# JAGS MODEL: IGNORES MEASUREMENT ERROR
# Treats Y_obs as if it were the true latent Z without measurement error
# ==============================================================================
naive_model <- "
model {
  # Naive AR(1) model on OBSERVED data (ignores measurement error)
  for (k in 1:K) {
    # Initial state - treats Y_obs as if it were true Z
    Z[1,k] ~ dnorm(0, tau_proc[k] * (1 - rho[k]^2))
    
    # AR(1) evolution
    for (t in 2:N) {
      Z[t,k] ~ dnorm(rho[k] * Z[t-1,k], tau_proc[k])
    }
  }
  
  # Outcome model - uses observed pollutants directly
  for (t in 1:N) {
    Y_out[t] ~ dpois(lambda[t])
    log(lambda[t]) <- alpha + inprod(beta[1:K], Y_obs[t,1:K]) + 
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
  # DATA GENERATING PROCESS (WITH MEASUREMENT ERROR)
  # ============================================================================
  
  # Generate AR(1) latent states (true unobserved pollutants)
  Z_true <- matrix(NA, nrow = N, ncol = K)
  for(k in 1:K) {
    Z_true[1, k] <- rnorm(1, 0, sigma_proc_true[k] / sqrt(1 - rho_true[k]^2))
    for(t in 2:N) {
      Z_true[t, k] <- rho_true[k] * Z_true[t-1, k] + 
        rnorm(1, 0, sigma_proc_true[k])
    }
  }
  
  # Add measurement error to create observed pollutants
  # This is the key: observed data = true latent + classical measurement error
  Y_obs <- Z_true + rnorm(N * K, 0, sigma_me_true)
  
  # Generate outcomes using TRUE latent pollutants (not observed!)
  log_lambda <- alpha_true + Z_true %*% beta_true + 
    gamma_true * temperature + log(population)
  Y_out <- rpois(N, exp(log_lambda))
  
  # ============================================================================
  # INFERENCE: IGNORES MEASUREMENT ERROR
  # The model treats Y_obs as if they were the true Z values
  # ============================================================================
  
  # Prepare data for JAGS
  data_list <- list(
    N = N, K = K,
    Y_obs = Y_obs,  # Used directly in outcome model (ignoring measurement error!)
    Y_out = Y_out,
    temperature = temperature,
    population = population
  )
  
  # Fit JAGS model (reduced iterations for speed)
  tryCatch({
    fit <- jags(data_list, 
                parameters.to.save = c("beta", "alpha", "gamma", "rho", 
                                       "sigma_proc", "log_lik"),
                model.file = textConnection(naive_model),
                n.chains = 2,
                n.iter = 5000,
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

# Beta results (expect bias due to ignoring measurement error)
beta_summary <- compute_summary(beta_storage[valid_sims, ], beta_true, "beta")
cat("\n=== BETA ESTIMATES (Naive model - ignoring measurement error) ===\n")
print(beta_summary)

# Rho results
rho_summary <- compute_summary(rho_storage[valid_sims, ], rho_true, "rho")
cat("\n=== RHO ESTIMATES ===\n")
print(rho_summary)

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

# Coverage probabilities (expect under-coverage due to measurement error)
beta_coverage_results <- compute_coverage(beta_storage[valid_sims, ], 
                                          beta_ci_lower[valid_sims, ], 
                                          beta_ci_upper[valid_sims, ], 
                                          beta_true)
rho_coverage_results <- compute_coverage(rho_storage[valid_sims, ], 
                                         rho_ci_lower[valid_sims, ], 
                                         rho_ci_upper[valid_sims, ], 
                                         rho_true)

cat("\n=== CONFIDENCE INTERVAL COVERAGE (95% CI) ===\n")
cat("\nBeta Coverage (Naive model - expect under-coverage):\n")
cat("----------------------------------------\n")
for(k in 1:K) {
  cat(sprintf("  β%-2d: %5.1f%% (%3d/%3d intervals)\n", 
              k, 
              beta_coverage_results$coverage_percent[k],
              beta_coverage_results$n_covered[k],
              beta_coverage_results$n_total[k]))
}
cat(sprintf("\n  Average Beta Coverage: %.1f%% (should be <95%% due to attenuation bias)\n", 
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

# ==============================================================================
# WAIC ANALYSIS
# ==============================================================================

valid_waic <- waic_storage[valid_sims & !is.na(waic_storage)]
n_valid_waic <- length(valid_waic)

cat("\n=== WAIC RESULTS ===\n")
cat("----------------------------------------\n")
cat(sprintf("Number of simulations with valid WAIC: %d\n", n_valid_waic))
cat(sprintf("Mean WAIC: %.2f\n", mean(valid_waic)))
cat(sprintf("Median WAIC: %.2f\n", median(valid_waic)))
cat(sprintf("SD of WAIC: %.2f\n", sd(valid_waic)))

# ==============================================================================
# ENHANCED VISUALIZATIONS
# ==============================================================================

# 1. Beta estimates with attenuation bias visualization
par(mfrow = c(2, 3), mar = c(4, 4, 3, 2))
for(k in 1:K) {
  hist(beta_storage[valid_sims, k], breaks = 30, 
       col = rgb(0.2, 0.4, 0.6, 0.7), border = "white",
       main = paste("β", k, "- Naive Estimates (Ignore ME)"),
       xlab = "Estimate", 
       sub = paste("True =", beta_true[k], 
                   "| Coverage =", round(beta_coverage_results$coverage_percent[k], 1), "%"))
  abline(v = beta_true[k], col = "red", lwd = 2, lty = 2)
  abline(v = mean(beta_storage[valid_sims, k], na.rm = TRUE), 
         col = "blue", lwd = 2, lty = 1)
  
  # Add annotation for attenuation bias direction
  if(mean(beta_storage[valid_sims, k], na.rm = TRUE) < beta_true[k]) {
    mtext("← Attenuation bias", side = 3, line = 0.5, col = "darkred", cex = 0.8)
  }
  
  legend("topright", legend = c("True", "Mean Estimate"), 
         col = c("red", "blue"), lty = c(2, 1), lwd = 2, cex = 0.8)
}

# 2. Boxplots with coverage annotation

png("/Users/yanfei/Desktop/Manuscript Prep/Simulation/beta_estimates_plot-MEnoME.png", width = 8, height = 6, units = "in", res = 300)


par(mfrow = c(1, 1))
boxplot(as.data.frame(beta_storage[valid_sims, ]), 
        col = rgb(0.2, 0.4, 0.6, 0.7), 
        # main = "β Estimates Ignoring Measurement Error\n(Expect attenuation bias and under-coverage)",
        # xlab = "Pollutant", 
        ylab = "Estimate",
        names = paste0("β", 1:K, "\n(", round(beta_coverage_results$coverage_percent, 1), "%)"))
points(1:K, beta_true, col = "red", pch = 8, cex = 1.5, lwd = 2)
legend("topright", legend = "True Values", col = "red", pch = 8, cex = 0.7)

dev.off()


# 3. Rho estimates
par(mfrow = c(2, 3))
for(k in 1:K) {
  hist(rho_storage[valid_sims, k], breaks = 30, 
       col = rgb(0.6, 0.4, 0.2, 0.7), border = "white",
       main = paste("ρ", k, "- Distribution"),
       xlab = "Estimate", 
       sub = paste("True =", rho_true[k], 
                   "| Coverage =", round(rho_coverage_results$coverage_percent[k], 1), "%"))
  abline(v = rho_true[k], col = "red", lwd = 2, lty = 2)
  abline(v = mean(rho_storage[valid_sims, k], na.rm = TRUE), 
         col = "blue", lwd = 2, lty = 1)
  legend("topright", legend = c("True", "Mean Estimate"), 
         col = c("red", "blue"), lty = c(2, 1), lwd = 2, cex = 0.8)
}

# 4. WAIC distribution
par(mfrow = c(1, 2))
hist(valid_waic, breaks = 30, 
     col = rgb(0.3, 0.5, 0.3, 0.7), border = "white",
     main = "WAIC Distribution (Naive Model)",
     xlab = "WAIC", 
     sub = paste("Mean =", round(mean(valid_waic), 2)))
abline(v = mean(valid_waic), col = "blue", lwd = 2, lty = 1)
abline(v = median(valid_waic), col = "red", lwd = 2, lty = 2)
legend("topright", legend = c("Mean", "Median"), 
       col = c("blue", "red"), lty = c(1, 2), lwd = 2, cex = 0.8)

boxplot(valid_waic, 
        col = rgb(0.3, 0.5, 0.3, 0.7),
        main = "WAIC Boxplot (Naive Model)",
        ylab = "WAIC")
points(1, mean(valid_waic), col = "blue", pch = 8, cex = 1.5)

# 5. Additional diagnostic plot: Attenuation factor
par(mfrow = c(1, 1))
beta_true_vec <- rep(beta_true, each = n_valid)
beta_est_vec <- as.vector(t(beta_storage[valid_sims, ]))
plot(beta_true_vec, beta_est_vec, 
     xlab = "True β", ylab = "Estimated β",
     main = "Attenuation Due to Measurement Error",
     pch = 19, col = rgb(0.2, 0.4, 0.6, 0.3))
abline(a = 0, b = 1, col = "red", lwd = 2, lty = 2)

# Add regression line to show attenuation
fit <- lm(beta_est_vec ~ beta_true_vec)
abline(fit, col = "blue", lwd = 2)
legend("bottomright", 
       legend = c("Ideal (slope=1)", paste0("Observed slope = ", round(coef(fit)[2], 3))),
       col = c("red", "blue"), lty = c(2, 1), lwd = 2, cex = 0.8)

cat("\n=== SUMMARY OF ATTENUATION BIAS ===\n")
cat("Expected attenuation factor (theoretically):", 
    sigma_proc_true[1]^2 / (sigma_proc_true[1]^2 + sigma_me_true^2), "\n")
cat("Observed attenuation factor (from regression):", round(coef(fit)[2], 3), "\n")