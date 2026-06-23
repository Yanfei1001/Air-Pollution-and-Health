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
# SIMULATION STUDY: 1000 REPETITIONS (WITHOUT MEASUREMENT ERROR IN DGP,
# AND INFERENCE CORRECTLY ASSUMES NO MEASUREMENT ERROR)
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
# NO measurement error in DGP (sigma_me_true = 0)

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
cat("Starting", n_sims, "simulations (NO measurement error in DGP)...\n")
start_time <- Sys.time()

# ==============================================================================
# JAGS MODEL: CORRECTLY SPECIFIED (NO MEASUREMENT ERROR)
# Treats observed data as true latent states (which is correct here)
# ==============================================================================
correct_model <- "
model {
  # AR(1) model on observed data (which equals true latent states in DGP)
  for (k in 1:K) {
    # Initial state
    Z[1,k] ~ dnorm(0, tau_proc[k] * (1 - rho[k]^2))
    
    # AR(1) evolution
    for (t in 2:N) {
      Z[t,k] ~ dnorm(rho[k] * Z[t-1,k], tau_proc[k])
    }
  }
  
  # Outcome model - uses observed pollutants directly
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
  # INFERENCE: CORRECTLY SPECIFIED MODEL (assumes no measurement error)
  # The model correctly treats Y_obs as the true Z values
  # ============================================================================
  
  # Prepare data for JAGS
  data_list <- list(
    N = N, K = K,
    Z = Y_obs,  # Using observed as true states (correct specification)
    Y_out = Y_out,
    temperature = temperature,
    population = population
  )
  
  # Fit JAGS model (reduced iterations for speed)
  tryCatch({
    fit <- jags(data_list, 
                parameters.to.save = c("beta", "alpha", "gamma", "rho", 
                                       "sigma_proc", "log_lik"),
                model.file = textConnection(correct_model),
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

# Beta results (should be unbiased since model is correctly specified)
beta_summary <- compute_summary(beta_storage[valid_sims, ], beta_true, "beta")
cat("\n=== BETA ESTIMATES (Correctly specified model - no measurement error) ===\n")
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

# Coverage probabilities (should be close to 95% since model is correct)
beta_coverage_results <- compute_coverage(beta_storage[valid_sims, ], 
                                          beta_ci_lower[valid_sims, ], 
                                          beta_ci_upper[valid_sims, ], 
                                          beta_true)
rho_coverage_results <- compute_coverage(rho_storage[valid_sims, ], 
                                         rho_ci_lower[valid_sims, ], 
                                         rho_ci_upper[valid_sims, ], 
                                         rho_true)

cat("\n=== CONFIDENCE INTERVAL COVERAGE (95% CI) ===\n")
cat("\nBeta Coverage (Correctly specified model):\n")
cat("----------------------------------------\n")
for(k in 1:K) {
  cat(sprintf("  β%-2d: %5.1f%% (%3d/%3d intervals)\n", 
              k, 
              beta_coverage_results$coverage_percent[k],
              beta_coverage_results$n_covered[k],
              beta_coverage_results$n_total[k]))
}
cat(sprintf("\n  Average Beta Coverage: %.1f%% (should be ~95%%)\n", 
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

# Create directory for plots (adjust path as needed)
# dir.create("/Users/yanfei/Desktop/Manuscript Prep/Simulation", showWarnings = FALSE)

# 1. Beta estimates - should be unbiased
png("/Users/yanfei/Desktop/Manuscript Prep/Simulation/beta_estimates_noME.png", 
    width = 8, height = 6, units = "in", res = 300)

par(mfrow = c(1, 1))
boxplot(as.data.frame(beta_storage[valid_sims, ]), 
        col = rgb(0.2, 0.6, 0.4, 0.7), 
        ylab = "Estimate",
        names = paste0("β", 1:K, "\n(", round(beta_coverage_results$coverage_percent, 1), "%)")
        # ,
        # main = "β Estimates (No Measurement Error in DGP or Inference)\nExpected: Unbiased estimates, ~95% coverage"
        )
points(1:K, beta_true, col = "red", pch = 8, cex = 1.5, lwd = 2)
legend("topright", legend = "True Values", col = "red", pch = 8, cex = 0.7)

dev.off()

# 2. Beta histograms by pollutant
png("/Users/yanfei/Desktop/Manuscript Prep/Simulation/beta_histograms_noME.png", 
    width = 12, height = 8, units = "in", res = 300)

par(mfrow = c(2, 3), mar = c(4, 4, 3, 2))
for(k in 1:K) {
  hist(beta_storage[valid_sims, k], breaks = 30, 
       col = rgb(0.2, 0.6, 0.4, 0.7), border = "white",
       main = paste("β", k, "- Correctly Specified Model"),
       xlab = "Estimate", 
       sub = paste("True =", beta_true[k], 
                   "| Coverage =", round(beta_coverage_results$coverage_percent[k], 1), "%"))
  abline(v = beta_true[k], col = "red", lwd = 2, lty = 2)
  abline(v = mean(beta_storage[valid_sims, k], na.rm = TRUE), 
         col = "blue", lwd = 2, lty = 1)
  legend("topright", legend = c("True", "Mean Estimate"), 
         col = c("red", "blue"), lty = c(2, 1), lwd = 2, cex = 0.8)
}

dev.off()

# 3. Rho estimates histograms
png("/Users/yanfei/Desktop/Manuscript Prep/Simulation/rho_histograms_noME.png", 
    width = 12, height = 8, units = "in", res = 300)

par(mfrow = c(2, 3), mar = c(4, 4, 3, 2))
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

dev.off()

# 4. WAIC distribution
png("/Users/yanfei/Desktop/Manuscript Prep/Simulation/waic_noME.png", 
    width = 10, height = 6, units = "in", res = 300)

par(mfrow = c(1, 2))
hist(valid_waic, breaks = 30, 
     col = rgb(0.3, 0.6, 0.3, 0.7), border = "white",
     main = "WAIC Distribution (Correct Model)",
     xlab = "WAIC", 
     sub = paste("Mean =", round(mean(valid_waic), 2)))
abline(v = mean(valid_waic), col = "blue", lwd = 2, lty = 1)
abline(v = median(valid_waic), col = "red", lwd = 2, lty = 2)
legend("topright", legend = c("Mean", "Median"), 
       col = c("blue", "red"), lty = c(1, 2), lwd = 2, cex = 0.8)

boxplot(valid_waic, 
        col = rgb(0.3, 0.6, 0.3, 0.7),
        main = "WAIC Boxplot (Correct Model)",
        ylab = "WAIC")
points(1, mean(valid_waic), col = "blue", pch = 8, cex = 1.5)

dev.off()

# 5. Bias comparison plot
png("/Users/yanfei/Desktop/Manuscript Prep/Simulation/bias_comparison.png", 
    width = 8, height = 6, units = "in", res = 300)

par(mfrow = c(1, 1))
beta_bias <- beta_summary$Bias
beta_true <- beta_summary$True

barplot(beta_bias, 
        names.arg = paste0("β", 1:K),
        col = ifelse(beta_bias >= 0, rgb(0.2, 0.6, 0.4, 0.7), rgb(0.8, 0.3, 0.3, 0.7)),
        main = "Bias in β Estimates (Correctly Specified Model)",
        xlab = "Pollutant", ylab = "Bias",
        ylim = c(-max(abs(beta_bias))*1.2, max(abs(beta_bias))*1.2))
abline(h = 0, col = "black", lwd = 2)

text(x = 1:K, y = beta_bias + sign(beta_bias)*0.01, 
     labels = round(beta_bias, 4), cex = 0.8)

dev.off()

# 6. Summary table
cat("\n\n=== SUMMARY COMPARISON ===\n")
cat("========================================\n")
cat(sprintf("Average Beta Bias: %.6f\n", mean(abs(beta_summary$Bias))))
cat(sprintf("Average Beta RMSE: %.6f\n", mean(beta_summary$RMSE)))
cat(sprintf("Average Beta Coverage: %.1f%%\n", mean(beta_coverage_results$coverage_percent)))
cat("========================================\n")
cat(sprintf("Average Rho Bias: %.6f\n", mean(abs(rho_summary$Bias))))
cat(sprintf("Average Rho RMSE: %.6f\n", mean(rho_summary$RMSE)))
cat(sprintf("Average Rho Coverage: %.1f%%\n", mean(rho_coverage_results$coverage_percent)))

# Save results for later comparison
save(beta_storage, rho_storage, sigma_proc_storage, 
     beta_summary, rho_summary, 
     beta_coverage_results, rho_coverage_results,
     valid_waic, n_valid,
     file = "/Users/yanfei/Desktop/Manuscript Prep/Simulation/noME_results.RData")