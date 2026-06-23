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
dir.create("/Users/yanfei/Desktop/Manuscript Prep/Simulation", showWarnings = FALSE)

# 1. Beta estimates - likely biased
png("/Users/yanfei/Desktop/Manuscript Prep/Simulation/beta_estimates_overME.png", 
    width = 8, height = 6, units = "in", res = 300)

par(mfrow = c(1, 1))
boxplot(as.data.frame(beta_storage[valid_sims, ]), 
        col = rgb(0.8, 0.3, 0.3, 0.7), 
        ylab = "Estimate",
        names = paste0("β", 1:K, "\n(", round(beta_coverage_results$coverage_percent, 1), "%)"),
        main = "β Estimates (Assuming ME when none exists)\nExpected: Biased estimates, reduced coverage")
points(1:K, beta_true, col = "red", pch = 8, cex = 1.5, lwd = 2)
legend("topright", legend = "True Values", col = "red", pch = 8, cex = 0.7)

dev.off()

# 2. Beta comparison: Correct vs Misspecified
png("/Users/yanfei/Desktop/Manuscript Prep/Simulation/beta_comparison_correct_vs_ME.png", 
    width = 12, height = 6, units = "in", res = 300)

par(mfrow = c(1, 2))

# Correct model bias (if available from previous run)
# For now, just show misspecified
barplot(beta_summary$Bias, 
        names.arg = paste0("β", 1:K),
        col = ifelse(beta_summary$Bias >= 0, rgb(0.8, 0.3, 0.3, 0.7), rgb(0.3, 0.3, 0.8, 0.7)),
        main = "Bias in β Estimates (Misspecified Model)",
        xlab = "Pollutant", ylab = "Bias",
        ylim = c(-max(abs(beta_summary$Bias))*1.2, max(abs(beta_summary$Bias))*1.2))
abline(h = 0, col = "black", lwd = 2)

text(x = 1:K, y = beta_summary$Bias + sign(beta_summary$Bias)*0.005, 
     labels = round(beta_summary$Bias, 4), cex = 0.8)

# RMSE comparison
rmse_values <- beta_summary$RMSE
barplot(rmse_values, 
        names.arg = paste0("β", 1:K),
        col = rgb(0.8, 0.3, 0.3, 0.7),
        main = "RMSE of β Estimates (Misspecified Model)",
        xlab = "Pollutant", ylab = "RMSE")

dev.off()

# 3. Measurement error estimates (should be >0 even though true=0)
png("/Users/yanfei/Desktop/Manuscript Prep/Simulation/sigma_ME_estimates.png", 
    width = 12, height = 8, units = "in", res = 300)

par(mfrow = c(2, 3), mar = c(4, 4, 3, 2))
for(k in 1:K) {
  hist(sigma_me_storage[valid_sims, k], breaks = 30, 
       col = rgb(0.6, 0.4, 0.8, 0.7), border = "white",
       main = paste("σ_me[", k, "] - Over-specified Model", sep=""),
       xlab = "Estimate", 
       sub = paste("True = 0", 
                   "| Mean Est =", round(mean(sigma_me_storage[valid_sims, k], na.rm = TRUE), 3)))
  abline(v = 0, col = "red", lwd = 2, lty = 2)
  abline(v = mean(sigma_me_storage[valid_sims, k], na.rm = TRUE), 
         col = "blue", lwd = 2, lty = 1)
  legend("topright", legend = c("True (0)", "Mean Estimate"), 
         col = c("red", "blue"), lty = c(2, 1), lwd = 2, cex = 0.8)
}

dev.off()

# 4. Comparison of process error estimates
png("/Users/yanfei/Desktop/Manuscript Prep/Simulation/sigma_proc_comparison.png", 
    width = 10, height = 6, units = "in", res = 300)

sigma_proc_summary <- compute_summary(sigma_proc_storage[valid_sims, ], sigma_proc_true, "sigma_proc")

par(mfrow = c(1, 1))
comparison_data <- data.frame(
  Pollutant = rep(1:K, 2),
  Type = rep(c("True", "Estimated"), each = K),
  Value = c(sigma_proc_true, sigma_proc_summary$Mean_Est)
)

boxplot(as.data.frame(sigma_proc_storage[valid_sims, ]),
        col = rgb(0.3, 0.6, 0.4, 0.5),
        ylab = "Process Error SD",
        names = paste0("σ_proc[", 1:K, "]"),
        main = "Process Error Estimates (Misspecified Model)")
points(1:K, sigma_proc_true, col = "red", pch = 8, cex = 1.5, lwd = 2)

legend("topright", legend = c("True Values", "Mean Estimates"), 
       col = c("red", "black"), pch = c(8, NA), lty = c(NA, NA), 
       fill = c(NA, rgb(0.3, 0.6, 0.4, 0.5)), border = "black")

dev.off()

# 5. Coverage comparison plot
png("/Users/yanfei/Desktop/Manuscript Prep/Simulation/coverage_ME_model.png", 
    width = 8, height = 6, units = "in", res = 300)

par(mfrow = c(1, 1))
coverage_data <- data.frame(
  Pollutant = rep(1:K, 2),
  Parameter = rep(c("Beta", "Rho"), each = K),
  Coverage = c(beta_coverage_results$coverage_percent, rho_coverage_results$coverage_percent)
)

barplot(height = coverage_data$Coverage, 
        beside = FALSE,
        names.arg = paste0(coverage_data$Parameter, coverage_data$Pollutant),
        col = c(rep(rgb(0.3, 0.6, 0.8, 0.7), K), rep(rgb(0.8, 0.6, 0.3, 0.7), K)),
        main = "95% CI Coverage (Misspecified Model)",
        xlab = "Parameter", ylab = "Coverage (%)",
        ylim = c(0, 100))
abline(h = 95, col = "red", lwd = 2, lty = 2)
text(x = 1:(2*K), y = coverage_data$Coverage + 2, 
     labels = paste0(round(coverage_data$Coverage, 1), "%"), cex = 0.7)

legend("bottomleft", legend = c("Beta Coverage", "Rho Coverage", "Nominal 95%"), 
       fill = c(rgb(0.3, 0.6, 0.8, 0.7), rgb(0.8, 0.6, 0.3, 0.7), NA),
       lty = c(NA, NA, 2), col = c(NA, NA, "red"), lwd = c(NA, NA, 2))

dev.off()

# 6. WAIC distribution
png("/Users/yanfei/Desktop/Manuscript Prep/Simulation/waic_overME.png", 
    width = 10, height = 6, units = "in", res = 300)

par(mfrow = c(1, 2))
hist(valid_waic, breaks = 30, 
     col = rgb(0.8, 0.4, 0.4, 0.7), border = "white",
     main = "WAIC Distribution (Over-specified Model)",
     xlab = "WAIC", 
     sub = paste("Mean =", round(mean(valid_waic), 2)))
abline(v = mean(valid_waic), col = "blue", lwd = 2, lty = 1)
abline(v = median(valid_waic), col = "red", lwd = 2, lty = 2)
legend("topright", legend = c("Mean", "Median"), 
       col = c("blue", "red"), lty = c(1, 2), lwd = 2, cex = 0.8)

boxplot(valid_waic, 
        col = rgb(0.8, 0.4, 0.4, 0.7),
        main = "WAIC Boxplot (Over-specified Model)",
        ylab = "WAIC")
points(1, mean(valid_waic), col = "blue", pch = 8, cex = 1.5)

dev.off()

# 7. Posterior distribution of measurement error (showing convergence issues)
png("/Users/yanfei/Desktop/Manuscript Prep/Simulation/sigma_ME_trace.png", 
    width = 10, height = 6, units = "in", res = 300)

# Plot for a single pollutant from last simulation
if(n_valid > 0) {
  # Get last valid simulation
  last_valid <- tail(which(valid_sims), 1)
  
  # Refit one simulation to get MCMC chains for demonstration
  set.seed(999)
  N_demo <- 100  # Smaller for demonstration
  Z_demo <- matrix(NA, nrow = N_demo, ncol = 1)
  rho_demo <- 0.7
  sigma_proc_demo <- 0.5
  Z_demo[1] <- rnorm(1, 0, sigma_proc_demo / sqrt(1 - rho_demo^2))
  for(t in 2:N_demo) {
    Z_demo[t] <- rho_demo * Z_demo[t-1] + rnorm(1, 0, sigma_proc_demo)
  }
  Y_obs_demo <- Z_demo  # No measurement error
  
  data_demo <- list(
    N = N_demo, K = 1,
    Y_obs = matrix(Y_obs_demo, ncol = 1),
    Y_out = rpois(N_demo, exp(-12 + 0.1*Z_demo + log(100000))),
    temperature = rep(0, N_demo),
    population = rep(100000, N_demo)
  )
  
  fit_demo <- jags(data_demo,
                   inits = list(list(sigma_me = 0.5)),
                   parameters.to.save = c("sigma_me", "beta", "rho"),
                   model.file = textConnection(misspecified_model),
                   n.chains = 1,
                   n.iter = 5000,
                   n.burnin = 1000,
                   n.thin = 2,
                   quiet = TRUE)
  
  # Plot trace of sigma_me
  sigma_me_samples <- fit_demo$BUGSoutput$sims.list$sigma_me[,1]
  plot(sigma_me_samples, type = "l", 
       main = "Trace Plot of σ_me (should converge to 0 but may not)",
       xlab = "Iteration", ylab = "σ_me",
       col = rgb(0.4, 0.4, 0.8, 0.7))
  abline(h = 0, col = "red", lwd = 2, lty = 2)
}

dev.off()

# 8. Summary table
cat("\n\n=== SUMMARY COMPARISON ===\n")
cat("========================================\n")
cat("DGP: NO measurement error\n")
cat("Inference: ASSUMES measurement error\n")
cat("========================================\n")
cat(sprintf("Average Beta Bias: %.6f\n", mean(abs(beta_summary$Bias))))
cat(sprintf("Average Beta RMSE: %.6f\n", mean(beta_summary$RMSE)))
cat(sprintf("Average Beta Coverage: %.1f%%\n", mean(beta_coverage_results$coverage_percent)))
cat("----------------------------------------\n")
cat(sprintf("Average Rho Bias: %.6f\n", mean(abs(rho_summary$Bias))))
cat(sprintf("Average Rho RMSE: %.6f\n", mean(rho_summary$RMSE)))
cat(sprintf("Average Rho Coverage: %.1f%%\n", mean(rho_coverage_results$coverage_percent)))
cat("----------------------------------------\n")
cat(sprintf("Average σ_me Estimate (True=0): %.6f\n", mean(sigma_me_summary$Mean_Est)))
cat(sprintf("σ_me Bias: %.6f\n", mean(sigma_me_summary$Bias)))
cat("========================================\n")

# Save results for later comparison
save(beta_storage, rho_storage, sigma_proc_storage, sigma_me_storage,
     beta_summary, rho_summary, sigma_me_summary,
     beta_coverage_results, rho_coverage_results,
     valid_waic, n_valid,
     file = "/Users/yanfei/Desktop/Manuscript Prep/Simulation/overME_results.RData")

# ==============================================================================
# DIAGNOSTIC: Check if measurement error is identifiable
# ==============================================================================

cat("\n\n=== DIAGNOSTIC: Measurement Error Identifiability ===\n")
cat("When true σ_me = 0, the model may still estimate positive values due to:\n")
cat("1. Prior specification (sigma_me ~ dunif(0,5) forces positive estimates)\n")
cat("2. Identifiability issues between process and measurement error\n")
cat("3. Overfitting - the extra parameter captures noise\n")

# Calculate how often sigma_me is significantly >0
sig_pos <- colMeans(sigma_me_ci_lower[valid_sims, ] > 0, na.rm = TRUE) * 100
cat("\nPercentage of simulations where 95% CI excludes 0:\n")
for(k in 1:K) {
  cat(sprintf("  σ_me[%d]: %.1f%%\n", k, sig_pos[k]))
}
cat("\nThis shows the model incorrectly identifies measurement error\n")
cat("even when none exists in the data generating process.\n")








png("/Users/yanfei/Desktop/Manuscript Prep/Simulation/estimates_plot_noMEME.png", width = 8, height = 6, units = "in", res = 300)

# ==============================================================================
# GAMMA COVERAGE COMPUTATION
# ==============================================================================

# Get gamma estimates from valid simulations
gamma_estimates <- gamma_storage[valid_sims]

# Calculate gamma statistics
gamma_mean <- mean(gamma_estimates, na.rm = TRUE)
gamma_sd_est <- sd(gamma_estimates, na.rm = TRUE)

# Calculate 95% CI for each simulation based on the overall distribution
gamma_ci_lower_est <- gamma_estimates - 1.96 * gamma_sd_est
gamma_ci_upper_est <- gamma_estimates + 1.96 * gamma_sd_est

# Calculate coverage
gamma_covered <- (gamma_ci_lower_est <= gamma_true) & 
  (gamma_ci_upper_est >= gamma_true)
gamma_coverage_percent <- mean(gamma_covered, na.rm = TRUE) * 100
gamma_n_covered <- sum(gamma_covered, na.rm = TRUE)
gamma_n_total <- sum(!is.na(gamma_covered))

# ==============================================================================
# BOXPLOT WITH COVERAGE RATES - ALL BLUE
# ==============================================================================

# Set up the plotting area
par(mfrow = c(1, 1), mar = c(8, 4, 4, 2))

# Prepare data
all_data <- c()
groups <- c()

# Add beta data (all K pollutants)
for(k in 1:K) {
  all_data <- c(all_data, beta_storage[valid_sims, k])
  groups <- c(groups, rep(paste0("β", k), sum(valid_sims)))
}

# Add gamma data
all_data <- c(all_data, gamma_storage[valid_sims])
groups <- c(groups, rep("γ", sum(valid_sims)))

# Create factor with proper order
groups <- factor(groups, levels = c(paste0("β", 1:K), "γ"))

# Create boxplot with NOTCH (recessed median) - ALL BLUE
boxplot(all_data ~ groups,
        xlab = "",
        ylab = "Estimate",
        col = c(rep(rgb(0.2, 0.4, 0.6, 0.7), K), 
                rgb(0.2, 0.4, 0.6, 0.7)),  # Changed gamma to blue
        border = "black",
        notch = TRUE,  # Creates the recessed median
        las = 1,
        cex.axis = 0.9,
        ylim = range(all_data, beta_true, gamma_true) * 1.1,
        xaxt = "n")

# Add custom x-axis labels with coverage rates
axis_labels <- c()
for(k in 1:K) {
  coverage <- round(beta_coverage_results$coverage_percent[k], 1)
  axis_labels <- c(axis_labels, paste0("β", k, "\n(", coverage, "%)"))
}
# Add gamma with its coverage rate
axis_labels <- c(axis_labels, paste0("γ\n(", round(gamma_coverage_percent, 1), "%)"))

# Add the axis
axis(1, at = 1:(K + 1), labels = axis_labels, las = 1, cex.axis = 0.85)

# Redraw boxplots on top (ALL BLUE)
boxplot(all_data ~ groups,
        xlab = "",
        ylab = "Estimate",
        col = c(rep(rgb(0.2, 0.4, 0.6, 0.7), K), 
                rgb(0.2, 0.4, 0.6, 0.7)),  # Changed gamma to blue
        border = "black",
        notch = TRUE,
        las = 1,
        cex.axis = 0.9,
        ylim = range(all_data, beta_true, gamma_true) * 1.1,
        xaxt = "n",
        add = TRUE)

# Add the axis again
axis(1, at = 1:(K + 1), labels = axis_labels, las = 1, cex.axis = 0.85)

# Add true values as points (ALL BLUE)
for(k in 1:K) {
  points(k, beta_true[k], col = "red", pch = 19, cex = 1.2)
}
points(K + 1, gamma_true, col = "red", pch = 19, cex = 1.2)

# Add mean values as diamonds (ALL DARK BLUE)
for(k in 1:K) {
  points(k, mean(beta_storage[valid_sims, k], na.rm = TRUE), 
         col = "green", pch = 18, cex = 1.2)
}
points(K + 1, mean(gamma_storage[valid_sims], na.rm = TRUE), 
       col = "green", pch = 18, cex = 1.2)

# Add legend
legend("topright", 
       legend = c("True Values", "Mean Estimates"), 
       col = c("red", "green"), 
       pch = c(19, 18),
       cex = 0.5,
       bg = "white")

# ==============================================================================
# PRINT COVERAGE SUMMARY
# ==============================================================================

cat("\n=== COMPREHENSIVE COVERAGE SUMMARY ===\n")
cat("========================================================================\n")
cat(sprintf("%-10s %-15s %-15s %-10s %-10s\n", 
            "Parameter", "Coverage (%)", "Covered/Total", "True Value", "Mean Est."))
cat("========================================================================\n")
for(k in 1:K) {
  cat(sprintf("β%d        %-15.1f %-15s %-10.3f %-10.3f\n", 
              k, 
              beta_coverage_results$coverage_percent[k],
              paste0(beta_coverage_results$n_covered[k], "/", 
                     beta_coverage_results$n_total[k]),
              beta_true[k],
              mean(beta_storage[valid_sims, k], na.rm = TRUE)))
}
cat(sprintf("γ         %-15.1f %-15s %-10.3f %-10.3f\n", 
            gamma_coverage_percent,
            paste0(gamma_n_covered, "/", gamma_n_total),
            gamma_true,
            mean(gamma_storage[valid_sims], na.rm = TRUE)))
cat("========================================================================\n")

dev.off()




# ==============================================================================
# WAIC ANALYSIS
# ==============================================================================

# Remove NA values from WAIC storage
valid_waic <- waic_storage[valid_sims & !is.na(waic_storage)]
n_valid_waic <- length(valid_waic)

cat("\n=== WAIC RESULTS ===\n")
cat("----------------------------------------\n")
cat(sprintf("Number of simulations with valid WAIC: %d\n", n_valid_waic))
cat(sprintf("Mean WAIC: %.2f\n", mean(valid_waic)))
cat(sprintf("Median WAIC: %.2f\n", median(valid_waic)))
cat(sprintf("SD of WAIC: %.2f\n", sd(valid_waic)))
cat(sprintf("Range of WAIC: [%.2f, %.2f]\n", min(valid_waic), max(valid_waic)))
cat(sprintf("95%% WAIC Interval: [%.2f, %.2f]\n", 
            quantile(valid_waic, 0.025), 
            quantile(valid_waic, 0.975)))
