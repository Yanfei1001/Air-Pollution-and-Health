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

# ==============================================================================
# ENHANCED VISUALIZATIONS
# ==============================================================================

# Create a comprehensive PDF report
# pdf("simulation_results.pdf", width = 11, height = 8.5)

# 1. Beta estimates with CI coverage visualization
par(mfrow = c(2, 3), mar = c(4, 4, 3, 2))
for(k in 1:K) {
  hist(beta_storage[valid_sims, k], breaks = 30, 
       col = rgb(0.2, 0.4, 0.6, 0.7), border = "white",
       main = paste("β", k, "- Distribution"),
       xlab = "Estimate", 
       sub = paste("True =", beta_true[k], 
                   "| Coverage =", round(beta_coverage_results$coverage_percent[k], 1), "%"))
  abline(v = beta_true[k], col = "red", lwd = 2, lty = 2)
  abline(v = mean(beta_storage[valid_sims, k], na.rm = TRUE), 
         col = "blue", lwd = 2, lty = 1)
  legend("topright", legend = c("True", "Mean Estimate"), 
         col = c("red", "blue"), lty = c(2, 1), lwd = 2, cex = 0.8)
}

# 2. Boxplots with coverage annotation
# par(mfrow = c(1, 1))
# boxplot(as.data.frame(beta_storage[valid_sims, ]), 
#         col = rgb(0.2, 0.4, 0.6, 0.7), 
#         main = "β Estimates with Coverage Percentages",
#         xlab = "Pollutant", ylab = "Estimate",
#         names = paste0("β", 1:K, "\n(", round(beta_coverage_results$coverage_percent, 1), "%)"))
# points(1:K, beta_true, col = "red", pch = 8, cex = 1.5, lwd = 2)
# legend("topright", legend = "True Values", col = "red", pch = 8, cex = 0.7)
# 

# Open PNG device
png("/Users/yanfei/Desktop/Manuscript Prep/Simulation/beta_estimates_plot.png", width = 8, height = 6, units = "in", res = 300)

# Create the plot
par(mfrow = c(1, 1))
boxplot(as.data.frame(beta_storage[valid_sims, ]), 
        col = rgb(0.2, 0.4, 0.6, 0.7), 
        # main = "β Estimates with Coverage Percentages",
        # xlab = "Pollutant", 
        ylab = "Estimate",
        names = paste0("β", 1:K, "\n(", round(beta_coverage_results$coverage_percent, 1), "%)"))
points(1:K, beta_true, col = "red", pch = 8, cex = 1.5, lwd = 2)
legend("topright", legend = "True Values", col = "red", pch = 8, cex = 0.7)

# Close the PNG device
dev.off()


# 3. Rho estimates with coverage
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
     main = "WAIC Distribution",
     xlab = "WAIC", 
     sub = paste("Mean =", round(mean(valid_waic), 2),
                 "| Range = [", round(min(valid_waic), 2), ",", 
                 round(max(valid_waic), 2), "]"))
abline(v = mean(valid_waic), col = "blue", lwd = 2, lty = 1)
abline(v = median(valid_waic), col = "red", lwd = 2, lty = 2)
legend("topright", legend = c("Mean", "Median"), 
       col = c("blue", "red"), lty = c(1, 2), lwd = 2,cex=0.8)

# Boxplot of WAIC
boxplot(valid_waic, 
        col = rgb(0.3, 0.5, 0.3, 0.7),
        main = "WAIC Boxplot",
        ylab = "WAIC",
        sub = paste("Range:", round(min(valid_waic), 2), "-", round(max(valid_waic), 2)))
points(1, mean(valid_waic), col = "blue", pch = 8, cex = 1.5)
text(1, mean(valid_waic), paste("Mean:", round(mean(valid_waic), 2)), pos = 2, col = "blue")









png("/Users/yanfei/Desktop/Manuscript Prep/Simulation/estimates_plot_MEME.png", width = 8, height = 6, units = "in", res = 300)

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






# 
# 
# 
# 
# 
# # ==============================================================================
# # SIMULATED VS PREDICTED MORTALITY PLOT
# # ==============================================================================
# 
# # For the last valid simulation, extract posterior predictions
# last_valid_sim <- which(valid_sims)[length(which(valid_sims))]
# 
# # Re-run JAGS for the last valid simulation to get predictions
# set.seed(100 + last_valid_sim)
# 
# # Generate data for this simulation (same as before)
# temperature <- 15 + 5 * sin(2 * pi * (1:N) / 52) + rnorm(N, 0, 1)
# population <- rep(100000, N)
# 
# # Generate AR(1) latent states
# Z_true <- matrix(NA, nrow = N, ncol = K)
# for(k in 1:K) {
#   Z_true[1, k] <- rnorm(1, 0, sigma_proc_true[k] / sqrt(1 - rho_true[k]^2))
#   for(t in 2:N) {
#     Z_true[t, k] <- rho_true[k] * Z_true[t-1, k] + rnorm(1, 0, sigma_proc_true[k])
#   }
# }
# 
# # Add measurement error
# Y_obs <- Z_true + rnorm(N * K, 0, sigma_me_true)
# 
# # Generate outcomes
# log_lambda <- alpha_true + Z_true %*% beta_true + gamma_true * temperature + log(population)
# Y_out <- rpois(N, exp(log_lambda))
# 
# # Prepare data for JAGS
# data_list <- list(
#   N = N, K = K,
#   Y_obs = Y_obs,
#   Y_out = Y_out,
#   temperature = temperature,
#   population = population
# )
# 
# # Fit model with predictions
# ar_model_pred <- "
# model {
#   # AR(1) latent process
#   for (k in 1:K) {
#     Z[1,k] ~ dnorm(0, tau_proc[k] * (1 - rho[k]^2))
#     for (t in 2:N) {
#       Z[t,k] ~ dnorm(rho[k] * Z[t-1,k], tau_proc[k])
#     }
#   }
#   
#   # Measurement model
#   for (t in 1:N) {
#     for (k in 1:K) {
#       Y_obs[t,k] ~ dnorm(Z[t,k], tau_me)
#     }
#   }
#   
#   # Outcome model with prediction
#   for (t in 1:N) {
#     Y_out[t] ~ dpois(lambda[t])
#     log(lambda[t]) <- alpha + inprod(beta[1:K], Z[t,1:K]) + 
#                       gamma * temperature[t] + log(population[t])
#     
#     # Posterior predictive distribution
#     Y_pred[t] ~ dpois(lambda[t])
#   }
#   
#   # Priors
#   for (k in 1:K) {
#     rho_raw[k] ~ dbeta(15, 5)
#     rho[k] <- 2 * rho_raw[k] - 1
#     tau_proc[k] ~ dgamma(2, 1)
#     sigma_proc[k] <- 1/sqrt(tau_proc[k])
#     beta[k] ~ dnorm(0, 0.01)
#   }
#   
#   tau_me ~ dgamma(10, 1)
#   sigma_me <- 1/sqrt(tau_me)
#   alpha ~ dnorm(0, 0.01)
#   gamma ~ dnorm(0, 0.01)
# }
# "
# 
# # Fit model with predictions
# fit_pred <- jags(data_list, 
#                  parameters.to.save = c("Y_pred", "lambda", "beta", "alpha", "gamma"),
#                  model.file = textConnection(ar_model_pred),
#                  n.chains = 2,
#                  n.iter = 5000,
#                  n.burnin = 1000,
#                  n.thin = 4,
#                  quiet = TRUE)
# 
# # Extract predictions
# Y_pred_samples <- fit_pred$BUGSoutput$sims.list$Y_pred
# lambda_samples <- fit_pred$BUGSoutput$sims.list$lambda
# 
# # Calculate summaries
# Y_pred_mean <- colMeans(Y_pred_samples)
# Y_pred_ci_lower <- apply(Y_pred_samples, 2, quantile, probs = 0.025)
# Y_pred_ci_upper <- apply(Y_pred_samples, 2, quantile, probs = 0.975)
# 
# lambda_mean <- colMeans(lambda_samples)
# lambda_ci_lower <- apply(lambda_samples, 2, quantile, probs = 0.025)
# lambda_ci_upper <- apply(lambda_samples, 2, quantile, probs = 0.975)
# 
# # Create the plot
# png("/Users/yanfei/Desktop/Manuscript Prep/Simulation/simulated_vs_predicted_mortality.png", 
#     width = 10, height = 6, units = "in", res = 300)
# 
# par(mfrow = c(1, 1), mar = c(4.5, 4.5, 3, 2))
# 
# # Plot 1: Simulated vs Predicted Mortality (Time Series)
# plot(1:N, Y_out, type = "l", col = "black", lwd = 2,
#      xlab = "Time", ylab = "Mortality Count",
#      main = "Simulated vs Predicted Mortality",
#      ylim = range(c(Y_out, Y_pred_ci_upper, Y_pred_ci_lower)))
# lines(1:N, Y_pred_mean, col = "red", lwd = 2)
# polygon(c(1:N, rev(1:N)), 
#         c(Y_pred_ci_lower, rev(Y_pred_ci_upper)),
#         col = rgb(1, 0, 0, 0.2), border = NA)
# legend("topright", 
#        legend = c("Simulated (True)", "Predicted Mean", "95% CI"),
#        col = c("black", "red", rgb(1, 0, 0, 0.3)),
#        lty = c(1, 1, 0),
#        lwd = c(2, 2, NA),
#        pch = c(NA, NA, 15),
#        pt.cex = c(NA, NA, 2),
#        bg = "white")
# 
# # Plot 2: Scatter plot with correlation
# plot(Y_out, Y_pred_mean, 
#      xlab = "Simulated Mortality", 
#      ylab = "Predicted Mortality",
#      main = paste("Correlation: r =", round(cor(Y_out, Y_pred_mean), 3)),
#      pch = 16, col = rgb(0.2, 0.4, 0.6, 0.6),
#      xlim = range(c(Y_out, Y_pred_ci_lower, Y_pred_ci_upper)),
#      ylim = range(c(Y_out, Y_pred_ci_lower, Y_pred_ci_upper)))
# abline(a = 0, b = 1, col = "red", lwd = 2, lty = 2)
# 
# # Add error bars
# segments(x0 = Y_out, y0 = Y_pred_ci_lower, 
#          x1 = Y_out, y1 = Y_pred_ci_upper,
#          col = rgb(0.2, 0.4, 0.6, 0.3))
# 
# # Add regression line
# fit_line <- lm(Y_pred_mean ~ Y_out)
# abline(fit_line, col = "blue", lwd = 2)
# legend("topleft",
#        legend = c("Identity Line", "Regression Line"),
#        col = c("red", "blue"),
#        lty = c(2, 1),
#        lwd = 2,
#        bg = "white")
# 
# # Add R-squared
# rsq <- summary(fit_line)$r.squared
# text(x = min(Y_out), y = max(Y_pred_mean), 
#      labels = paste("R² =", round(rsq, 3)),
#      pos = 4, cex = 0.9)
# 
# dev.off()
# 
# # Print summary statistics
# cat("\n=== MORTALITY PREDICTION ACCURACY ===\n")
# cat("----------------------------------------\n")
# cat(sprintf("Correlation coefficient: %.4f\n", cor(Y_out, Y_pred_mean)))
# cat(sprintf("R-squared: %.4f\n", rsq))
# cat(sprintf("Mean Absolute Error (MAE): %.2f\n", mean(abs(Y_out - Y_pred_mean))))
# cat(sprintf("Root Mean Square Error (RMSE): %.2f\n", sqrt(mean((Y_out - Y_pred_mean)^2))))
# cat(sprintf("Mean prediction CI width: %.2f\n", mean(Y_pred_ci_upper - Y_pred_ci_lower)))
