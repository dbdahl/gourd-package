#' @export
fit <- function(data, state, hyperparameters, partitionDistribution=CRPPartition(n_items, 1), nIterations=1000, burnin=500, thin=10, mcmcTuning=list(TRUE, TRUE, TRUE, TRUE, length(data$response)/2, 1.0), missingItems=integer(0), save=list(samples=TRUE, logLikelihoodContributions=c("none", "all", "missing")[1]), progress=TRUE) {
  # Verify data
  if ( ! is.list(data) || length(data) != 3 || any(names(data) != c("response", "global_covariates", "clustered_covariates")) ) {
    stop("'data' must be a named list of elements: 1. 'response', 2. 'global_covariates', 3. 'clustered_covariates'")
  }
  data$global_covariates <- as.matrix(data$global_covariates)
  data$clustered_covariates <- as.matrix(data$clustered_covariates)
  if ( ! is.numeric(data$response) || ! is.numeric(data$global_covariates) || ! is.numeric(data$clustered_covariates) ) {
    stop("Elements of 'data' must be numeric values.")
  }
  if ( ! is.matrix(data$global_covariates) || ! is.matrix(data$clustered_covariates) ) {
    stop("'data$global_covariates' and 'data$clustered_covariates' must be matrices.")
  }
  n_items <- length(data$response)
  if ( ( nrow(data$global_covariates) != n_items ) || ( nrow(data$clustered_covariates) != n_items ) ) {
    stop("Elements of 'data' indicate an inconsistent number of observations.")
  }
  if ( ( length(missingItems) > 0 ) && ( ! is.numeric(missingItems) || min(missingItems) < 1 || max(missingItems) > n_items ) ) {
    stop("Elements of 'missing' are out of range.")
  }
  n_global_coefficients <- ncol(data$global_covariates)
  n_clustered_coefficients <- ncol(data$clustered_covariates)
  data <- .Call(.data_r2rust, data, missingItems)
  # Verify state
  if ( ! is.list(state) || length(state) != 4 || any(names(state) != c("precision_response", "global_coefficients", "clustering", "clustered_coefficients")) ) {
    stop("'state' must be a named list of elements: 1. 'precision_response', 2. 'global_coefficients', 3. 'clustering', 4. 'clustered_coefficients")
  }
  state$global_coefficients <- as.matrix(state$global_coefficients)
  if ( ! is.numeric(state$precision_response) || ! is.numeric(state$global_coefficients) || ! is.numeric(state$clustering) ) {
    stop("Elements of 'state' must be numeric values.")
  }
  if ( length(state$precision_response) != 1 || state$precision_response <= 0.0 ) {
    stop("'state$precision_response' must be a strictly positive scalar.")
  }
  if ( length(state$global_coefficients) != n_global_coefficients ) {
    stop("Inconsistent number of global covariates.")
  }
  if ( length(state$clustering) != n_items ) {
    stop("'state$clustering' indicates an inconsistent number of observations.")
  }
  if ( any( state$clustering %% 1 != 0 ) ) {
    stop("'state$clustering' must contain only integers.")
  }
  if ( min(state$clustering) != 1 ) {
    stop("Smallest label in 'state$clustering' must be 1.")
  }
  if ( ! is.list(state$clustered_coefficients) ) {
    stop("'state$clustered_coefficients' must be a list.")
  }
  if ( length(state$clustered_coefficients) != max(state$clustering) ) {
    stop("Inconsistent number of clusters.")
  }
  for ( coef in state$clustered_coefficients ) {
    if ( ! is.numeric(coef) || length(coef) != n_clustered_coefficients ) {
      stop("Inconsistent number of clustered covariates.")
    }
  }
  state <- .Call(.state_r2rust, state)
  # Verify hyperparameters
  if ( ! is.list(hyperparameters) || length(hyperparameters) != 9 || any(names(hyperparameters) != c("precision_response_shape", "precision_response_rate", "global_coefficients_mean", "global_coefficients_precision", "clustered_coefficients_mean", "clustered_coefficients_precision", "shrinkage_reference", "shrinkage_shape", "shrinkage_rate")) ) {
    stop("'hyperparameters' must be a named list of elements: 1. 'precision_response_shape', 2. 'precision_response_rate', 3. 'global_coefficients_mean', 4. 'global_coefficients_precision', 5. 'clustered_coefficients_mean', 6. 'clustered_coefficients_precision', 7. 'shrinkage_reference', 8. 'shrinkage_shape', 9. 'shrinkage_rate'.")
  }
  if ( ! is.numeric(hyperparameters$precision_response_shape) || ! is.numeric(hyperparameters$precision_response_rate) || ! is.numeric(hyperparameters$global_coefficients_mean) || ! is.numeric(hyperparameters$global_coefficients_precision)  || ! is.numeric(hyperparameters$clustered_coefficients_mean) || ! is.numeric(hyperparameters$clustered_coefficients_precision) ) {
    stop("Elements of 'hyperparameters' must be numeric values.")
  }
  if ( length(hyperparameters$precision_response_shape) != 1 || hyperparameters$precision_response_shape <= 0.0 ) {
    stop("'hyperparameters$precision_response_shape' must be a strictly positive scalar.")
  }
  if ( length(hyperparameters$precision_response_rate) != 1 || hyperparameters$precision_response_rate <= 0.0 ) {
    stop("'hyperparameters$precision_response_rate' must be a strictly positive scalar.")
  }
  if ( length(hyperparameters$global_coefficients_mean) != n_global_coefficients ) {
    stop("Inconsistent number of global covariates.")
  }
  if ( ( ! is.matrix(hyperparameters$global_coefficients_precision) ) || ( any(dim(hyperparameters$global_coefficients_precision) != rep(n_global_coefficients,2)) ) ) {
    stop("Inconsistent number of global covariates.")
  }
  if ( length(hyperparameters$clustered_coefficients_mean) != n_clustered_coefficients ) {
    stop("Inconsistent number of clustered covariates.")
  }
  if ( ! is.matrix(hyperparameters$clustered_coefficients_precision) || any(dim(hyperparameters$clustered_coefficients_precision) != rep(n_clustered_coefficients,2)) ) {
    stop("Inconsistent number of clustered covariates.")
  }
  if ( length(hyperparameters$shrinkage_reference) != 1 || hyperparameters$shrinkage_reference < 1 || hyperparameters$shrinkage_reference > n_items ) {
    stop("'hyperparameters$shrinkage_reference' must be in [1, 2, ..., n_items].")
  }
  if ( length(hyperparameters$shrinkage_shape) != 1 || hyperparameters$shrinkage_shape <= 0.0 ) {
    stop("'hyperparameters$shrinkage_shape' must be a strictly positive scalar.")
  }
  if ( length(hyperparameters$shrinkage_rate) != 1 || hyperparameters$shrinkage_rate <= 0.0 ) {
    stop("'hyperparameters$shrinkage_rate' must be a strictly positive scalar.")
  }
  hyperparameters <- .Call(.hyperparameters_r2rust, hyperparameters)
  # Verify MCMC tuning parameters
  if ( ! is.list(mcmcTuning) || length(mcmcTuning) != 6 ) {
    stop("'mcmc_tuning' should be a list of length 6.")
  }
  monitor <- .Call(.monitor_new)
  partitionDistribution <- mkDistrPtr(partitionDistribution)  # DBD: Memory leak!!!!!
  rngs <- .Call(.rngs_new)
  # Run MCMC
  if ( progress ) cat("Burning in...")
  permutation_bucket <- integer(n_items)
  shrinkage_bucket <- numeric(n_items)
  tmp <- .Call(.fit, burnin, data, state, hyperparameters, monitor, partitionDistribution, mcmcTuning, missingItems, permutation_bucket, shrinkage_bucket, rngs)
  state <- tmp[[1]]
  monitor <- tmp[[2]]
  .Call(.monitor_reset, monitor);
  if ( progress ) cat("\r")
  nSamples <- floor((nIterations-burnin)/thin)
  if ( save$samples ) {
    samples <- list(
      precision_response=numeric(nSamples),
      global_coefficients=matrix(0.0, nrow=nSamples, ncol=n_global_coefficients),
      clustering=matrix(0L, nrow=nSamples, ncol=n_items),
      clustered_coefficients=vector(mode="list", nSamples),
      permutation=matrix(0L, nrow=nSamples, ncol=n_items),
      shrinkage=matrix(0, nrow=nSamples, ncol=n_items)
    )
  }
  if ( save$logLikelihoodContributions != "none" ) {
    ncol <- if ( save$logLikelihoodContributions == "all" ) n_items else length(missingItems)
    logLikeContr <- matrix(0.0, nrow=nSamples, ncol=ncol)
  }
  if ( progress ) { pb <- txtProgressBar(0,nSamples,style=3) }
  for ( i in seq_len(nSamples) ) {
    tmp <- .Call(.fit, thin, data, state, hyperparameters, monitor, partitionDistribution, mcmcTuning, missingItems, permutation_bucket, shrinkage_bucket, rngs)
    state <- tmp[[1]]
    monitor <- tmp[[2]]
    if ( save$logLikelihoodContributions != "none" ) {
      logLikeContr[i,] <- if ( save$logLikelihoodContributions == "all" ) {
        .Call(.log_likelihood_contributions, state, data)
      } else {
        .Call(.log_likelihood_contributions_of_missing, state, data)
      }
    }
    if ( save$samples ) {
      tmp <- .Call(.state_rust2r_as_reference, state)
      samples$precision_response[i] <- tmp[[1]]
      samples$global_coefficients[i,] <- tmp[[2]]
      samples$clustered_coefficients[[i]] <- tmp[[4]]
      samples$clustering[i,] <- tmp[[3]]
      samples$permutation[i,] <- permutation_bucket
      samples$shrinkage[i,] <- shrinkage_bucket
    }
    if ( progress ) setTxtProgressBar(pb, i)
  }
  if ( progress ) close(pb)
  rates <- c(permutation_acceptance_rate = .Call(.monitor_rate, monitor))
  .Call(.rust_free, data)
  .Call(.rust_free, state)
  .Call(.rust_free, hyperparameters)
  .Call(.rust_free, monitor)
  .Call(.rust_free, rngs[[1]])
  .Call(.rust_free, rngs[[2]])
  result <- list()
  if ( save$samples ) {
    result <- c(result, list(samples=samples, rates=rates))
  }
  if ( save$logLikelihoodContributions != "none" ) {
    result <- c(result, list(logLikelihoodContributions=logLikeContr, rates=rates))
  }
  result
}

fit_all <- function(all, shrinkage, nIterations, doBaselinePartition) {
  all_ptr <- .Call(.all, all)
  .Call(.fit_all, all_ptr, shrinkage, nIterations, doBaselinePartition)
}
