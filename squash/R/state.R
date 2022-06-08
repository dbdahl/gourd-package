#' @export
fit <- function(data, state, fixed=rep(FALSE,5), hyperparameters, partitionDistribution=CRPPartition(n_items, 1), nIterations=1000, burnin=500, thin=10, logLikelihoodContributions=FALSE, missingItems=integer(0), progress=TRUE) {
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
  if ( ! is.list(state) || length(state) != 5 || any(names(state) != c("precision_response", "global_coefficients", "clustering", "clustered_coefficients", "permutation")) ) {
    stop("'state' must be a named list of elements: 1. 'precision_response', 2. 'global_coefficients', 3. 'clustering', 4. 'clustered_coefficients, 5. 'permutation'")
  }
  state$global_coefficients <- as.matrix(state$global_coefficients)
  if ( ! is.numeric(state$precision_response) || ! is.numeric(state$global_coefficients) || ! is.numeric(state$clustering) || ! is.numeric(state$permutation) ) {
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
  state$clustering <- state$clustering - 1
  for ( coef in state$clustered_coefficients ) {
    if ( ! is.numeric(coef) || length(coef) != n_clustered_coefficients ) {
      stop("Inconsistent number of clustered covariates.")
    }
  }
  if ( length(state$permutation) != n_items ) {
    stop("'state$permutation' indicates an inconsistent number of observations.")
  }
  if ( length(unique(state$permutation)) != n_items ) {
    stop("'state$permutation' elements are not unique.")
  }
  if ( any( state$permutation %% 1 != 0 ) ) {
    stop("'state$permutation' must contain only integers.")
  }
  if ( ( min(state$permutation) != 1 ) || ( max(state$permutation) != n_items ) ) {
    stop("'state$permutation' must range between one and the number of items.")
  }
  state$permutation <- state$permutation - 1
  state <- .Call(.state_r2rust, state)
  # Verify hyperparameters
  if ( ! is.list(hyperparameters) || length(hyperparameters) != 6 || any(names(hyperparameters) != c("precision_response_shape", "precision_response_rate", "global_coefficients_mean", "global_coefficients_precision", "clustered_coefficients_mean", "clustered_coefficients_precision")) ) {
    stop("'hyperparameters' must be a named list of elements: 1. 'precision_response_shape', 2. 'precision_response_rate', 3. 'global_coefficients_mean', 4. 'global_coefficients_precision', 5. 'clustered_coefficients_mean', 6. 'clustered_coefficients_precision'")
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
  hyperparameters <- .Call(.hyperparameters_r2rust, hyperparameters)
  partitionDistribution <- pumpkin::mkDistrPtr(partitionDistribution)  # DBD: Memory leak!!!!!
  # Run MCMC
  if ( progress ) cat("Burning in...")
  state <- .Call(.fit, burnin, data, state, fixed, hyperparameters, partitionDistribution, missingItems)
  if ( progress ) cat("\r")
  nSamples <- floor((nIterations-burnin)/thin)
  samples <- list(
    precision_response=numeric(nSamples),
    global_coefficients=matrix(0.0, nrow=nSamples, ncol=n_global_coefficients),
    clustering=matrix(0L, nrow=nSamples, ncol=n_items),
    clustered_coefficients=vector(mode="list", nSamples),
    permutation=matrix(0L, nrow=nSamples, ncol=n_items)
  )
  if ( logLikelihoodContributions ) {
    logLikeContr <- matrix(0.0, nrow=nSamples, ncol=n_items)
  }
  if ( progress ) { pb <- txtProgressBar(0,nSamples,style=3) }
  for ( i in seq_len(nSamples) ) {
    state <- .Call(.fit, thin, data, state, fixed, hyperparameters, partitionDistribution, missingItems)
    if ( logLikelihoodContributions ) {
      logLikeContr[i,] <- .Call(.log_likelihood_contributions, state, data)
    }
    tmp <- .Call(.state_rust2r_as_reference,state)
    samples$precision_response[i] <- tmp[[1]]
    samples$global_coefficients[i,] <- tmp[[2]]
    samples$clustered_coefficients[[i]] <- tmp[[4]]
    samples$clustering[i,] <- tmp[[3]]
    samples$permutation[i,] <- tmp[[5]]
    if ( progress ) setTxtProgressBar(pb, i)
  }
  if ( progress ) close(pb)
  .Call(.rust_free, data)
  .Call(.rust_free, state)
  .Call(.rust_free, hyperparameters)
  result <- list(samples=samples)
  if ( logLikelihoodContributions ) {
    result <- c(result, list(logLikelihoodContributions=logLikeContr))
  }
  result
}
