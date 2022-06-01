#' @export
state_r2rust <- function(state) {
  .Call(.state_r2rust, state)
}

#' @export
state_rust2r <- function(state) {
  .Call(.state_rust2r, state)
}

#' @export
hyperparameters_r2rust <- function(hyperparameters) {
  .Call(.hyperparameters_r2rust, hyperparameters)
}

#' @export
data_r2rust <- function(data) {
  .Call(.data_r2rust, data)
}

#' @export
fit <- function(data, state, hyperparameters, fixed=rep(FALSE,4), nUpdates=500, thin=10, progress=TRUE) {
  # Verify data
  if ( ! is.list(data) || length(data) != 3 || any(names(data) != c("response", "global_covariates", "clustered_covariates")) ) {
    stop("'data' must be a named list of elements: 1. 'response', 2. 'global_covariates', 3. 'clustered_covariates'")
  }
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
  # Verify state
  if ( ! is.list(state) || length(data) != 5 || any(names(state) != c("precision_response", "global_coefficients", "clustering", "clustered_coefficients", "permutation")) ) {
    stop("'state' must be a named list of elements: 1. 'precision_response', 2. 'global_coefficients', 3. 'clustering', 4. 'clustered_coefficients, 5. 'permutation'")
  }
  if ( ! is.numeric(state$precision_response) || ! is.numeric(state$global_coefficients) || ! is.numeric(state$clustering) || ! is.numeric(state$clustered_coefficients) || ! is.numeric(state$permutation) ) {
    stop("Elements of 'state' must be numeric values.")
  }
  if ( length(state$precision_responses) != 1 || state$precision_response <= 0.0 ) {
    stop("'state$precision_response' must be a strictly positive scalar.")
  }
  n_global_coefficients <- ncol(data$global_covariates)
  if ( length(state$global_coefficients) != n_global_coefficients ) {
    stop("Inconsistent number of global covariates.")
  }
  if ( length(state$clustering) != n_items ) {
    stop("'state$clustering' indicates an inconsistent number of observations.")
  }
  if ( ! is.list(state$clustered_coefficients) ) {
    stop("'state$clustered_coefficients' must be a list.")
  }
  if ( length(state$clustered_covariates) != max(state$clustering) ) {
    stop("Inconsistent number of clusters.")
  }
  n_clustered_coefficients <- ncol(data$clustered_covariates)
  for ( coef in state$clustered_coefficients ) {
    if ( ! is.numeric(coef) || length(coef) != n_clustered_coefficients ) {
      stop("Inconsistent number of clustered covariates.")
    }
  }
  state <- .Call(.state_r2rust, state)
  # Verify hyperparameters
  if ( ! is.list(hyperparameters) || length(hyperparameters) != 6 || any(names(hyperparameters) != c("precision_response_shape", "precision_response_rate", "global_coefficients_mean", "global_coefficients_precision", "clustered_coefficients_mean", "clustered_coefficients_precision")) ) {
    stop("'hyperparameters' must be a named list of elements: 1. 'precision_response_shape', 2. 'precision_response_rate', 3. 'global_coefficients_mean', 4. 'global_coefficients_precision', 5. 'clustered_coefficients_mean', 6. 'clustered_coefficients_precision'")
  }
  if ( ! is.numeric(hyperparameters$precision_response_shape) || ! is.numeric(hyperparameters$precision_response_rate) || ! is.numeric(hyperparameters$global_coefficients_mean) || ! is.numeric(hyperparameters$global_coefficients_precision)  || ! is.numeric(hyperparameters$clustered_coefficients_mean) || ! is.numeric(hyperparameters$clustered_coefficients_precision) ) {
    stop("Elements of 'hyperparameters' must be numeric values.")
  }
  if ( length(hyperparameters$precision_responses_shape) != 1 || hyperparameters$precision_response_shape <= 0.0 ) {
    stop("'hyperparameters$precision_response_shape' must be a strictly positive scalar.")
  }
  if ( length(hyperparameters$precision_responses_rate) != 1 || hyperparameters$precision_response_rate <= 0.0 ) {
    stop("'hyperparameters$precision_response_rate' must be a strictly positive scalar.")
  }
  if ( length(hyperparameters$global_coefficients_mean) != n_global_coefficients ) {
    stop("Inconsistent number of global covariates.")
  }
  if ( ! is.matrix(hyperparameters$global_coefficients_precision) || dim(hyperparameters$global_coefficients_precision) != rep(n_global_coefficients,2) ) {
    stop("Inconsistent number of global covariates.")
  }
  if ( length(hyperparameters$clustered_coefficients_mean) != n_clustered_coefficients ) {
    stop("Inconsistent number of clustered covariates.")
  }
  if ( ! is.matrix(hyperparameters$clustered_coefficients_precision) || dim(hyperparameters$clustered_coefficients_precision) != rep(n_clustered_coefficients,2) ) {
    stop("Inconsistent number of clustered covariates.")
  }
  hyperparameters <- .Call(.hyperparameters_r2rust, hyperparameters)
  # Run MCMC
  samples <- vector(mode="list", floor(nUpdates/thin))
  if ( progress ) { pb <- txtProgressBar(length(samples)) }
  for ( i in seq_along(samples) ) {
    state <- .Call(.fit, thin, data, state, fixed, hyperparameters)
    samples[[i]] <- state_rust2r(state)
    if ( progress ) setTxtProgressBar(pb, i)
  }
  if ( progress ) close(pb)
}
