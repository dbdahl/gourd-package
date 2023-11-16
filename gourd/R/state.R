#' @export
fit <- function(data, state, hyperparameters, partitionDistribution=CRPPartition(n_items, 1), nIterations=1000, burnin=500, thin=10,
                mcmcTuning=list(TRUE, TRUE, TRUE, TRUE, length(data$response)/2, 1.0), missingItems=integer(0), validationData=NULL,
                save=list(samples=TRUE, logLikelihoodContributions=c("none", "all", "missing", "validation")[1]), progress=TRUE) {
  # Verify data
  verifyData <- function(data) {
    if ( ! is.list(data) || length(data) != 4 || any(names(data) != c("response", "global_covariates", "clustered_covariates", "item_sizes")) ) {
      stop("'data' must be a named list of elements: 1. 'response', 2. 'global_covariates', 3. 'clustered_covariates', 4. 'item_sizes'")
    }
    data$global_covariates <- as.matrix(data$global_covariates)
    data$clustered_covariates <- as.matrix(data$clustered_covariates)
    if ( ! is.numeric(data$response) || ! is.numeric(data$global_covariates) || ! is.numeric(data$clustered_covariates) || ! is.numeric(data$item_sizes) ) {
      stop("Elements of 'data' must be numeric values.")
    }
    if ( ! is.matrix(data$global_covariates) || ! is.matrix(data$clustered_covariates) ) {
      stop("'data$global_covariates' and 'data$clustered_covariates' must be matrices.")
    }
    if ( any(data$item_sizes < 1L) ) {
      stop("All 'item_sizes' must be at least 1.")
    }
    n_items <- length(data$item_sizes)
    n_observations <- length(data$response)
    if ( sum(data$item_sizes) != n_observations ) {
      stop("Elements of 'data' indicate an inconsistent number of items.")
    }
    if ( ( nrow(data$global_covariates) != n_observations ) || ( nrow(data$clustered_covariates) != n_observations ) ) {
      stop("Elements of 'data' indicate an inconsistent number of observations.")
    }
    if ( ( length(missingItems) > 0 ) && ( ! is.numeric(missingItems) || min(missingItems) < 1 || max(missingItems) > n_items ) ) {
      stop("Elements of 'missing' are out of range.")
    }
    list(obj = .Call(.data_r2rust, data, missingItems), n_items = n_items, n_global_coefficients = ncol(data$global_covariates), n_clustered_coefficients = ncol(data$clustered_covariates))
  }
  dataList <- verifyData(data)
  if (!is.null(validationData)) {
    validationDataList <- verifyData(validationData)
    if (validationDataList$n_items != dataList$n_items) {
      stop("'validationData' does not have the same number of items as 'data'.")
    }
    if (validationDataList$n_global_coefficients != dataList$n_global_coefficients) {
      stop("'validationData' does not have the same number of global coefficients as 'data'.")
    }
    if (validationDataList$n_clustered_coefficients != dataList$n_clustered_coefficients) {
      stop("'validationData' does not have the same number of clustered coefficients as 'data'.")
    }
    validationData <- validationDataList$obj
  }
  data <- dataList$obj
  n_items <- dataList$n_items
  n_global_coefficients <- dataList$n_global_coefficients
  n_clustered_coefficients <- dataList$n_clustered_coefficients
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
  hyperparameters <- .Call(.hyperparameters_r2rust, hyperparameters)
  monitor <- .Call(.monitor_new)
  partitionDistribution <- pumpkin::mkDistrPtr(partitionDistribution)
  rngs <- .Call(.rngs_new)
  # Run MCMC
  if ( progress ) cat("Burning in...")
  permutation_bucket <- integer(n_items)
  shrinkage_bucket <- numeric(n_items)
  grit_bucket <- numeric(1L)
  .Call(.fit, burnin, data, state, hyperparameters, monitor, partitionDistribution, mcmcTuning, permutation_bucket, shrinkage_bucket, grit_bucket, rngs)
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
      shrinkage=matrix(0, nrow=nSamples, ncol=n_items),
      grit=numeric(nSamples)
    )
  }
  if ( save$logLikelihoodContributions != "none" ) {
    ncol <- if (save$logLikelihoodContributions %in% c("all", "validation")) {
      n_items
    } else if (save$logLikelihoodContributions == "missing") {
      length(missingItems)
    } else {
      stop("Unsupported option.")
    }
    logLikeContr <- matrix(0.0, nrow=nSamples, ncol=ncol)
  }
  if ( progress ) { pb <- txtProgressBar(0,nSamples,style=3) }
  for ( i in seq_len(nSamples) ) {
    .Call(.fit, thin, data, state, hyperparameters, monitor, partitionDistribution, mcmcTuning, permutation_bucket, shrinkage_bucket, grit_bucket, rngs)
    if ( save$logLikelihoodContributions != "none" ) {
      logLikeContr[i,] <- if ( save$logLikelihoodContributions == "all" ) {
        .Call(.log_likelihood_contributions, state, data)
      } else if (save$logLikelihoodContributions == "validation") {
        .Call(.log_likelihood_contributions, state, validationData)
      } else if (save$logLikelihoodContributions == "missing") {
        .Call(.log_likelihood_contributions_of_missing, state, data)
      } else {
        stop("Unsupported option.")
      }
    }
    if ( save$samples ) {
      tmp <- .Call(.state_rust2r, state)
      samples$precision_response[i] <- tmp[[1]]
      samples$global_coefficients[i,] <- tmp[[2]]
      samples$clustered_coefficients[[i]] <- tmp[[4]]
      samples$clustering[i,] <- tmp[[3]]
      samples$permutation[i,] <- permutation_bucket
      samples$shrinkage[i,] <- shrinkage_bucket
      samples$grit[i] <- grit_bucket
    }
    if ( progress ) setTxtProgressBar(pb, i)
  }
  if ( progress ) close(pb)
  result <- list(rates = c(permutation_acceptance_rate = .Call(.monitor_rate, monitor)))
  if ( save$samples ) {
    result <- c(result, list(samples=samples))
  }
  if ( save$logLikelihoodContributions != "none" ) {
    result <- c(result, list(logLikelihoodContributions=logLikeContr))
  }
  result
}

#' @export
fit_hierarchical_model <- function(all, unit_mcmc_tuning, global_hyperparameters, global_mcmc_tuning) {
  check_list(unit_mcmc_tuning, "bbbbid")
  check_list(global_hyperparameters, "ddddidd")
  check_list(global_mcmc_tuning, "iiibiidl")
  all_ptr <- .Call(.all, all)
  .Call(.fit_hierarchical_model, all_ptr, unit_mcmc_tuning, global_hyperparameters, global_mcmc_tuning)
}

#' @export
fit_temporal_model <- function(all, unit_mcmc_tuning, global_hyperparameters, global_mcmc_tuning) {
  check_list(unit_mcmc_tuning, "bbbbid")
  check_list(global_hyperparameters, "ddidd")
  check_list(global_mcmc_tuning, "iiibiidl")
  all_ptr <- .Call(.all, all)
  .Call(.fit_temporal_model, all_ptr, unit_mcmc_tuning, global_hyperparameters, global_mcmc_tuning)
}

check_list <- function(x, arg_types) {
  name <- deparse(substitute(x))
  arg_types <- strsplit(arg_types, "")[[1]]
  if (!is.list(x) || length(x) != length(arg_types)) {
    stop(sprintf("'%s' should be a list of length %s.", name, length(arg_types)))
  }
  for (i in seq_along(arg_types)) {
    y <- x[[i]]
    t <- arg_types[[i]]
    if (length(y) != 1 && t != "l") stop(sprintf("Element %s of '%s' should be a scalar.", i, name))
    if (t == "b") {
      if (!is.logical(y)) stop(sprintf("Element %s of '%s' should be a logical.", i, name))
    } else if (t == "i") {
      if (!is.numeric(y)) stop(sprintf("Element %s of '%s' should be an integer.", i, name))
    } else if (t == "d") {
      if (!is.numeric(y)) stop(sprintf("Element %s of '%s' should be a double.", i, name))
    } else if (t == "l") {
      if (!is.list(y) && !is.null(y)) stop(sprintf("Element %s of '%s' should be a list.", i, name))
    } else {
      stop("Unrecognized type")
    }
  }
}
