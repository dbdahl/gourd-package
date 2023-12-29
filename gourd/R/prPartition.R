#' Compute the (Log of) the Probability of a Partition
#'
#' This function computes (the log of) the probability of the supplied
#' partitions for the given partition distribution.
#'
#' @inheritParams samplePartition
#' @param partition A matrix of integers giving cluster labels on the rows. For
#'   partition \eqn{k}, items \eqn{i} and \eqn{j} are in the same cluster if and
#'   only if \code{partition[k,i] == partition[k,j]}.
#' @param log A logical indicating whether the probability (\code{FALSE}) or its
#'   natural logarithm (\code{TRUE}) is desired.
#'
#' @return A numeric vector giving either probabilities or log probabilities for
#'   the supplied partitions.
#'
#' @seealso \code{\link{CRPPartition}}, \code{\link{ShrinkagePartition}}
#'
#' @example man/examples/prPartition.R
#' @export
#'
prPartition <- function(distr, partition, log=TRUE) {
  if ( ! is.numeric(partition) ) stop("'partition' should be numeric.")
  storage.mode(partition) <- "double"
  if ( ! is.matrix(partition) ) partition <- matrix(partition, nrow=1)
  if ( distr$nItems != ncol(partition) ) stop("Number of items is not consistent.")
  nSamples <- nrow(partition)
  if ( nSamples == 0 ) return(numeric(0))
  logProbabilities <- .Call(.prPartition, partition, mkDistrPtr(distr))
  if ( inherits(distr, "CenteredPartition") ) {
    warning("Calculations for 'CenteredPartition' are not normalized (i.e., they don't sum to one across all partitions).")
  }
  if (isTRUE(log)) logProbabilities else exp(logProbabilities)
}
