#' Probabilities for the Jensen Liu Partition Distribution
#'
#' This function specifies the Jensen Liu (2008) partition distribution for given
#' concentration and permutation parameters.
#'
#' @inheritParams CRPPartition
#' @inheritParams ShrinkagePartition
#'
#' @return An object of class \code{PartitionDistribution} representing this
#'   partition distribution.
#'
#' @example man/examples/JensenLiuPartition.R
#' @export
#'
JensenLiuPartition <- function(concentration, permutation) {
  md <- coerceConcentrationDiscount(concentration, 0.0)
  nItems <- length(permutation)
  permutation <- coercePermutation(permutation, nItems)
  distrEnv("JensenLiuPartition", list(nItems=nItems, concentration=md$concentration, permutation=permutation, .permutation=permutation-1L))
}

#' @export
print.JensenLiuPartition <- function(x, ...) {
  cat("\nJensen Liu partition distribution\n\n")
  print(distrAsList(x))
}
