#' Probabilities for the Location Scale Partition Distribution
#'
#' This function specifies the location scale partition distribution for given
#' anchor partition, shrinkage, and permutation parameters.  The shrinkage parameter
#' is the reciprocal of the scale parameter.
#'
#' @inheritParams CRPPartition
#' @inheritParams ShrinkagePartition
#' @param shrinkage A scalar giving the value of the shrinkage parameter, i.e.,the
#'   reciprocal of the scale parameter.
#'
#' @return An object of class \code{PartitionDistribution} representing this
#'   partition distribution.
#'
#' @example man/examples/LocationScalePartition.R
#' @export
#'
LocationScalePartition <- function(anchor, shrinkage, concentration, permutation) {
  anchor <- coerceAnchor(anchor)
  shrinkage <- coerceShrinkage(shrinkage)
  md <- coerceConcentrationDiscount(concentration, 0.0);
  nItems <- length(anchor)
  permutation <- coercePermutation(permutation, nItems)
  distrEnv("LocationScalePartition", list(nItems=nItems, anchor=anchor, shrinkage=shrinkage, concentration=concentration, permutation=permutation, .permutation=permutation-1L))
}

#' @export
print.LocationScalePartition <- function(x, ...) {
  cat("\nLocation scale partition distribution\n\n")
  print(distrAsList(x))
}
