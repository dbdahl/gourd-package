#' Probabilities for the Shrinkage Partition Distribution
#'
#' This function specifies the shrinkage partition distribution given an anchor
#' partition, shrinkage, permutation and a baseline distribution.
#'
#' @param anchor An integer vector giving the anchor partition
#'   (a.k.a., center partition, location partition).
#' @param shrinkage A numeric vector of length equal to the length of
#'   \code{anchor} (i.e., the number of items) giving the shrinkage
#'   probability for each item. This can also be a scalar, in which case that
#'   value is used for each item.
#' @param permutation A vector containing the integers \eqn{1, 2, \ldots, n}
#'   giving the order in which items are allocated to the partition.
#' @param grit A numeric value controlling the amount of clustering, with small
#'   values encouraging few clusters and large values encouraging more clusters.
#'   Values between 0 and 1 (exclusive) ensure that, as shrinakge goes to
#'   infinity, the partition distribution concentrates on the anchor partition.
#' @param baseline An object of class \code{PartitionDistribution}
#'   representing a partition distribution.  Currently, only
#'   \code{\link{UniformPartition}}, \code{\link{JensenLiuPartition}} and
#'   \code{\link{CRPPartition}} are supported.
#'
#' @return An object of class \code{PartitionDistribution} representing this
#'   partition distribution.
#'
#' @example man/examples/ShrinkagePartition.R
#' @export
#'
ShrinkagePartition <- function(anchor, shrinkage, permutation, grit, baseline) {
  anchor <- coerceAnchor(anchor)
  nItems <- length(anchor)
  shrinkage <- coerceShrinkageVector(shrinkage, nItems)
  permutation <- coercePermutation(permutation, nItems)
  checkBaseline(baseline, nItems)
  checkGrit(grit)
  distrEnv("ShrinkagePartition", list(nItems=nItems, anchor=anchor, shrinkage=shrinkage, permutation=permutation, .permutation=permutation-1L, grit=grit, baseline=baseline))
}

#' @export
print.ShrinkagePartition <- function(x, ...) {
  cat("\nShrinkage partition distribution\n\n")
  print(distrAsList(x))
}
