#' Probabilities for the Fixed Partition Distribution
#'
#' This function specifies the fixed partition distribution for a given anchor
#' partition.  This is a point-mass distribution at the anchor partition.
#'
#' @inheritParams ShrinkagePartition
#'
#' @return An object of class \code{PartitionDistribution} representing this
#'   partition distribution.
#'
#' @example man/examples/FixedPartition.R
#' @export
#'
FixedPartition <- function(anchor) {
  anchor <- coerceAnchor(anchor)
  distrEnv("FixedPartition", list(nItems=length(anchor), anchor=anchor))
}

#' @export
print.FixedPartition <- function(x, ...) {
  cat("\nFixed partition distribution\n\n")
  print(distrAsList(x))
}
