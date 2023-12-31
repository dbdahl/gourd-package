#' Probabilities for the Uniform Partition Distribution
#'
#' This function specifies the uniform partition distribution.
#'
#' @param nItems An integer giving the number of items in each partition.
#'
#' @return An object of class \code{PartitionDistribution} representing this
#'   partition distribution.
#'
#' @example man/examples/UniformPartition.R
#' @export
#'
UniformPartition <- function(nItems) {
  nItems <- coerceNItems(nItems)
  distrEnv("UniformPartition", list(nItems=nItems))
}

#' @export
print.UniformPartition <- function(x, ...) {
  cat("\nUniform partition distribution\n\n")
  print(distrAsList(x))
}
