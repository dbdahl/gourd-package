#' Probabilities for the Chinese Restaurant Process (CRP) Partition Distribution
#'
#' This function specifies the Chinese restaurant process (CRP) partition
#' distribution for given concentration and discount parameters.
#'
#' @param nItems The number of items in each partition, as an integer greater
#'   than or equal to \eqn{1}.
#' @param concentration The concentration parameter, as a numeric value
#'   greater than \eqn{-1*discount}.
#' @param discount The discount parameter, as a numeric value in \eqn{[0,1)}.
#'
#' @return An object of class \code{PartitionDistribution} representing this
#'   partition distribution.
#'
#' @example man/examples/CRPPartition.R
#' @export
#'
CRPPartition <- function(nItems, concentration, discount=0) {
  nItems <- coerceNItems(nItems)
  md <- coerceConcentrationDiscount(concentration, discount)
  distrEnv("CRPPartition", list(nItems=nItems, concentration=md$concentration, discount=md$discount))
}

#' @export
print.CRPPartition <- function(x, ...) {
  cat("\nChinese restaurant process partition distribution\n\n")
  print(distrAsList(x))
}
