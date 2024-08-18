#' Probabilities for the Centered Partition Process
#'
#' This function specifies the centered partition distribution for given an
#' anchor partition, shrinkage, concentration, and discount
#' parameters.
#'
#' @inheritParams CRPPartition
#' @inheritParams ShrinkagePartition
#' @param shrinkage A scalar giving the value of the shrinkage parameter.
#' @param useVI Should the distance between a particular partition and the
#'   anchor partition be measured using the variation of information
#'   (\code{TRUE}) or using Binder loss (\code{FALSE})?
#' @param a A nonnegative scalar influencing the relative cost of placing two items
#'   in separate clusters when in truth they belong to the same cluster.  This
#'   defaults to \eqn{1}, meaning equal costs.
#'
#' @return A numeric vector giving either probabilities or log probabilities for
#'   the supplied partitions.
#'
#' @example man/examples/CenteredPartition.R
#' @export
#'
CenteredPartition <- function(anchor, shrinkage, baseline, useVI=TRUE, a=1.0) {
  anchor <- coerceAnchor(anchor)
  nItems <- length(anchor)
  shrinkage <- coerceDouble(shrinkage, "shrinkage")
  if ( shrinkage < 0.0 ) stop("'shrinkage' must be nonnegative.")
  checkBaseline(baseline, nItems)
  useVI <- coerceLogical(useVI, "useVI")
  a <- coerceDouble(a,"a")
  if ( a < 0.0 ) stop("'a' must be nonnegative.")
  distrEnv("CenteredPartition", list(nItems=nItems, anchor=anchor, shrinkage=shrinkage, baseline=baseline, useVI=useVI, a=a))
}

#' @export
print.CenteredPartition <- function(x, ...) {
  cat("\nCentered partition distribution\n\n")
  print(distrAsList(x))
}
