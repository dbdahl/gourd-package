#' Sample From a Partition Distribution
#'
#' This function samples from a partition distribution.
#'
#' Note that the centered partition distribution \code{\link{CenteredPartition}}
#' is not supported.
#'
#' @param distr A specification of the partition distribution, i.e., an object
#'   of class \code{PartitionDistribution} as returned by, for example, a
#'   function such as \code{\link{CRPPartition}}.
#' @param nSamples An integer giving the number of partitions to sample.
#' @param randomizePermutation Should the permutation be uniformly randomly
#'   sampled for each partition?
#' @param randomizeShrinkage Should the shrinkage be random for each
#'   permutation?  Specifically, the shrinkage is the same for every
#'   observations and sampled from a beta distribution with parameters
#'   \code{shape1} and \code{shape2}, as scaled to have a maximum value of
#'   \code{max} (instead of 1).
#' @param max The scalar giving the maximum value of the shrinkage.
#' @param shape1 The first shape parameter of the beta distribution for
#'   randomizing the shrinkage.
#' @param shape2 The second shape parameter of the beta distribution for
#'   randomizing the shrinkage.
#' @param nCores The number of CPU cores to use. A value of zero indicates to
#'   use all cores on the system.
#'
#' @return An integer matrix containing a partition in each row using cluster
#'   label notation.
#'
#' @seealso \code{\link{CRPPartition}}, \code{\link{ShrinkagePartition}},
#'   \code{\link{LocationScalePartition}}, \code{\link{CenteredPartition}},
#'   \code{\link{prPartition}}
#'
#' @example man/examples/prPartition.R
#'
#' @export
#'
samplePartition <- function(distr, nSamples, randomizePermutation=FALSE, randomizeShrinkage=c("fixed","common","cluster","idiosyncratic")[1], max=4, shape1=1.0, shape2=1.0, nCores=0) {
  UseMethod("samplePartition")
}

#' @export
#'
samplePartition.default <- function(distr, nSamples, randomizePermutation=FALSE, randomizeShrinkage=c("fixed","common","cluster","idiosyncratic")[1], max=4, shape1=1.0, shape2=1.0, nCores=0) {
  nSamples <- coerceInteger(nSamples)
  if ( nSamples < 1 ) stop("'nSamples' should be at least one.")
  if ( ( length(randomizeShrinkage) != 1 ) || ( ! randomizeShrinkage %in% c("fixed","common","cluster","idiosyncratic") ) ) stop("'randomizeShrinkage' has an invalid value.")
  if ( inherits(distr,c("LocationScalePartition")) && ( ! randomizeShrinkage %in% c("fixed","common") ) ) stop("'randomizeShrinkage' must be 'fixed' or 'common' for the LocationScalePartition distribution.")
  p <- mkDistrPtr(distr, excluded = c("CenteredPartition"))
  .Call(.samplePartition, nSamples, distr$nItems, p, isTRUE(randomizePermutation), randomizeShrinkage, max, shape1, shape2, nCores)
}
