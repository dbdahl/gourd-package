#' Sample From a Partition Distribution
#'
#' This function samples from a partition distribution.
#'
#' Note that the centered partition distribution \code{\link{CenteredPartition}}
#' uses MCMC to sample.
#'
#' @param distr A specification of the partition distribution, i.e., an object
#'   of class \code{PartitionDistribution} as returned by, for example, a
#'   function such as \code{\link{CRPPartition}}.
#' @param nSamples An integer giving the number of partitions to sample.
#' @param randomizePermutation Should the permutation be uniformly randomly
#'   sampled for each partition?
#' @param randomizeShrinkage Should the shrinkage be random for each
#'   sample?  Specifically, the shrinkage is the same for every
#'   observations and sampled from a gamma distribution with parameters
#'   \code{shrinkage_shape} and \code{shrinakge_rate}.
#' @param randomizeGrit Should the grit of the \code{\link{ShrinkagePartition}}
#'   be sampled from a beta distribution for each partition? This is ignored
#'   for other distributions.
#' @param shrinkage_shape The shape parameter of the gamma distribution for
#'   randomizing the shrinkage.
#' @param shrinkage_rate The rate parameter of the gamma distribution for
#'   randomizing the shrinkage.
#' @param grit_shape1 The first parameter of the beta distribution for
#'   randomizing the grit.
#' @param grit_shape2 The first parameter of the beta distribution for
#'   randomizing the grit.
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
samplePartition <- function(distr, nSamples, randomizePermutation=FALSE, randomizeShrinkage=c("fixed","common","cluster","idiosyncratic")[1], randomizeGrit=FALSE, shrinkage_shape=5.0, shrinkage_rate=1.0, grit_shape1=1.0, grit_shape2=1.0, nCores=0) {
  UseMethod("samplePartition")
}

#' @export
#'
samplePartition.default <- function(distr, nSamples, randomizePermutation=FALSE, randomizeShrinkage=c("fixed","common","cluster","idiosyncratic")[1], randomizeGrit=FALSE, shrinkage_shape=5.0, shrinkage_rate=1.0, grit_shape1=1.0, grit_shape2=1.0, nCores=0) {
  nSamples <- coerceInteger(nSamples)
  if ( nSamples < 1 ) stop("'nSamples' should be at least one.")
  if ( ( length(randomizeShrinkage) != 1 ) || ( ! randomizeShrinkage %in% c("fixed","common","cluster","idiosyncratic") ) ) stop("'randomizeShrinkage' has an invalid value.")
  if ( inherits(distr,c("LocationScalePartition")) && ( ! randomizeShrinkage %in% c("fixed","common") ) ) stop("'randomizeShrinkage' must be 'fixed' or 'common' for the LocationScalePartition distribution.")
  p <- mkDistrPtr(distr)
  .Call(.samplePartition, nSamples, distr$nItems, p, isTRUE(randomizePermutation), randomizeShrinkage, randomizeGrit, shrinkage_shape, shrinkage_rate, grit_shape1, grit_shape2, nCores)
}
