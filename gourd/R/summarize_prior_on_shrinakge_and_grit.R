#' Summarize Implications of the Prior Distribution of the SP Distribution
#'
#' To aid in the prior elicitation process for the shrinkage and grit parameters
#' in the Shrinkage Partition (SP) distribution, this function computes the log
#' of the prior joint density of the shrinkage and grit parameters and the
#' expected Rand index and the expected entropy using samples from the SP
#' distribution for combinations of shrinkage and grit parameters. These
#' computed quantities can then be displayed as shown in the examples below.
#'
#' @param anchor Anchor partition in the Shrinkage Partition (SP) distribution
#'   as a numeric vector of cluster labels.
#' @param shrinkage_shape Shape parameter of the gamma prior distribution for
#'   the shrinkage parameter of the SP distribution.
#' @param shrinkage_rate Rate parameter of the gamma prior distribution for the
#'   shrinkage parameter of the SP distribution.
#' @param grit_shape1 First shape parameter of the beta prior distribution for
#'   the grit parameter of the SP distribution.
#' @param grit_shape2 Second shape parameter of the beta prior distribution for
#'   the grit parameter of the SP distribution.
#' @param use_crp Use the Chinese restaurant process (CRP) which serves as the
#'   baseline distribution?  If `FALSE`, the Jensen Liu partition (JLP) is
#'   used instead.
#' @param concentration Concentration parameter of the baseline distribution
#'   for the SP distribution.
#' @param shrinkage_n Length of the evenly-spaced grid for the shrinkage
#'   parameter of the SP distribution.
#' @param grit_n Length of the evenly-spaced grid for the grit parameter of the
#'   SP distribution.
#' @param n_mc_samples Number of Monte Carlo samples used to compute the
#'   expectation of the Rand index and entropy.
#' @param domain_specification List to control the domain for the computations,
#'   containing either: 1. Elements named `shrinkage_lim` and `grit_lim` (each
#'   vectors of length two indicating the lower and upper bound), or 2. Elements
#'   named `n_mc_samples` (giving the number of samples to draw to learn the
#'   domain) and `percentile` (indicating which percentile should be used to
#'   determine the upper bound).
#' @param n_cores Number of CPU cores to use, where 0 (default) indicates all
#'   cores.
#'
#' @returns A list containing five elements. Vectors `shrinkage` and `grit` give
#'   the locations of grid values.  Matrix `log_density` gives the log of the
#'   prior density at the combinations of shrinakge and grit values.  Likewise,
#'   matrices `expected_rand_index` and `expected_entropy` give expectations at
#'   the combinations of shrinkage and grit values.  See the examples below to
#'   see how to use this output.
#'
#' @export
#' @examples
#' anchor <- rep(1:4, each = 13)
#' out <- summarize_prior_on_shrinkage_and_grit(anchor, n_mc_samples = 100, n_cores = 1)
#' if (requireNamespace("fields") ) {
#'   fields::image.plot(out$shrinkage, out$grit, out$expected_rand_index,
#'                      xlab = "Shrinkage", ylab = "Grit")
#'   contour(out$shrinkage, out$grit, exp(out$log_density), add = TRUE, labcex = 1.0)
#' }
#' image(out$shrinkage, out$grit, out$expected_entropy, xlab = "Shrinkage", ylab = "Grit")
#' contour(out$shrinkage, out$grit, exp(out$log_density), add = TRUE, labcex = 1.0)
#' 
summarize_prior_on_shrinkage_and_grit <- function(
    anchor, shrinkage_shape = 4, shrinkage_rate = 1, grit_shape1 = 2, grit_shape2 = 2,
    use_crp = TRUE, concentration = 1.0, shrinkage_n = 25, grit_n = 25, n_mc_samples = 100,
    domain_specification = list(n_mc_samples = 1000, percentile = 0.95), n_cores = 0) {
  .Call(.summarize_prior_on_shrinkage_and_grit,
    anchor, shrinkage_shape, shrinkage_rate, grit_shape1, grit_shape2, use_crp, concentration,
    shrinkage_n, grit_n, n_mc_samples, domain_specification, n_cores)
}
