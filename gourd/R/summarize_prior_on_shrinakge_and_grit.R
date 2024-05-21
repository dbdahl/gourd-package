#' @export
summarize_prior_on_shrinkage_and_grit <- function(anchor, domain_specification = list(n = 1000, percentile = 0.95), shrinkage_shape = 5, shrinkage_rate = 1, shrinkage_n = 25, grit_shape1 = 1, grit_shape2 = 9, grit_n = 25, concentration = 1.0, n_mc_samples = 1000) {
  .Call(.summarize_prior_on_shrinkage_and_grit,
    anchor, domain_specification, shrinkage_shape, shrinkage_rate, shrinkage_n, grit_shape1, grit_shape2, grit_n, concentration, n_mc_samples)
}
