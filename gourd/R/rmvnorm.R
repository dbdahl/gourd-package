#' @export
rmvnorm <- function(n, mean, precision) {
  .Call(.sample_multivariate_normal, n, mean, precision)
}
