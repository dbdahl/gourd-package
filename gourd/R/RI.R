#' @export
RI <- function(truth, estimate, a = 1.0) {
  .Call(.rand_index_for_r, truth, estimate, a)
}
