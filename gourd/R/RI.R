#' @export
RI <- function(x, y) {
  .Call(.rand_index, x, y)
}
