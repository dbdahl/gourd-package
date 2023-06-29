#' @docType package
#' @usage NULL
#' @useDynLib gourd, .registration = TRUE
NULL

.Kall <- function(...) {
  x <- .Call(...)
  if (inherits(x, "error")) stop(x) else x
}
