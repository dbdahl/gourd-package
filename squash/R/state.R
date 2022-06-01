#' @export
state_r2rust <- function(state) {
  .Call(.state_r2rust, state)
}

#' @export
state_rust2r <- function(state) {
  .Call(.state_rust2r, state)
}
