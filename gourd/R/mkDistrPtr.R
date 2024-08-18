#' Make a Pointer to Partitional Distribution Parameters
#'
#' Users should not call this function.  This is an internal function exported
#' for the sake of developers of packages depending on this package.
#'
#' @param distr An object of class \sQuote{PartitionDistribution}.
#' @param excluded A character vector of explicitly excluded partition distributions.
#' @param included A character vector of explicitly included partition distributions.
#'
#' @return A pointer
#' @export
#'
mkDistrPtr <- function(distr, excluded=NULL, included=NULL) {
  stp <- function() stop(sprintf("'%s' is not supported.", paste(class(distr),collapse=",")))
  if ( ( ! is.null(excluded) ) && ( inherits(distr,excluded) ) ) stp()
  if ( ( ! is.null(included) ) && ( ! inherits(distr,included) ) ) stp()
  if ( inherits(distr,"FixedPartition") ) {
    .Call(.new_FixedPartitionParameters, distr$anchor)
  } else if ( inherits(distr,"CRPPartition") ) {
    .Call(.new_CrpParameters, distr$nItems, distr$concentration, distr$discount)
  } else if ( inherits(distr, "LocationScalePartition") ) {
    .Call(.new_LspParameters, distr$anchor, distr$shrinkage, distr$concentration, distr$.permutation)
  } else if ( inherits(distr, "CenteredPartition") ) {
    p <- mkDistrPtr(distr$baseline, included=supportedBaselines)
    .Call(.new_CppParameters, distr$anchor, distr$shrinkage, p, distr$useVI, distr$a)
  } else if ( inherits(distr, "ShrinkagePartition") ) {
    p <- mkDistrPtr(distr$baseline, included=supportedBaselines)
    .Call(.new_SpParameters, distr$anchor, distr$shrinkage, distr$.permutation, distr$grit, p, distr$optimized)
  } else if ( inherits(distr,"UniformPartition") ) {
    .Call(.new_UpParameters, distr$nItems)
  } else if ( inherits(distr,"JensenLiuPartition") ) {
    .Call(.new_JlpParameters, distr$concentration, distr$.permutation)
  } else stp()
}
