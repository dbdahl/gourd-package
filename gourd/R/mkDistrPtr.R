mkDistrPtr <- function(distr, excluded=NULL, included=NULL) {
  stp <- function() stop(sprintf("'%s' is not supported.", paste(class(distr),collapse=",")))
  if ( ( ! is.null(excluded) ) && ( inherits(distr,excluded) ) ) stp()
  if ( ( ! is.null(included) ) && ( ! inherits(distr,included) ) ) stp()
  if ( inherits(distr,"FixedPartition") ) {
    .Call(.new_FixedPartitionParameters, distr$anchor)
  } else if ( inherits(distr,"CRPPartition") ) {
    .Call(.new_CrpParameters, distr$nItems, distr$mass, distr$discount)
  } else if ( inherits(distr, "FocalPartition") ) {
    .Call(.new_FrpParameters, distr$anchor, distr$shrinkage, distr$.permutation, distr$mass, distr$discount, distr$power)
  } else if ( inherits(distr, "LocationScalePartition") ) {
    .Call(.new_LspParameters, distr$anchor, distr$shrinkage, distr$mass, distr$.permutation)
  } else if ( inherits(distr, "CenteredPartition") ) {
    p <- mkDistrPtr(distr$baseline, included=supportedBaselines)
    .Call(.new_CppParameters, distr$anchor, distr$shrinkage, p, distr$useVI, distr$a)
  } else if ( inherits(distr, "EPAPartition") ) {
    .Call(.new_EpaParameters, distr$similarity, distr$.permutation, distr$mass, distr$discount)
  } else if ( inherits(distr, "OldShrinkagePartition") ) {
    p <- mkDistrPtr(distr$baseline, included=supportedBaselines)
    .Call(.new_OldSpParameters, distr$anchor, distr$shrinkage, distr$.permutation, p, distr$useVI, distr$a, distr$scalingExponent)
  } else if ( inherits(distr, "Old2ShrinkagePartition") ) {
    p <- mkDistrPtr(distr$baseline, included=supportedBaselines)
    .Call(.new_Old2SpParameters, distr$anchor, distr$shrinkage, distr$.permutation, p)
  } else if ( inherits(distr, "ShrinkagePartition") ) {
    p <- mkDistrPtr(distr$baseline, included=supportedBaselines)
    .Call(.new_SpParameters, distr$anchor, distr$shrinkage, distr$.permutation, p)
  } else if ( inherits(distr, "ShrinkageMixturePartition") ) {
    p <- mkDistrPtr(distr$baseline, included=supportedBaselines)
    .Call(.new_SpMixtureParameters, distr$anchor, distr$shrinkage, distr$.permutation, p)
  } else if ( inherits(distr,"UniformPartition") ) {
    .Call(.new_UpParameters, distr$nItems)
  } else if ( inherits(distr,"JensenLiuPartition") ) {
    .Call(.new_JlpParameters, distr$mass, distr$.permutation)
  } else stp()
}

supportedBaselines <- c("UniformPartition","JensenLiuPartition","CRPPartition")

