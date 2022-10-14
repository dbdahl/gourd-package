mkDistrPtr <- function(distr, excluded=NULL, included=NULL) {
  stp <- function() stop(sprintf("'%s' is not supported.", paste(class(distr),collapse=",")))
  if ( ( ! is.null(excluded) ) && ( inherits(distr,excluded) ) ) stp()
  if ( ( ! is.null(included) ) && ( ! inherits(distr,included) ) ) stp()
  if ( inherits(distr,"FixedPartition") ) {
    .Call(.new_FixedPartitionParameters, distr$baselinePartition)
  } else if ( inherits(distr,"CRPPartition") ) {
    .Call(.new_CrpParameters, distr$nItems, distr$mass, distr$discount)
  } else if ( inherits(distr, "FocalPartition") ) {
    .Call(.new_FrpParameters, distr$baselinePartition, distr$shrinkage, distr$.permutation, distr$mass, distr$discount, distr$power)
  } else if ( inherits(distr, "LocationScalePartition") ) {
    .Call(.new_LspParameters, distr$baselinePartition, distr$shrinkage, distr$mass, distr$.permutation)
  } else if ( inherits(distr, "CenteredPartition") ) {
    p <- mkDistrPtr(distr$baselineDistribution, included=supportedBaselineDistributions)
    .Call(.new_CppParameters, distr$baselinePartition, distr$shrinkage, p, distr$useVI, distr$a)
  } else if ( inherits(distr, "EPAPartition") ) {
    .Call(.new_EpaParameters, distr$similarity, distr$.permutation, distr$mass, distr$discount)
  } else if ( inherits(distr, "OldShrinkagePartition") ) {
    p <- mkDistrPtr(distr$baselineDistribution, included=supportedBaselineDistributions)
    .Call(.new_OldSpParameters, distr$baselinePartition, distr$shrinkage, distr$.permutation, p, distr$useVI, distr$a, distr$scalingExponent)
  } else if ( inherits(distr, "ShrinkagePartition") ) {
    p <- mkDistrPtr(distr$baselineDistribution, included=supportedBaselineDistributions)
    .Call(.new_SpParameters, distr$baselinePartition, distr$shrinkage, distr$.permutation, p)
  } else if ( inherits(distr,"UniformPartition") ) {
    .Call(.new_UpParameters, distr$nItems)
  } else if ( inherits(distr,"JensenLiuPartition") ) {
    .Call(.new_JlpParameters, distr$mass, distr$.permutation)
  } else stp()
}

supportedBaselineDistributions <- c("UniformPartition","JensenLiuPartition","CRPPartition")
