getOr <- function(x, or) {
  if ( is.null(x) ) or else x
}

coerceLogical <- function(x, name) {
  if ( ! is.logical(x) ) stop(sprintf("'%s' must be logical.",name))
  if ( storage.mode(x) != "logical" ) storage.mode(x) <- "logical"
  if ( length(x) != 1 ) stop(sprintf("'%s' must be a scalar value.",name))
  x
}

coerceInteger <- function(x, name) {
  if ( ! is.numeric(x) ) stop(sprintf("'%s' must be numeric.",name))
  if ( storage.mode(x) != "integer" ) storage.mode(x) <- "integer"
  if ( length(x) != 1 ) stop(sprintf("'%s' must be a scalar value.",name))
  x
}

coerceIntegerVector <- function(x, name) {
  if ( ! is.numeric(x) ) stop(sprintf("'%s' must be numeric.",name))
  if ( storage.mode(x) != "integer" ) storage.mode(x) <- "integer"
  x
}

coerceDouble <- function(x, name) {
  if ( ! is.numeric(x) ) stop(sprintf("'%s' must be numeric.",name))
  if ( storage.mode(x) != "double" ) storage.mode(x) <- "double"
  if ( length(x) != 1 ) stop(sprintf("'%s' must be a scalar value.",name))
  x
}

coerceSimilarity <- function(x) {
  if ( ! is.numeric(x) ) stop("'similarity' must be numeric.")
  if ( storage.mode(x) != "double" ) storage.mode(x) <- "double"
  if ( ! is.matrix(x) ) stop("'similarity' must be a matrix.")
  if ( ! isSymmetric(x) ) stop("'similarity' must be a symmetric matrix.")
  if ( any( x <= 0.0 ) ) stop("'similarity' must be have strictly positive enteries.")
  if ( nrow(x) < 1 ) stop("The number of rows in 'similarity' must be at least one.")
  x
}

coerceNItems <- function(x) {
  x <- coerceInteger(x)
  if ( x < 1 ) stop("'nItems' should be at least one.")
  x
}

coerceAnchor <- function(x) {
  x <- coerceIntegerVector(x, "anchor")
  if ( length(x) < 1 ) stop("The number of items in 'anchor' must be at least one.")
  x
}

coercePermutation <- function(x, nItems) {
  if ( nItems < 1 ) stop("Number of items must be at least one.")
  if ( ! is.numeric(x) ) stop("'permutation' must be numeric.")
  if ( storage.mode(x) != "integer" ) storage.mode(x) <- "integer"
  if ( ( length(x) != nItems ) || ( min(x) < 1L ) || ( max(x) > nItems ) || ( length(unique(x)) != nItems ) ) stop("'permutation' is not a valid permutation.")
  x
}

coerceShrinkage <- function(x) {
  if ( ! is.numeric(x) ) stop("'shrinkage' must be numeric.")
  if ( storage.mode(x) != "double" ) storage.mode(x) <- "double"
  if ( length(x) != 1 ) stop("'shrinkage' must be a scalar.")
  if ( x < 0.0 ) stop("'shrinkage' must be nonnegative.")
  x
}

coerceShrinkageVector <- function(x, nItems) {
  if ( nItems < 1 ) stop("Number of items must be at least one.")
  if ( ! is.numeric(x) ) stop("'shrinkage' must be numeric.")
  if ( storage.mode(x) != "double" ) storage.mode(x) <- "double"
  if ( any(x < 0.0) ) stop("'shrinkage' must be nonnegative.")
  if ( length(x) == 1 ) x <- rep(x, nItems)
  else if ( length(x) != nItems ) stop("Incorrect length for 'shrinkage'.")
  x
}

coerceShrinkageProbabilityVector <- function(x, nItems) {
  if ( nItems < 1 ) stop("Number of items must be at least one.")
  if ( ! is.numeric(x) ) stop("'shrinkage' must be numeric.")
  if ( storage.mode(x) != "double" ) storage.mode(x) <- "double"
  if ( any(x < 0.0 | x > 1.0) ) stop("'shrinkage' must be in [0,1].")
  if ( length(x) == 1 ) x <- rep(x, nItems)
  else if ( length(x) != nItems ) stop("Incorrect length for 'shrinkage'.")
  x
}

coerceConcentrationDiscount <- function(concentration, discount) {
  discount <- coerceDouble(discount, "discount")
  if ( ( discount < 0.0 ) || ( discount >= 1.0 ) ) stop("'discount' must be in [0,1).")
  concentration <- coerceDouble(concentration, "concentration")
  if ( concentration <= -discount ) stop("'concentration' must be greater than -'discount'.")
  list(concentration=concentration, discount=discount)
}

supportedBaselines <- c("UniformPartition","JensenLiuPartition","CRPPartition")

checkBaseline <- function(baseline, nItems) {
  if ( ! inherits(baseline, "PartitionDistribution") ) stop("'baseline' must be a valid partition distribution.")
  if ( baseline$nItems != nItems ) stop("'anchor' implies a different number of items than the 'baseline'.")
  if ( ! inherits(baseline,supportedBaselines) ) stop("Unsupported 'baseline'.")
}

checkGrit <- function(x) {
  if ( ! is.numeric(x) ) stop("'grit' must be numeric.")
  if ( storage.mode(x) != "double" ) storage.mode(x) <- "double"
  if ( length(x) != 1 ) stop("'grit' must be a scalar.")
  # if ( x <= 0.0 || x >= 1.0 ) stop("'grit' must be in (0,1).")
}

isCanonical <- function(labels) isCanonicalFromUnique(unique(labels))

isCanonicalFromUnique <- function(u) {
  if ( min(u) != 1L ) return(FALSE)
  if ( max(u) != length(u) ) return(FALSE)
  all(diff(u) > 0)
}

canonicalForm <- function(labels) {
  temp <- integer(length(labels))
  i <- 1
  for (s in unique(labels)) {
    temp[which(labels == s)] <- i
    i <- i + 1
  }
  temp
}

distrEnv <- function(class, list, lock=TRUE) {
  result <- list2env(list)
  result$.names <- names(list)
  class(result) <- c(class, "PartitionDistribution")
  if ( lock ) lockEnvironment(result, TRUE)
  result
}

distrAsList <- function(env) {
  mget(env$.names[!grepl("^\\.",env$.names)], envir=env)
}

distrClone <- function(env) {
  distrEnv(class(env)[1], mget(ls(envir=env, all.names=TRUE), envir=env), FALSE)
}

distrLock <- function(env) {
  lockEnvironment(env, TRUE)
}

distrUnlock <- function(env) {
  for ( s in ls(envir=env, all.names=TRUE) ) do.call("unlockBinding", list(s, env))
  env
}

