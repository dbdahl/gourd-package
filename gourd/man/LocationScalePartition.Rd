% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/LocationScalePartition.R
\name{LocationScalePartition}
\alias{LocationScalePartition}
\title{Probabilities for the Location Scale Partition Distribution}
\usage{
LocationScalePartition(anchor, shrinkage, concentration, permutation)
}
\arguments{
\item{anchor}{An integer vector giving the anchor partition
(a.k.a., center partition, location partition).}

\item{shrinkage}{A scalar giving the value of the shrinkage parameter, i.e.,the
reciprocal of the scale parameter.}

\item{concentration}{The concentration parameter, as a numeric value
greater than \eqn{-1*discount}.}

\item{permutation}{A vector containing the integers \eqn{1, 2, \ldots, n}
giving the order in which items are allocated to the partition.}
}
\value{
An object of class \code{PartitionDistribution} representing this
partition distribution.
}
\description{
This function specifies the location scale partition distribution for given
anchor partition, shrinkage, and permutation parameters.  The shrinkage parameter
is the reciprocal of the scale parameter.
}
\examples{
scale <- 0.1
LocationScalePartition(anchor=c(1,1,1,2,2), shrinkage=1/scale,
                       concentration=0.6, permutation=c(1,5,4,2,3))
}
