concentration <- 1.0
discount <- 0.1
nSamples <- 3

distr <- CRPPartition(nItems=5, concentration=concentration, discount=discount)
x <- samplePartition(distr, nSamples, nCores=1)
prPartition(distr, x)

anchor <- c(1,1,1,2,2)
permutation <- c(1,5,4,2,3)
n_items <- length(permutation)

distr <- ShrinkagePartition(anchor=anchor, shrinkage=c(0,0,0,0.3,0.3),
             permutation=permutation, grit=0.2,
             CRPPartition(nItems=n_items, concentration=concentration, discount=discount))
x <- samplePartition(distr, nSamples, nCores=1)
prPartition(distr, x)
