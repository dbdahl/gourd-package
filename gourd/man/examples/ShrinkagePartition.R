ShrinkagePartition(anchor=c(1,1,1,2,2), shrinkage=c(10,10,10,3,3),
                      permutation=c(1,5,4,2,3), grit=0.1,
                      baseline=CRPPartition(nItems=5, concentration=1.5, discount=0.1))

ShrinkagePartition(anchor=c(1,1,1,2,2), shrinkage=c(1,1,1,3,3),
                      permutation=c(1,5,4,2,3), grit=0.2,
                      baseline=UniformPartition(nItems=5))


ShrinkagePartition(anchor=c(1,1,1,2,2), shrinkage=c(0,0,0,3,3),
                      permutation=c(1,5,4,2,3), grit=0.2,
                      baseline=JensenLiuPartition(concentration=0.5, permutation=c(1,5,4,2,3)))
