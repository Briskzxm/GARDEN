library(MERINGUE)

show_network <- function(cct, nct, weight, pos=NULL, plot=FALSE, ...) {
  weightIc <- weight[c(cct, nct), c(cct, nct)]
  weightIc[nct,nct] <- 0
  weightIc[cct,cct] <- 0
  if(plot) {
    plotNetwork(pos, weightIc, line.col	='black' ,line.power	
=2,...)
    points(pos[nct,], col='#f2656a', pch=16)
    points(pos[cct,], col='#71b499', pch=16)
  }
  return(weightIc)
}

posw <- read.csv('spatial.csv',sep=' ')
rownames(posw) <- paste0('cell', 1:dim(posw)[1])
weight <- getSpatialNeighbors(posw, filterDist = 1000)
plotNetwork(posw, weight,col='red')
domain <- read.csv('domain.csv')
domain <- data.frame(my_column = domain)
rownames(domain)<- paste0('cell', 1:dim(posw)[1])

ctA <- rownames(domain)[domain$celltype == 12]
ctB <- rownames(domain)[domain$celltype == 2]
print(intersect(ctA, ctB))
weightIc <- getInterCellTypeWeight(ctA, ctB, 
                                   weight, posw, 
                                   plot=TRUE, 
                                   main='Adjacency Weight Matrix\nBetween Cell-Types')
par(mfrow=c(1,1), mar=rep(5,4))
pdf(file = 'net.pdf')
weightIc <- show_network(ctA, ctB, 
                                   weight, posw, 
                                   plot=TRUE
                                   )
dev.off()