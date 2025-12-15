library(anndata)
library(CellChat)
library(dplyr)

ad <- read_h5ad("cancer.h5ad")
counts <- t(as.matrix(ad$X))
library.size <- Matrix::colSums(counts)
data.input <- as(counts, "dgCMatrix")

meta <- ad$obs 
meta$labels <- meta[["domain"]]
cellchat <- createCellChat(object = data.input, meta = meta, group.by = "labels")

CellChatDB <- CellChatDB.human 
showDatabaseCategory(CellChatDB)

CellChatDB.use <- subsetDB(CellChatDB, search = "Secreted Signaling", key = "annotation") # use Secreted Signaling
cellchat@DB <- CellChatDB.use

cellchat_E1 <- subsetData(cellchat)
future::plan("multisession", workers = 5) 
cellchat_E1 <- identifyOverExpressedGenes(cellchat_E1)
cellchat_E1 <- identifyOverExpressedInteractions(cellchat_E1, variable.both = F)

cellchat_E1 <- computeCommunProb(cellchat_E1, type = "triMean", trim = 0.1,
                              distance.use = TRUE, interaction.range = 250, 
                              scale.distance = 0.01,
                              contact.dependent = TRUE, contact.range = 100)
                              
cellchat_E1 <- filterCommunication(cellchat_E1, min.cells = 10)
cellchat_E1 <- computeCommunProbPathway(cellchat_E1)
cellchat_E1 <- aggregateNet(cellchat_E1) 

mat <- cellchat_E1@net$weight
i <- 11
mat2 <- matrix(0, nrow = nrow(mat), ncol = ncol(mat), dimnames = dimnames(mat))
mat2[i, ] <- mat[i, ]
netVisual_circle(mat2, vertex.weight = rowSums(cellchat_E1@net$weight), color.use = color,weight.scale = T, edge.weight.max = max(mat), title.name = rownames(mat)[i])