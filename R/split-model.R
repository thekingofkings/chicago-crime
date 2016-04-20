# This model train two separate models, which is not fair to compare



# split the Chicago into north and south and train two models


library(maptools)
library(sp)
library(doParallel)
cl <- makeCluster(4)
registerDoParallel(cl)

source("NBUtilsSplit.R")




z <- file("split-north-south-chicago.out", open="wa")
sink(z, append=TRUE, type="output", split=FALSE)


ca <- readShapeSpatial("../data/ChiCA_gps/ChiCaGPS")


# 41.861096  -- South of this is a better model
DIVISION <- 41.921096

caN <- ca[coordinates(ca)[,2] > DIVISION, drop=FALSE]
caS <- ca[coordinates(ca)[,2] <= DIVISION, drop=FALSE]


idN <- sort(as.numeric(levels(ca$AREA_NUMBE)[caN$AREA_NUMBE]))
idS <- sort(as.numeric(levels(ca$AREA_NUMBE)[caS$AREA_NUMBE]))



demos <- read.table('pvalue-demo.csv', header=TRUE, sep=",")
focusColumn <- names(demos) %in% c("total.population", "population.density",
                                   "disadvantage.index", "residential.stability",
                                   "ethnic.diversity")
demos.part <- demos[focusColumn]
cat("Selected Demographics features:\n", names(demos.part), "\n")


w2 <- as.matrix(read.csv('pvalue-sociallag.csv', header=FALSE))
rownames(w2) <- as.character(1:77)


# crime
Y <- read.csv('pvalue-crime.csv', header=FALSE)
Y <- Y$V1


# directly predict crime rate
Y <- Y / demos$total.population * 10000
# use exposure to indirectly predict rate
# demos.part$total.population = log(demos.part$total.population)


cat("====================== split data and split model ==================\n")

sn <- "bysource"
exp <- "noexposure"
uselag <- "1100"



# north Chicago
mae.org <- leaveOneOut(demos.part[idN,], caN, w2[idN, idN], Y[idN], coeff=TRUE, normalize=TRUE, socialnorm=sn, exposure=exp, lagstr=uselag)
cat("North:", mean(mae.org), "\n")


# south Chicago
mae.org2 <- leaveOneOut(demos.part[idS,], caS, w2[idS, idS], Y[idS], coeff=TRUE, normalize=TRUE, socialnorm=sn, exposure=exp, lagstr=uselag)
cat("South:", mean(mae.org2), "\n")


cat("Overall:", mean( c(mae.org, mae.org2) ), "\n")



# One model
mae.org3 <- leaveOneOut(demos.part, ca, w2, Y, coeff=TRUE, normalize=TRUE, socialnorm=sn, exposure=exp, lagstr=uselag)
cat("One model\nNorth:", mean(mae.org3[idN]), "\nSouth:", mean(mae.org3[idS]), "\nOverall:", mean(mae.org3), "\n")



plot(mae.org3)
points(1:length(mae.org), mae.org, pch=2, col='red')
points(  (length(mae.org)+1) :length(mae.org3), mae.org2, pch=2, col='blue')







cat("\n====================== NOT split data but split model ==================\n")

mae.org4 <- leaveOneOut.split(demos.part, ca, w2, Y, idN, idS, socialnorm=sn, exposure=exp)

cat("North:", mean(mae.org4[idN]), "\nSouth:", mean(mae.org4[idS]), "\n")
cat("Overall:", mean(mae.org4), "\n")


sink()


stopCluster(cl)
