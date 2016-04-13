library(MASS)
library(maptools)
library(sp)
library(spdep)


source("NBUtils.R")

args <- commandArgs(trailingOnly = TRUE)
z = file(paste("glmmadmb-", args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], ".out", sep="-"), open="wa")





# generate the contiguous spatial weight
ca = readShapeSpatial("../data/ChiCA_gps/ChiCaGPS")
w1 <- spatialWeight(ca)

demos <- read.table('pvalue-demo.csv', header=TRUE, sep=",")
focusColumn <- names(demos) %in% c("total.population", "population.density",
                                   "poverty.index", "residential.stability",
                                   "ethnic.diversity")
demos.part <- demos[focusColumn]
cat("Selected Demographics features:\n", names(demos.part), "\n")


# spatial matrix
# w1 <- as.matrix(read.csv('pvalue-spatiallag.csv', header=FALSE))


# social matrix
# The entry (i,j) means the flow from j entering i.
# row i means, the flow from other CAs entering CA_i
w2 <- as.matrix(read.csv('pvalue-sociallag.csv', header=FALSE))

# crime
Y <- read.csv('pvalue-crime.csv', header=FALSE)
Y <- Y$V1

# use crime rate instead of crime count
# Y <- Y / demos$total.population * 10000


if (args[5] == "logpop") {
    demos.part$total.population = log(demos.part$total.population)    
}


if (args[9] == "logpopdensty" ){
    demos.part$population.density = log(demos.part$population.density)
}


SOCIALLAG <- if (args[6] == "useLEHD") TRUE else FALSE
SPATIALLAG <- if (args[7] == "useGeo") TRUE else FALSE



normalize <- TRUE
sn <- args[3]

cat(args, "\n")
sink(z, append=TRUE, type="output", split=FALSE)
mae.org <- leaveOneOut(demos.part, ca, w2, Y, coeff=TRUE, normalize=normalize, socialnorm=sn, exposure=args[4], SOCIALLAG=SOCIALLAG, SPATIALLAG=SPATIALLAG)
cat(mae.org, "\n")
itersN <- strtoi(args[8])


pvalues <- c()


# permute demographics
for (i in 1:ncol(demos.part)) {
    
    featureName <- colnames(demos.part)[i]
    cat(featureName, ' ')
    cnt <- 0
    
    for (j in 1:itersN) {
        demos.copy <- demos.part
        # permute features
        demos.copy[,i] <- sample( demos.part[,i] )
        mae <- leaveOneOut(demos.copy, ca, w2, Y, normalize=normalize, socialnorm=sn, exposure=args[4], SOCIALLAG=SOCIALLAG, SPATIALLAG=SPATIALLAG)
        if (j %% (itersN %/% 5)  == 0) {
            cat("-->", mae, "\n")
        }
        if (mae.org > mae) {
            cnt = cnt + 1
        }
    }
    pvalues$featureName <- cnt/itersN
}



if (SOCIALLAG || SPATIALLAG) {
    # permute lag
    cnt.social = 0
    cnt.spatial = 0
	for (j in 1:itersN) {
		mae = leaveOneOut.PermuteLag(demos.part, ca, w2, Y, normalize, socialnorm=sn, exposure=args[4], SOCIALLAG=SOCIALLAG, SPATIALLAG=SPATIALLAG)
		if (j %% (itersN %/% 5) == 0) {
			cat("-->", mae, "\n")
		}
		if (SOCIALLAG && mae.org > mae[1]) { # first one is social lag
			cnt.social = cnt.social + 1
		}
		if (!SOCIALLAG) {
			if (SPATIALLAG && mae.org > mae[1]) {
				cnt.spatial = cnt.spatial + 1
			}
		} else {
			if (SPATIALLAG && mae.org > mae[2]) {
				cnt.spatial = cnt.spatial + 1
			}
		}
	}

	if (SOCIALLAG) {
		pvalues <- c(pvalues, social.lag=cnt.social / itersN)
		cat("social.lag ", cnt.social / itersN, "\n")
	}
	

	if (SPATIALLAG) {
		pvalues <- c(pvalues, spatial.lag=cnt.spatial / itersN)
		cat("spatial.lag", cnt.spatial / itersN, "\n")
	}
}

cat(names(unlist(pvalues)), "\n")
cat(unlist(pvalues), "\n")


sink()

