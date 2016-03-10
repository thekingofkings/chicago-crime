library(MASS)
library(maptools)
library(sp)
library(spdep)



spatialWeight <- function( ca, leaveOneOut = -1 ) {
    if (leaveOneOut > -1) {
        ca <- ca[ ! ca$AREA_NUMBE == as.character(leaveOneOut), ]
    }

    ids <- as.numeric(as.vector(ca$AREA_NUMBE))
    w1 <- nb2mat(poly2nb(ca, row.names=ids), zero.policy=TRUE)
    ids.name <- as.character(ids)
    rownames(w1) <- ids.name
    colnames(w1) <- ids.name
    w1[ids.name,ids.name] <- w1
    return(w1)
}




leaveOneOut <- function(demos, ca, w2, Y, coeff=FALSE, normalize=FALSE, socialnorm="bydestination") {
    N <- length(Y)
    # leave one out evaluation
    errors = c()
    if (coeff) {
        coeffs = vector("double", ncol(demos) + 3)
    }

    w1 <- spatialWeight(ca)
    for ( i in 1:N ) {
        F <- demos[-i, , drop=FALSE]
        spt <- spatialWeight(ca, i)
        sco <- w2[-i, -i]

        if (socialnorm == "bysource") {
            cs <- colSums(sco)
            sco <- sweep(sco, 2, cs, "/")
        } else if (socialnorm == "bydestination") {
            rs <- rowSums(sco)
            sco <- sweep(sco, 1, rs, "/")
        } else if (socialnorm == 'bypair') {
            sco <- sco + t(sco)
            s <- sum(sco)
            sco <- sco / s
            stopifnot( sco[4, 4] == 0, nrow(sco)==N-1, ncol(sco)==N-1, sum(sco) == 1)
        }
            
        y <- Y[-i]

        F[,'spatial.lag'] = as.vector(spt %*% y)
        F[,'social.lag'] = as.vector(sco %*% y)

        # normalize features
		if (normalize) {
			F <- scale(F, center=TRUE, scale=TRUE)
			F.center <- as.vector(attributes(F)["scaled:center"][[1]])
			F.scale <- as.vector(attributes(F)["scaled:scale"][[1]])
		}
        # fit NB model
        dat <- data.frame(y, F)
        mod <- glm.nb(data=dat)
        if (coeff) {
            coeffs <- coeffs + as.vector(mod$coefficients)
        }

        # predict at point i
        spatial.lag <- w1[i,-i] %*% y
        if (socialnorm == "bysource") {
            social.lag <- (w2[i,-i] / cs) %*% y
        } else if (socialnorm == "bydestination") {
            social.lag <- w2[i,-i]  %*% y / sum(w2[i,-i])
        } else if (socialnorm == "bypair") {
            social.lag <- (w2[i,-i] / s) %*% y
        } else {
            social.lag <- w2[i,-i] %*% y
        }
        stopifnot( length(social.lag) == 1)
        
        dn <- data.frame(demos[i, , drop=FALSE], spatial.lag, social.lag)

	if (normalize) {
            dn <- data.frame(scale(dn, center=F.center, scale=F.scale))
	}
        ybar <- predict(mod, newdata=dn, type=c('response'))
        errors <- c(errors, abs(ybar - Y[i]))
    }

    if (coeff) {
        cat("names ", names(mod$coefficients), "\n")
        cat(coeffs / N, "\n")
    }

    return(mean(errors))
}



leaveOneOut.PermuteLag <- function(demos, ca, w2, Y, normalize=FALSE, socialnorm="bydestination") {
    N <- length(Y)
    # permute lag matix is equivalent to permute Y
    y = sample(Y)
    mae = c()
    w1 <- spatialWeight(ca)
    # control which lags to use
    for (lag in c("social", "spatial")) {
        # leave one out evaluation
        errors = c()
        for ( i in 1:N ) {
            F <- demos[-i, , drop=FALSE]
            spt <- spatialWeight(ca, i)
            sco <- w2[-i, -i]

            if (socialnorm == "bysource") {
                cs <- colSums(sco)
                sco <- sweep(sco, 2, cs, "/")
            } else if (socialnorm == "bydestination") {
                rs <- rowSums(sco)
                sco <- sweep(sco, 1, rs, "/")
            } else if (socialnorm == 'bypair') {
                sco <- sco + t(sco)
                s <- sum(sco)
                sco <- sco / s
                stopifnot( sco[4, 4] == 0, nrow(sco)==N-1, ncol(sco)==N-1, sum(sco)== 1)
            }
            

            if (lag == "social") {
                F[,'spatial.lag'] = as.vector(spt %*% Y[-i])
                F[,'social.lag'] = as.vector(sco %*% y[-i])
            } else {
                F[,'spatial.lag'] = as.vector(spt %*% y[-i])
                F[,'social.lag'] = as.vector(sco %*% Y[-i])
            }

            # normalize features
			if (normalize) {
				F <- scale(F, center=TRUE, scale=TRUE)
				F.center <- as.vector(attributes(F)["scaled:center"][[1]])
				F.scale <- as.vector(attributes(F)["scaled:scale"][[1]])
			}
            # fit NB model
            dat <- data.frame(Y[-i], F)
            mod <- glm.nb(data=dat)

            # predict at point i
            spatial.lag <- w1[i,-i] %*% y[-i]
            if (socialnorm == "bysource") {
                social.lag <- (w2[i,-i] / cs) %*% y[-i]
            } else if (socialnorm == "bydestination") {
                social.lag <- w2[i,-i]  %*% y[-i] / sum(w2[i,-i])
            } else if (socialnorm == "bypair") {
                social.lag <- (w2[i,-i] / s) %*% y[-1]
            } else {
                social.lag <- w2[i,-i] %*% y[-i]
            }
            stopifnot( length(social.lag) == 1)
            dn <- data.frame(demos[i, , drop=FALSE], spatial.lag, social.lag)
			if (normalize) {
				dn <- data.frame(scale(dn, center=F.center, scale=F.scale))
			}
            ybar <- predict(mod, newdata=dn, type=c('response'))
            errors <- c(errors, abs(ybar - Y[i]))
        }
        mae = c(mae, mean(errors))
    }

    return(mae)
}   




# generate the contiguous spatial weight
ca = readShapeSpatial("../data/ChiCA_gps/ChiCaGPS")
w1 <- spatialWeight(ca)

demos <- read.table('pvalue-demo.csv', header=TRUE, sep=",")
focusColumn <- names(demos) %in% c("total.population", "poverty.index", "residential.stability")
demos.part <- demos[focusColumn]
cat(names(demos.part), "\n")


# spatial matrix
# w1 <- as.matrix(read.csv('pvalue-spatiallag.csv', header=FALSE))


# social matrix
# The entry (i,j) means the flow from j entering i.
# row i means, the flow from other CAs entering CA_i
w2 <- as.matrix(read.csv('pvalue-sociallag.csv', header=FALSE))
w2 <- t(w2)
# crime
Y <- read.csv('pvalue-crime.csv', header=FALSE)
Y <- Y$V1
#Y <- Y / demos$total.population * 10000

demos.part$total.population = log(demos.part$total.population)

normalize <- TRUE
sn <- "bypair"

mae.org <- leaveOneOut(demos.part, ca, w2, Y, coeff=TRUE, normalize=normalize, socialnorm=sn)
cat(mae.org, "\n")
itersN <- 200

if (FALSE) {
# permute demographics
for (i in 1:ncol(demos.part)) {
    
    cat(colnames(demos.part)[i], ' ')
    cnt <- 0
    
    for (j in 1:itersN) {
        demos.copy <- demos.part
        # permute features
        demos.copy[,i] <- sample( demos.part[,i] )
        mae <- leaveOneOut(demos.copy, ca, w2, Y, normalize=normalize, socialnorm=sn)
		if (j %% 100 == 0) {
			cat("-->", mae, "\n")
		}
        if (mae.org > mae) {
            cnt = cnt + 1
        }
    }
    cat(cnt / itersN, '\n')
}
}

# permute lag
cnt.social = 0
cnt.spatial = 0
for (j in 1:itersN) {
    mae = leaveOneOut.PermuteLag(demos.part, ca, w2, Y, normalize, socialnorm=sn)
	if (j %% 100 == 0) {
		cat("-->", mae, "\n")
	}
    if (mae.org > mae[1]) { # first one is social lag
        cnt.social = cnt.social + 1
    }
    if (mae.org > mae[2]) {
        cnt.spatial = cnt.spatial + 1
    }
}

cat("social.lag ", cnt.social / itersN, "\nspatial.lag", cnt.spatial / itersN, "\n")
