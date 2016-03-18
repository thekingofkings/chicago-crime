library(MASS)
library(maptools)
library(sp)
library(spdep)
library(glmmADMB)

args <- commandArgs(trailingOnly = TRUE)
z = file(paste("glmmadmb-", args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], ".out", sep="-"), open="wa")



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




leaveOneOut <- function(demos, ca, w2, Y, coeff=FALSE, normalize=FALSE, socialnorm="bydestination", exposure="exposure", SOCIALLAG=TRUE, SPATIALLAG=TRUE) {
    N <- length(Y)
    # leave one out evaluation
    errors = c()
    if (coeff) {
        coeffs = vector("double")
    }

    w1 <- spatialWeight(ca)
    for ( i in 1:N ) {
        F <- demos[-i, , drop=FALSE]
        y <- Y[-i]
        test.dn <- demos[i, , drop=FALSE]
        
        if (SOCIALLAG) {
            # training set
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

            F[,'social.lag'] = as.vector(sco %*% y)

            # testing point i
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
            test.dn['social.lag'] = social.lag
        }

        if (SPATIALLAG) {
            # training set
            spt <- spatialWeight(ca, i)
            F[,'spatial.lag'] = as.vector(spt %*% y)
            # testing point i
            spatial.lag <- w1[i,-i] %*% y
            test.dn['spatial.lag'] <- spatial.lag
        }
        

        # normalize features
        if (normalize) {
            # normalize training set
            F <- scale(F, center=TRUE, scale=TRUE)
            F.center <- as.vector(attributes(F)["scaled:center"][[1]])
            F.scale <- as.vector(attributes(F)["scaled:scale"][[1]])
            test.dn <- data.frame(scale(test.dn, center=F.center, scale=F.scale))
        }

        
        # fit NB model
        dat <- data.frame(y, F)
        stopifnot( all(is.finite(as.matrix(dat))) )
        mod <- tryCatch( {
            if (exposure == "exposure") {
                mod <- glmmadmb(y ~ . - total.population + offset(total.population), data=dat, family="nbinom", verbose=FALSE)
            } else {
                mod <- glmmadmb(y ~ ., data=dat, family="nbinom", verbose=FALSE)
            }
            mod
        }, error=function(e) e)

        if (inherits(mod, "error")) {
            warning("error in glmmadbm fitting - skip this iteration\n")
            next
        }
        
        if (coeff) {
            coef_iter <- as.vector(mod$b)
            if (length(coeffs) == 0) {
                coeffs <- vector("double", length(coef_iter))
            }
            coeffs <- coeffs + coef_iter
            stopifnot( length(coeffs) == length(coef_iter) )
        }

        
        ybar <- predict(mod, newdata=test.dn, type=c('response'))
        errors <- c(errors, abs(ybar - Y[i]))
    }

    if (coeff) {
        cat("Coefficients\n", names(mod$b), "\n")
        cat(coeffs / N, "\n")
    }

    return(mean(errors))
}



leaveOneOut.PermuteLag <- function(demos, ca, w2, Y, normalize=FALSE, socialnorm="bydestination", exposure="exposure",  SOCIALLAG=TRUE, SPATIALLAG=TRUE) {
    N <- length(Y)
    # permute lag matix is equivalent to permute Y
    y = sample(Y)
    mae = c()
    w1 <- spatialWeight(ca)

    lags <- c()
    if (SOCIALLAG) {
        lags <- c(lags, "social")
    }
    if (SPATIALLAG) {
        lags <- c(lags, "spatial")
    }
    # control which lags to use
    for (lag in lags) {
        # leave one out evaluation
        errors = c()
        for ( i in 1:N ) {
            F <- demos[-i, , drop=FALSE]
            test.dn <- demos[i, , drop=FALSE]

            if (SPATIALLAG) {
                # training set
                spt <- spatialWeight(ca, i)
                if (lag == "social") {
                    F[,'spatial.lag'] = as.vector(spt %*% Y[-i])
                } else {
                    F[,'spatial.lag'] = as.vector(spt %*% y[-i])
                }

                # testing data at point i
                spatial.lag <- w1[i,-i] %*% y[-i]
                test.dn['spatial.lag'] <- spatial.lag
            }

            if (SOCIALLAG) {
                # training set
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
                    F[,'social.lag'] = as.vector(sco %*% y[-i])
                } else {
                    F[,'social.lag'] = as.vector(sco %*% Y[-i])
                }

                # testing data at point i
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
                test.dn['social.lag'] = social.lag
            }
            

            

            # normalize features
            if (normalize) {
                F <- scale(F, center=TRUE, scale=TRUE)
                F.center <- as.vector(attributes(F)["scaled:center"][[1]])
                F.scale <- as.vector(attributes(F)["scaled:scale"][[1]])
                test.dn <- data.frame(scale(test.dn, center=F.center, scale=F.scale))
            }
            
            # fit NB model
            dat <- data.frame(Y[-i], F)
            names(dat)[1] <- "y"

            mod <- tryCatch( {
                if (exposure == "exposure") {
                    mod <- glmmadmb(y ~ . - total.population + offset(total.population), data=dat, family="nbinom", verbose=FALSE)
                } else {
                    mod <- glmmadmb(y ~ ., data=dat, family="nbinom", verbose=FALSE)
                }
                mod
            }, error=function(e) e)

            if (inherits(mod, "error")) {
                 warning("error in glmmadbm fitting - skip this iteration\n")
                 next
             }

            
            ybar <- predict(mod, newdata=test.dn, type=c('response'))
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
w2 <- t(w2)
# crime
Y <- read.csv('pvalue-crime.csv', header=FALSE)
Y <- Y$V1

# use crime rate instead of crime count
# Y <- Y / demos$total.population * 10000


if (args[5] == "logpop") {
    demos.part$total.population = log(demos.part$total.population)    
}


SOCIALLAG <- if (args[6] == "useLEHD") TRUE else FALSE
SPATIALLAG <- if (args[7] == "useGeo") TRUE else FALSE



normalize <- TRUE
sn <- args[3]


sink(z, append=TRUE, type="output", split=FALSE)
cat(args, "\n")
mae.org <- leaveOneOut(demos.part, ca, w2, Y, coeff=TRUE, normalize=normalize, socialnorm=sn, exposure=args[4], SOCIALLAG=SOCIALLAG, SPATIALLAG=SPATIALLAG)
cat(mae.org, "\n")
itersN <- strtoi(args[8])


# permute demographics
for (i in 1:ncol(demos.part)) {
    
    cat(colnames(demos.part)[i], ' ')
    cnt <- 0
    
    for (j in 1:itersN) {
        demos.copy <- demos.part
        # permute features
        demos.copy[,i] <- sample( demos.part[,i] )
        mae <- leaveOneOut(demos.copy, ca, w2, Y, normalize=normalize, socialnorm=sn, exposure=args[4], SOCIALLAG=SOCIALLAG, SPATIALLAG=SPATIALLAG)
		if (j %% 100  == 0) {
			cat("-->", mae, "\n")
		}
        if (mae.org > mae) {
            cnt = cnt + 1
        }
    }
    cat(cnt / itersN, '\n')
}



if (SOCIALLAG || SPATIALLAG) {
    # permute lag
    cnt.social = 0
    cnt.spatial = 0
    for (j in 1:itersN) {
        mae = leaveOneOut.PermuteLag(demos.part, ca, w2, Y, normalize, socialnorm=sn, exposure=args[4], SOCIALLAG=SOCIALLAG, SPATIALLAG=SPATIALLAG)
	if (j %% 5 == 0) {
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

    if (SOCIALLAG) 
        cat("social.lag ", cnt.social / itersN, "\n")

    if (SPATIALLAG)
        cat("spatial.lag", cnt.spatial / itersN, "\n")
}
sink()

