
# Implement some utility function for NB regression with glmmADMB


library(glmmADMB)
library(spdep)
library(foreach)



# calculate contiguous spatial matrix
spatialWeight <- function( ca, leaveOneOut = -1 ) {
    if (leaveOneOut > -1) {
        ca <- ca[ ! ca$AREA_NUMBE == as.character(leaveOneOut), ]
    }

    ids <- as.numeric(as.vector(ca$AREA_NUMBE))
    w1 <- nb2mat(poly2nb(ca, row.names=ids), zero.policy=TRUE)

    ids.name <- as.character(ids)
    colnames(w1) <- ids.name

    order.ids.n <- as.character(sort(ids))
    w1.res <- w1[order.ids.n, order.ids.n]
    
    return(w1.res)
}



normalize.social.lag <- function( sco, socialnorm="bysource" ) {
    N <- nrow(sco)
    stopifnot( N == ncol(sco) )
    if (socialnorm == "bysource") {
        rs <- rowSums(sco)
        sco <- sweep(sco, 1, rs, "/")
        stopifnot( sco[4,4] == 0, nrow(sco)==N, ncol(sco)==N, abs(sum(sco[3,])-1) <= 0.0000001)
    } else if (socialnorm == "bydestination") {
        sco <- t(sco)
        rs <- rowSums(sco)
        sco <- sweep(sco, 1, rs, "/")
        stopifnot( sco[4,4] == 0, nrow(sco)==N, ncol(sco)==N, abs(1-sum(sco[3,])) <= 0.0000001)
    } else if (socialnorm == 'bypair') {
        sco <- sco + t(sco)
        s <- sum(sco)
        sco <- sco / s
        stopifnot( sco[4, 4] == 0, nrow(sco)==N, ncol(sco)==N, abs(sum(sco) - 1) <= 0.00000001)
    }

    return (sco)
}






learnNB <- function( dat, exposure ) {
    mod <- tryCatch( {
        if (exposure == "exposure") {
            mod <- glmmadmb(y ~ . - total.population + offset(total.population), data=dat, family="nbinom", verbose=FALSE)
        } else {
            mod <- glmmadmb(y ~ ., data=dat, family="nbinom", verbose=FALSE)
        }
        mod
    }, error=function(e) e)
    return (mod)
}



# one run of the leave-one-out evaluation
############################
# by default the w2 is out-flow matrix
############################
leaveOneOut <- function(demos, ca, w2, Y, coeff=FALSE, normalize=FALSE, socialnorm="bydestination", exposure="exposure", lagstr="1111") {
    N <- length(Y)
    lags <- unlist(strsplit(lagstr, split=""))
    # leave one out evaluation
    if (coeff) {
        if (exposure == "exposure")
            cat("Coefficients\n", "(intercepts)", names(demos[, ! c(colnames(demos) %in% "total.population")]), "")
        else
            cat("Coefficients\n", "(intercepts)", names(demos), "")
        if (lags[1] == "1")
            cat("social.lag ")
        if (lags[2] == "1")
            cat("spatial.lag ")
        if (lags[3] == "1")
            cat("social.lag.disadv")
        if (lags[4] == "1")
            cat("spatial.lag.disadv")
        cat("\n")
    }

    w1 <- spatialWeight(ca)

    errors <- foreach ( i = 1:N, .combine="cbind", .export=c("normalize.social.lag", "spatialWeight", "learnNB"), 
					   .packages=c("sp", "spdep", "glmmADMB") ) %dopar%  {
        F <- demos[-i, , drop=FALSE]
        y <- Y[-i]
        y2 <- F$disadvantage.index # demos$poverty.index[-i] #
        test.dn <- demos[i, , drop=FALSE]

        # social lag
        if (lags[1] == "1" || lags[3] == "1") {
            # training set
            sco <- w2[-i, -i]

            sco <- normalize.social.lag(sco, socialnorm)

            if (lags[1] == "1")
                F[,'social.lag'] = as.vector(sco %*% y)
            if (lags[3] == "1")
                F[,'social.lag.disadv'] = as.vector(sco %*% y2)

            # testing point i
            if (socialnorm == "bysource") {
                social.lag <- w2[i,-i]  %*% y / sum(w2[i,-i])
                sl.disadv <- w2[i, -i] %*% y2 / sum(w2[i,-i])
            } else if (socialnorm == "bydestination") {
                w2h <- t(w2)
                social.lag <- w2h[i,-i]  %*% y / sum(w2h[i,-i])
                sl.disadv <- w2h[i, -i] %*% y2 / sum(w2h[i,-i])
            } else if (socialnorm == "bypair") {
                social.lag <- (w2[i,-i] / s) %*% y
                sl.disadv <- (w2[i,-i] / s) %*% y2
            } else {
                social.lag <- w2[i,-i] %*% y
                sl.disadv <- w2[i,-i] %*% y2
            }
            stopifnot( length(social.lag) == 1)
            if (lags[1] == "1")
                test.dn['social.lag'] = social.lag
            if (lags[3] == "1")
                test.dn['social.lag.disadv'] = sl.disadv
        }

        # spatial lag
        if (lags[2] == "1" || lags[4] == "1") {
            # training set
            stopifnot(rownames(w2) != NULL)
            dropCA <- rownames(w2[i, ,drop=FALSE])
            spt <- spatialWeight(ca, as.numeric(dropCA) )
            if (lags[2] == "1")
                F[,'spatial.lag'] = as.vector(spt %*% y)
            if (lags[4] == "1")
                F[,'spatial.lag.disadv'] = as.vector(spt %*% y2)
            # testing point i
            spatial.lag <- w1[i,-i] %*% y
            spl.disadv <- w1[i,-i] %*% y2
            if (lags[2] == "1")
                test.dn['spatial.lag'] <- spatial.lag
            if (lags[4] == "1")
                test.dn['spatial.lag.disadv'] <- spl.disadv
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
        dat <- data.frame(Y[-i], F)
        colnames(dat)[1] <- "y"

        stopifnot( all(is.finite(as.matrix(dat))) )

        mod <- learnNB(dat, exposure)

        if (inherits(mod, "error")) {
            warning("error in glmmadbm fitting - skip this iteration\n")
            return(NULL)
        }
        
        
        ybar <- predict(mod, newdata=test.dn, type=c('response'))
        if (coeff) {
            return ( c(abs(ybar - Y[i]), as.vector(mod$b)) )
        }
        else {
            return(abs(ybar - Y[i]))
        }
    }
    # end of anonymous function

    if (coeff) {
        cat(rowSums(errors)[-1]  / N, "\n")
        return (errors[1,])
    }

    return(mean(errors))
}








############################
# by default the w2 is out-flow matrix
############################
leaveOneOut.PermuteLag <- function(demos, ca, w2, Y, normalize=FALSE, socialnorm="bydestination", exposure="exposure",  lagstr="1111") {
    N <- length(Y)
    lags.flg <- unlist(strsplit(lagstr, split=""))

    # social lag
    sco <- normalize.social.lag(w2, socialnorm)
    social.lag <- sco %*% Y
    social.lag.p <- sample(social.lag)


    sociallag.disadv <- sco %*% demos$disadvantage.index
    sociallag.disadv.p <- sample(sociallag.disadv)


    # spatial lag
    w1 <- spatialWeight(ca)
    spatial.lag <- w1 %*% Y
    spatial.lag.p <- sample(spatial.lag)

    spatiallag.disadv <- w1 %*% demos$disadvantage.index
    spatiallag.disadv.p <- sample(spatiallag.disadv)

    
    
    mae = c()

    lags <- c()
    if (lags.flg[1] == "1") {
        lags <- c(lags, "social")
    }
    if (lags.flg[2] == "1") {
        lags <- c(lags, "spatial")
    }
    if (lags.flg[3] == "1") {
        lags <- c(lags, "social.disadv")
    }
    if (lags.flg[4] == "1") {
        lags <- c(lags, "spatial.disadv")
    }
    # control which lags to use
    for (lag in lags) {
        # leave one out evaluation
        errors <- foreach ( i=1:N, .combine="cbind", .export=c("normalize.social.lag", "spatialWeight", "learnNB"), 
						   .packages=c("sp", "spdep", "glmmADMB") ) %dopar%  {
            F <- demos[-i, , drop=FALSE]
            test.dn <- demos[i, , drop=FALSE]



            # social based lags
            if (lags.flg[1] == "1" || lags.flg[3] == "1") {
                
                if (lag == "social") {
                    F[,'social.lag'] <- social.lag.p[-i]
                    test.dn['social.lag'] <- social.lag.p[i]
                } else {
                    if (lags.flg[1] == "1") {
                        F[,'social.lag'] = social.lag[-i]
                        test.dn['social.lag'] <- social.lag[i]
                    }
                }

                if (lag == "social.disadv"){
                    F[,'social.lag.disadv'] <- sociallag.disadv.p[-i]
                    test.dn['social.lag.disadv'] <- sociallag.disadv.p[i]
                } else {
                    if (lags.flg[4] == "1") {
                        F[,'social.lag.disadv'] <- sociallag.disadv[-i]
                        test.dn['social.lag.disadv'] <- sociallag.disadv[i]
                    }
                }
            }



            # spatial based lags
            if (lags.flg[2] == "1" || lags.flg[4] == "1") {

                if (lag == "spatial") { # spatial lag should be permuted
                    F[,'spatial.lag'] <- spatial.lag.p[-i]
                    test.dn['spatial.lag'] <- spatial.lag.p[i]
                } else {
                    if (lags.flg[2] == "1") { # spatial lag is included but not permuted
                        F[,'spatial.lag'] <- spatial.lag[-i]
                        test.dn['spatial.lag'] <- spatial.lag[i] 
                    }
                }

                if (lag == "spatial.disadv") {
                    F[,'spatial.lag.disadv'] <- spatiallag.disadv.p[-i]
                    test.dn['spatial.lag.disadv'] <- spatiallag.disadv.p[i]
                } else {
                    if (lags.flg[4] == "1") {
                        F[,'spatial.lag.disadv'] <- spatiallag.disadv[-i]
                        test.dn['spatial.lag.disadv'] <- spatiallag.disadv[i]
                    }
                }                
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


            mod <- learnNB( dat, exposure )

            if (inherits(mod, "error")) {
                 warning("error in glmmadbm fitting - skip this iteration\n")
                 return(NULL)
             }

            
            ybar <- predict(mod, newdata=test.dn, type=c('response'))
            return(abs(ybar - Y[i]))
        }
        # end of anonymous function

        mae[lag] <- mean(errors)
    }

    return(mae)
}   


