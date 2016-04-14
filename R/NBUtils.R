# Implement some utility function for NB regression with glmmADMB


library(glmmADMB)
library(spdep)


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






# one run of the leave-one-out evaluation
############################
# by default the w2 is out-flow matrix
############################
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
        y <- Y[-i]   # demos$poverty.index[-i] #
        test.dn <- demos[i, , drop=FALSE]
        
        if (SOCIALLAG) {
            # training set
            sco <- w2[-i, -i]

            soc <- normalize.social.lag(sco, socialnorm)

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
            dropCA <- rownames(w2[i, ,drop=FALSE])
            spt <- spatialWeight(ca, as.numeric(dropCA) )
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
        dat <- data.frame(Y[-i], F)
        names(dat)[1] <- "y"
        
        stopifnot( all(is.finite(as.matrix(dat))) )
        mod <- tryCatch( {
            if (exposure == "exposure") {
                mod <- glmmadmb(y ~ .  + offset(total.population), data=dat, family="nbinom", verbose=FALSE)
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
        return (errors)
    }

    return(mean(errors))
}








############################
# by default the w2 is out-flow matrix
############################
leaveOneOut.PermuteLag <- function(demos, ca, w2, Y, normalize=FALSE, socialnorm="bydestination", exposure="exposure",  SOCIALLAG=TRUE, SPATIALLAG=TRUE) {
    N <- length(Y)
    toPermute <- Y # demos$poverty.index
    # permute lag matix is equivalent to permute Y
    y = sample(toPermute)
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
                dropCA <- rownames(w2[i, ,drop=FALSE])
                spt <- spatialWeight(ca, as.numeric(dropCA) )

                if (lag == "social") {
                    F[,'spatial.lag'] = as.vector(spt %*% toPermute[-i])
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

                sco <- normalize.social.lag(sco, socialnorm)
                
                if (lag == "social") {
                    F[,'social.lag'] = as.vector(sco %*% y[-i])
                } else {
                    F[,'social.lag'] = as.vector(sco %*% toPermute[-i])
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
                    mod <- glmmadmb(y ~ . + offset(total.population), data=dat, family="nbinom", verbose=FALSE)
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
        stopifnot( sco[4, 4] == 0, nrow(sco)==N, ncol(sco)==N, sum(sco) == 1)
    }

    return (sco)
}
