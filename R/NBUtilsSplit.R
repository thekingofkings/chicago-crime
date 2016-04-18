# train the Negative Binomial model separately on joint data
source("NBUtils.R")



# Show coefficient, normalize all features for regression, normalize the social flow by source
# use both social/spatial lag
leaveOneOut.split <- function( demos, ca, w2, Y, idN, idS, socialnorm="bysource", exposure="exposure" )
{
    N <- length(Y)

    cat("Coefficients\n", "(intercepts)", names(demos), "social.lag", "spatial.lag", "\n")

    w1 <- spatialWeight(ca)

    errors <- foreach(i = 1:N, .combine="cbind", .export=c("normalize.social.lag", "spatialWeight", "learnNB"),
                      .packages=c("sp", "spdep", "glmmADMB") )  %dopar% {
        F <- demos[-i, , drop=FALSE]
        y <- Y[-i]
        test.dn <- demos[i, ,drop=FALSE]

        # social lag training set
        sco <- w2[-i, -i]
        sco <- normalize.social.lag(sco, socialnorm)
        F[,"social.lag"] <- as.vector(sco %*% y)

        # social lag testing point
        if (socialnorm == "bysource") {
            social.lag <- w2[i, -i] %*% y / sum(w2[i,-i])
        } else if (socialnorm == "bydestination") {
            w2h <- t(w2)
            social.lag <- w2h[i,-i] %*% y / sum(w2h[i,-i])
        } else if (socialnorm == "bypair") {
            social.lag <- (w2[i,-i] / s) %*% y
        } else {
            social.lag <- w2[i, -i] %*% y
        }
        test.dn["social.lag"] <- social.lag



        # spatial lag training set
        stopifnot( rownames(w2) != NULL )
        dropCA <- rownames(w2[i, , drop=FALSE])
        spt <- spatialWeight(ca, as.numeric(dropCA))
        F[,"spatial.lag"] <- as.vector(spt %*% y)

        # spatial lag testing point
        spatial.lag <- w1[i,-i] %*% y
        test.dn["spatial.lag"] <- spatial.lag


        # normalize independent variables
        F <- scale(F, center=TRUE, scale=TRUE)
        F.center <- as.vector(attributes(F)["scaled:center"][[1]])
        F.scale <- as.vector(attributes(F)["scaled:scale"][[1]])
        stopifnot( length(F.center) == length(F.scale), length(test.dn) == length(F.center))
        test.dn <- data.frame(scale(test.dn, center=F.center, scale=F.scale))
        

        # fit NB model
        dat.full <- data.frame(Y[-i], F)
        colnames(dat.full)[1] <- "y"
        rns <- 1:N
        rns <- rns[! rns %in% i]
        rns <- as.character( rns )
        stopifnot( nrow(dat.full) == N-1, length(rns) == N-1)
        rownames(dat.full) <- rns

        # select only subset of data to learn the NB model
        if ( i %in% idN ) {
            idN.noi <- idN[! idN %in% i]
            dat <- dat.full[as.character(idN.noi), ]
        } else {
            idS.noi <- idS[! idN %in% i]
            dat <- dat.full[as.character(idS.noi), ]
        }

        stopifnot( is.data.frame(dat.full), is.data.frame(dat) )
        #dat <- as.data.frame(dat)
        #dat[ is.infinite(dat) ] <- 0
        stopifnot( nrow(dat) <  nrow(dat.full) )

        mod <- learnNB(dat, exposure)

        # if current model does not converge, skip
        if (inherits(mod, "error")) {
            warning("error in glmmadbm fitting - skip this iteration\n")
            return(NULL)
        }

        # prediction
        ybar <- predict(mod, newdata=test.dn, type=c("response"))
        stopifnot( is.finite(ybar) )
        return ( c(abs(ybar-Y[i]), as.vector(mod$b)) )

    }

    cat(rowSums(errors)[-1] / N, "\n")
    return (errors[1,])
}

        
                          
