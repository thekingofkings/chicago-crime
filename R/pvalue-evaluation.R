library(MASS)




leaveOneOut <- function(demos, w1, w2, Y, coeff=FALSE, normalize=FALSE) {
    N <- length(Y)
    # leave one out evaluation
    errors = c()
    if (coeff) {
        coeffs = vector("double", 11)
    }
    for ( i in 1:N ) {
        F <- demos[-i,]
        spt <- w1[-i, -i]
        sco <- w2[-i, -i]
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
        social.lag <- w2[i,-i] %*% y
        dn <- data.frame(demos[i,], spatial.lag, social.lag)
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



leaveOneOut.PermuteLag <- function(demos, w1, w2, Y, normalize=FALSE) {
    N <- length(Y)
    # permute lag matix is equivalent to permute Y
    y = sample(Y)
    mae = c()
    for (lag in c("social", "spatial")) {
        # leave one out evaluation
        errors = c()
        for ( i in 1:N ) {
            F <- demos[-i,]
            spt <- w1[-i, -i]
            sco <- w2[-i, -i]
            

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
            social.lag <- w2[i,-i] %*% y[-i]
            dn <- data.frame(demos[i,], spatial.lag, social.lag)
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



demos <- read.table('pvalue-demo.csv', header=TRUE, sep=",")
# spatial matrix
w1 <- as.matrix(read.csv('pvalue-spatiallag.csv', header=FALSE))
# social matrix
w2 <- as.matrix(read.csv('pvalue-sociallag.csv', header=FALSE))
# crime
Y <- read.csv('pvalue-crime.csv', header=FALSE)
Y <- Y$V1
Y <- Y / demos$total.population * 10000

demos$total.population = log(demos$total.population)


mae.org <- leaveOneOut(demos, w1, w2, Y, coeff=TRUE, normalize=TRUE)
cat(mae.org, "\n")
itersN <- 100
normalize=TRUE

# permute demographics
for (i in 1:ncol(demos)) {
    
    cat(colnames(demos)[i], ' ')
    cnt <- 0
    
    for (j in 1:itersN) {
        demos.copy <- demos
        # permute features
        demos.copy[,i] <- sample( demos[,i] )
        mae <- leaveOneOut(demos.copy, w1, w2, Y, normalize=normalize)
        if (mae.org > mae) {
            cnt = cnt + 1
        }
    }
    cat(cnt / itersN, '\n')
}


# permute lag
cnt.social = 0
cnt.spatial = 0
for (j in 1:itersN) {
    mae = leaveOneOut.PermuteLag(demos, w1, w2, Y, normalize)
    if (mae.org > mae[1]) { # first one is social lag
        cnt.social = cnt.social + 1
    }
    if (mae.org > mae[2]) {
        cnt.spatial = cnt.spatial + 1
    }
}

cat("social.lag ", cnt.social / itersN, "\nspatial.lag", cnt.spatial / itersN, "\n")
