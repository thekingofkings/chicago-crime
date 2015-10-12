

library(MASS)
options(warn=-1)

args <- commandArgs(TRUE)


if (length(args) == 1 && args[1] == 'verbose') {
    cat('Rscript for glm.nb\n')
}

y <- read.csv('Y.csv', header=FALSE)
f <- read.csv('f.csv')



errors = c()
cv <- sample(1:77*11, 50, replace=F)
for (i in cv) {
	dat <- data.frame( y[-i,], f[-i,] )
	mod <- glm.nb( data=dat )
	ybar <- predict(mod, newdata=f[i,], type=c('response') )
	
    if (length(args) == 1 && args[1] == 'verbose') {
        cat(paste(y[i,], ybar, '\n'))
    }
	errors = c(errors, abs(ybar - y[i,]) )
}


if (length(args) == 1 && args[1] == 'verbose') {
    cat(paste("MAE", mean(errors), "SD", sd(errors), "MRE", mean(errors) / mean(y$V1), '\n'))
} else {
    cat(paste(mean(errors), sd(errors), mean(errors) / mean(y$V1)))
}


