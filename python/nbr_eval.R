

library(MASS)


cat('Rscript for glm.nb\n')

y <- read.csv('Y.csv', header=FALSE)
f <- read.csv('f.csv')



errors = c()
for (i in 1:77) {
	dat <- data.frame( y[-i,], f[-i,] )
	mod <- glm.nb( data=dat )
	ybar <- predict(mod, newdata=f[i,], type=c('response') )
	
	cat(paste(y[i,], ybar, '\n'))
	errors = c(errors, abs(ybar - y[i,]) )
}

cat(paste("MAE", mean(errors), "SD", sd(errors), "MRE", mean(errors) / mean(y$V1), '\n'))




