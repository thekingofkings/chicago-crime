

library(MASS)

y <- read.csv('Y.csv', header=FALSE)
f <- read.csv('f.csv')



errors = c()
for (i in 1:77) {
	dat <- data.frame( y[-i,], f[-i,] )
	mod <- glm.nb( data=dat )
	ybar <- predict(mod, newdata=f[i,], type=c('response') )
	
	print(paste(y[i,], ybar))
	errors = c(errors, abs(ybar - y[i,]) )
}

print(paste("MAE", mean(errors), "SD", sd(errors), "MRE", mean(errors) / mean(y$V1)))




