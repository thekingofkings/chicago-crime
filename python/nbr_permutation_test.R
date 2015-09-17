

library(MASS)


cat('Rscript for glm.nb\n')

y <- read.csv('Y.csv', header=FALSE)
f <- read.csv('f.csv')

dat <- data.frame(y, f)
mod <- glm.nb( data=dat )

coeff <- summary(mod)$coefficients[,'Estimate']

write(paste(coeff, sep=" ", collapse=","), file="coefficients.txt", append=TRUE)





