# Date: 1/14/2016
# Hongjian
# 
# This is the old version of negative binomial regression permutation test.
# The disadvantage of this script is that, each time the newly permuated
# matrix is genrated by Python, and write on local disk. This is silly and
# not efficient.
# 
# Since the permutation relies on the same input f and y, we can permutate
# within this script.
#
# The new permutation implementation keeps the old name 
# 	``nbr_permutation_test.R''
# and This one is renamed as
# 	``nbr_permutation_test_obsolete.R''

library(MASS)

options(warn=-1)
# cat('Rscript for glm.nb\n')

y <- read.csv('Y.csv', header=FALSE)
f <- read.csv('f.csv')

dat <- data.frame(y, f)
mod <- glm.nb( data=dat )

coeff <- summary(mod)$coefficients[,'Estimate']

write(paste(coeff, sep=" ", collapse=","), file="coefficients.txt", append=TRUE)





