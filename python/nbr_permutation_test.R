# Date: 1/14/2016
# Hongjian
# 
# This is the new version of negative binomial regression permutation test.
# 
# Since the permutation relies on the same input f and y, we can permutate
# within this script.
#
# The new permutation implementation in this file replaces the old 
# implementation
# 	``nbr_permutation_test_obsolete.R''
#
# More importantly, the original NB model consider the spatial lag in the 
# wrong way, which is corrected in this implementation.

library(MASS)

options(warn=-1, digits=3)
# cat('Rscript for glm.nb\n')

args <- commandArgs(trailingOnly = TRUE)
iters = strtoi(args)


y <- read.csv('Y.csv', header=FALSE)
f <- read.csv('f.csv')

dat <- data.frame(y, f)

# calculate the original coefficient coeff
mod <- glm.nb( data=dat )
coeff0 <- summary(mod)$coefficients[,'Estimate']

# iterate through all columns
nf = ncol(dat)
signif = numeric(nf-2)
for (i in 2:nf ) {
	if (colnames(dat)[i] == 'intercept') {
		next
	}
	cat(colnames(dat)[i], "\n")

	# permutate focal column
	coeff <- matrix(0, nrow=iters, ncol=nf-1)
	for (j in 1:iters) {
		dat[,i] = sample( dat[,i] )
		mod <- glm.nb( data=dat )
		coeff[j,] <- summary(mod)$coefficients[, 'Estimate']
	}

	for ( c in coeff[,i-1] ) {
		if (c > coeff0[i-1])
			signif[i-1]  = signif[i-1] + 1
	}

	signif[i-1] = signif[i-1] / iters
	cat(paste(coeff0[i-1], signif[i-1]), '\n')


	# visualize the significance
	hist(coeff[,i-1], breaks=20, xlim=c(min(coeff[,i-1], coeff0[i-1]), max(coeff[,i-1], coeff0[i-1])),
		 xlab='coefficients',
		 main=paste('Histgram of', colnames(dat)[i], '(', round(coeff0[i-1], digits=3), ')'))
	abline(v=coeff0[i-1], lty=3)
	pdf(file=paste(colnames(dat)[i], '.pdf', sep=''))
}






