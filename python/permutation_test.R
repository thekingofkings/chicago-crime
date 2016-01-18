# Date: 1/14/2016
# Hongjian
# 
# This is the new version of regression permutation test.
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
library(spdep)

options(warn=-1, digits=3)
# cat('Rscript for glm.nb\n')

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 0) {
	iters = strtoi(args)
} else {
	iters = 10
}

y <- read.csv('Y.csv', header=FALSE)
f <- read.csv('f.csv')
flr <- read.csv('flr.csv')
W <- as.matrix(read.csv('W.csv', header=FALSE), nrow=nrow(y), ncol=nrow(y) )
W2 <- as.matrix( read.csv('W2.csv', header=FALSE), nrow=nrow(y), ncol=nrow(y) )

dat <- data.frame(y, f)
datlr <- data.frame(y, flr)

# calculate the original nb coefficient coeff

w <- mat2listw(W)
w2 <- mat2listw(W2)

lrm <- sacsarlm(V1 ~ total.population + population.density + poverty.index + disadvantage.index
				+ residential.stability + ethnic.diversity + pct.black + pct.hispanic + temporal.lag, 
				data = datlr, listw = w)
coeff0 <- lrm$coefficients

# iterate through all columns
nf = ncol(dat)
signif = numeric(nf)
for (i in 2:nf ) {
	if (colnames(dat)[i] == 'intercept' || colnames(dat)[i] == 'spatial.lag' ||
		colnames(dat)[i] == 'social.lag' ) {
		next
	}
	cn = colnames(dat)[i]
	cat(cn, "\n")

	# permutate focal column
	coeff <- matrix(0, nrow=iters, ncol=10, dimnames=list(NULL, names(coeff0)) )
	for (j in 1:iters) {
		dat[,i] = sample( dat[,i] )
		lrm <- sacsarlm(V1 ~ total.population + population.density + poverty.index + disadvantage.index
						+ residential.stability + ethnic.diversity + pct.black + pct.hispanic+ temporal.lag, 
						data = datlr, listw = w)
		coeff[j,] <- lrm$coefficients
	}

	for ( c in coeff[,cn] ) {
		if (c > coeff0[cn])
			signif[i]  = signif[i] + 1
	}

	signif[i] = signif[i] / iters
	cat(paste(coeff0[cn], signif[i]), '\n')


	# visualize the significance
	pdf(file=paste(colnames(dat)[i], '.pdf', sep=''))
	hist(coeff[,cn], breaks=20, xlim=c(min(coeff[,cn], coeff0[cn]), max(coeff[,cn], coeff0[cn])),
		 xlab='coefficients',
		 main=paste('Histgram of', colnames(dat)[i], '(', round(coeff0[cn], digits=3), ')'))
	abline(v=coeff0[cn], lty=3)
	dev.off()
}






