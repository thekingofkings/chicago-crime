
library(maptools)
library(gplots)
library(grDevices)

categories = c('Food', 'Residence', 'Travel', 'Arts & Entertainment', 'Outdoors & Recreation', 
			   'College & Education', 'Nightlife', 'Professional', 'Shops', 'Event')
CAs = readShapeSpatial('../data/ChiCA_gps/ChiCaGPS')
ids = as.numeric(as.character(CAs$AREA_NUMBE))
correlations = read.csv('poi_correlation_ca.csv', header=FALSE)
corr_ordered <- list()
for (i in 1:77) {
	corr_ordered[[i]] = correlations[ids[i],]
}
corr <- matrix(unlist(corr_ordered), nrow=77, byrow=F)


# build color
chooseColor <- colorRampPalette( c('blue','white',  'red') )


for ( i in 1:10 ) {
	pdf(file=paste('poi-correlation', i, '.pdf', sep=''), width=7, height=7 )
	par(mai=c(0,0,0,0))
	plot(CAs, border='blue', col=chooseColor(41)[findInterval(corr[,i], seq(-1, 1, 0.05))])
	cr = coordinates(CAs)
	text(cr, labels=as.character(ids))
	dev.off()
}
