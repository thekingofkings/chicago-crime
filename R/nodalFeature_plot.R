
library(maptools)
library(gplots)
library(grDevices)

categories = c('Food', 'Residence', 'Travel', 'Arts & Entertainment', 'Outdoors & Recreation', 
			   'College & Education', 'Nightlife', 'Professional', 'Shops', 'Event')
CAs = readShapeSpatial('../data/ChiCA_gps/ChiCaGPS')
ids = as.numeric(as.character(CAs$AREA_NUMBE))
feature = read.csv('poi_dist.csv', header=FALSE)
f_ordered <- list()
for (i in 1:77) {
	f_ordered[[i]] = feature[ids[i],]
}
fts <- matrix(unlist(f_ordered), nrow=77, byrow=T)


# build color
chooseColor <- colorRampPalette( c('white', 'red') )


for ( i in 1:10 ) {
	pdf(file=paste('poi-dist', i, '.pdf', sep=''), width=7, height=7 )
	par(mai=c(0,0,0,0))
	plot(CAs, border='blue', col=chooseColor(21)[findInterval(fts[,i], seq(0, max(fts), max(fts)/20))])
	cr = coordinates(CAs)
	text(cr, labels=as.character(ids))
	dev.off()
}
