
library(maptools)
library(gplots)
library(grDevices)

categories = c('Food', 'Residence', 'Travel', 'Arts & Entertainment', 'Outdoors & Recreation', 
			   'College & Education', 'Nightlife', 'Professional', 'Shops', 'Event')
CAs = readShapeSpatial('../data/ChiCA_gps/ChiCaGPS')
ids = as.numeric(as.character(CAs$AREA_NUMBE))


args <- commandArgs(trailingOnly=T)

if (length(args) < 1) {
	cat('Usage: <poi|demo|crime>\n')
} else if (args[1] == 'poi') {

	feature = read.csv('poi_dist.csv', header=FALSE)
	f_ordered <- list()
	for (i in 1:77) {
		f_ordered[[i]] = feature[ids[i],]
	}
	fts <- matrix(unlist(f_ordered), nrow=77, byrow=T)


	# build color
	chooseColor <- colorRampPalette( c('white', 'yellow') )


	for ( i in 1:10 ) {
		pdf(file=paste('poi-dist', i, '.pdf', sep=''), width=7, height=7 )
		par(mai=c(0,0,0,0))
		plot(CAs, border='blue', col=chooseColor(21)[findInterval(fts[,i], seq(min(fts[,i]), max(fts[,i]), (max(fts[,i])-min(fts[,i]))/20))])
		cr = coordinates(CAs)
		text(cr, labels=as.character(ids))
		dev.off()
	}

} else if (args[1] == 'demo') {


	demo_header = c('total population', 'population density', 'poverty index', 
					'disadvantage index', 'residential stability',
					'ethnic diversity', 'pct black', 'pct hispanic')
	demos = read.csv('demo-f.csv', header=FALSE)
	demo_ord <- list()
	for (i in 1:77) {
		demo_ord[[i]] = demos[ids[i],]
	}
	demos = matrix(unlist(demo_ord), nrow=77, byrow=T)

	chooseColor <- colorRampPalette( c('white', 'blue') )


	for ( i in 1:8 ) {
		pdf(file=paste('demo-f', i, '.pdf', sep=''), width=7, height=7 )
		par(mai=c(0,0,0,0))
		plot(CAs, border='darkgreen', col=chooseColor(21)[findInterval(demos[,i], 
																  seq(min(demos[,i]), 
																	  max(demos[,i]), (max(demos[,i]) - min(demos[,i]))/20))])
		cr = coordinates(CAs)
		text(cr, labels=as.character(ids))
		dev.off()
	}

} else if (args[1] == 'crime') {
	crimes = read.csv('../python/Y.csv', header=FALSE)
	crimes_ord <- list()
	for ( i in 1:77 ) {
		crimes_ord[[i]] <- crimes[ids[i],]
	}
	crimes = as.vector(crimes$V1)

	colorMap <- colorRampPalette( c('white', 'red') )

	pdf(file='crime-ca.pdf', width=7, height=7)
	par(mai=c(0,0,0,0))
	plot(CAs, border='black', col=colorMap(21)[findInterval(crimes, seq(min(crimes), max(crimes),
																		(max(crimes)-min(crimes)) / 20))])

	cr = coordinates(CAs)
	text(cr, labels=as.character(ids))
	dev.off()
} else {
	cat('Usage: <poi|demo|crime>\n')
}
