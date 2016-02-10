
library(maptools)
library(gplots)
library(grDevices)

categories = c('Food', 'Residence', 'Travel', 'Arts & Entertainment', 'Outdoors & Recreation', 
			   'College & Education', 'Nightlife', 'Professional', 'Shops', 'Event')
CAs = readShapeSpatial('../data/ChiCA_gps/ChiCaGPS')
cr = coordinates(CAs)

ids = as.numeric(as.character(CAs$AREA_NUMBE))


#args <- commandArgs(trailingOnly=T)
args = c('error')

if (length(args) < 1) {
	cat('Usage: <poi|demo|crime|poi_count|error>\n')
} else if (args[1] == 'poi') {

	feature = read.csv('poi_dist.csv', header=FALSE)
	f_ordered <- list()
	for (i in 1:77) {
		f_ordered[[i]] = feature[ids[i],]
	}
	fts <- matrix(unlist(f_ordered), nrow=77, byrow=T)


	# build color
	chooseColor <- colorRampPalette( c('white', 'darkgreen') )


	for ( i in 1:10 ) {
		pdf(file=paste('poi-dist', i, '.pdf', sep=''), width=7, height=7 )
		par(mai=c(0,0,0,0))
		plot(CAs, border='blue', col=chooseColor(21)[findInterval(fts[,i], seq(min(fts[,i]), max(fts[,i]), (max(fts[,i])-min(fts[,i]))/20))])
		text(cr, labels=as.character(ids))
		dev.off()
	}

} else if (args[1] == 'poi_count') {
	
	poic = read.csv('../python/POI_cnt.csv', header=FALSE)
	f_ordered <- list()
	for (i in 1:77) {
		f_ordered[[i]] = poic[ids[i],]
	}
	fts <- matrix(unlist(f_ordered), nrow=77, byrow=T)


	# build color
	chooseColor <- colorRampPalette( c('white', 'darkgreen') )

	for (i in 1:3) { 
		pdf(file=paste('poi-cnt', i, '.pdf', sep=''), width=7, height=7) 
		par(mai=c(0,0,0,0)) 
		plot(CAs, border='darkblue', col=chooseColor(51)[findInterval(fts[,i], seq(min(fts[,i]), 6000, (6000-min(fts[,i]))/50))])
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
	demos = matrix(unlist(demo_ord), nrow=77, byrow=F)

	chooseColor <- colorRampPalette( c('white', 'blue') )


	for ( i in 1:8 ) {
		pdf(file=paste('demo-f', i, '.pdf', sep=''), width=7, height=7 )
		par(mai=c(0,0,0,0))
		plot(CAs, border='darkgreen', col=chooseColor(21)[findInterval(demos[,i], 
																  seq(min(demos[,i]), 
																	  max(demos[,i]), (max(demos[,i]) - min(demos[,i]))/20))])
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

	text(cr, labels=as.character(ids))
	dev.off()
} else if (args[1] == 'error') {
	crimes = read.csv('crime-ca.csv',  header=F)
	crimes_ord <- list()
	for ( i in 1:77 ) {
		crimes_ord[[i]] <- crimes[ids[i],]
	}
	crimes = as.vector(crimes$V1)
	
	errors = read.table('error-by-region.txt', sep='', header=FALSE)
	err_ord <- list()
	for ( i in 1:77 ) {
		err_ord[[i]] <- errors[,ids[i]]
	}
	errors = matrix(unlist(err_ord), nrow=3, byrow=F)
	
	
	cat(paste(mean(errors[1,]), mean(errors[2,]), mean(errors[3,]), sep='\t'), '\n')
	p_imp = errors[2,] - errors[1,]
	t_imp = errors[3,] - errors[1,]
	
	colorMap <- colorRampPalette( c('red', 'white', 'blue') )
	
	pdf(file='poi-improve.pdf', width=7, height=7)
	par(mai=c(0,0,0,0))
	plot(CAs, border='black', col=colorMap(21)[findInterval(p_imp, seq(- max(p_imp), max(p_imp), (2 * max(p_imp)) / 20))])
	text(cr, labels=as.character(ids))
	dev.off()
	
	
	pdf(file='taxi-improve.pdf', width=7, height=7)
	par(mai=c(0,0,0,0))
	plot(CAs, border='black', col=colorMap(21)[findInterval(t_imp, seq(- max(t_imp), max(t_imp), (2* max(t_imp)) / 20))])
	text(cr, labels=as.character(ids))
	dev.off()
	
	
	
	if (FALSE) {
	mier = min(errors)
	mxer = max(errors)
	for ( j in 1:3) {
		er = errors[j,]
		rela_err = er / crimes
		
		colorMap <- colorRampPalette( c('white', 'red') )

		pdf(file=paste('mae', j, '.pdf', sep=''), width=7, height=7)
		par(mai=c(0,0,0,0))
		plot(CAs, border='black', col=colorMap(41)[findInterval(er, seq(mier, mxer,
																			(mxer - mier) / 40))])
		text(cr, labels=as.character(ids))
		dev.off()
		
		
		pdf(file=paste('mre', j, '.pdf', sep=''), width=7, height=7)
		par(mai=c(0,0,0,0))
		plot(CAs, border='black', col=colorMap(41)[findInterval(rela_err, seq(0, 1, 1 / 40))])
		text(cr, labels=as.character(ids))
		dev.off()
	}
	}
	
} else {
	cat('Usage: <poi|demo|crime|poi_count|error>\n')
}
