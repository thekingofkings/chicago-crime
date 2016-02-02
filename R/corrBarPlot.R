corr_ca <- read.csv(con <- textConnection('0.324394971027
			  0.235673422821
			  0.0745214228252
			  0.275504753056
			  0.261190374394
			  0.223771141573
			  0.279942642382
			  0.259269714939
			  0.323913559267
			  0.288823289313'), header=FALSE)
close(con)


corr_tract <- read.csv( con<- textConnection('0.327584916166
											0.074239438061
											0.133783843612
											0.265826316389
											0.216258143229
											0.250871184023
											0.193533609036
											0.349164584765
											0.324864038947
											0.247768867858'), header=FALSE)
close(con)


barlabs = c('Food', 'Residence', 'Travel', 'Entertainment', 
			'Recreation', 'Education', 'Nightlife', 
			'Professional', 'Shops', 'Event')

counts <- matrix(data=c(corr_ca$V1, corr_tract$V1), nrow=2, byrow=TRUE)
pdf('bar-corr-socialflow.pdf', height=6, width=8)
par(las=2, mar=c(7,3,2,0))
barplot( counts, ylab="correlation coefficient",
		 col=c('darkblue', 'red'), legend=c('CA', 'Tract'), beside=TRUE,
		 names.arg=barlabs, args.legend=c(x='top')) 
box()
grid()

dev.off()




# plot demographis correlation
corr_demo <- read.csv( con<- textConnection(' 0.722374033897
											0.35093768889
											0.294226059912
											0.30810219065
											-0.182884883457
											-0.260857809453
											0.348864415942
											-0.0798976457118'), header=FALSE)
close(con)
corr_demo <- as.vector(corr_demo$V1)

barlabs = c('total population', 'population density', 'poverty index', 'disadvantage index', 
			'residential stability', 'ethnic diversity', 'pct black', 'pct hispanic')

color_keys = character(length(corr_demo))
for (i in 1:length(corr_demo)) {
	if ( corr_demo[i] >= 0 ) {
		color_keys[i] = 'darkgreen'
	} else {
		color_keys[i] = 'red'
	}
}

pdf('bar-corr-demo.pdf', height=6, width=8)
par(las=2, mar=c(10,3,2,0))
barplot( corr_demo, ylab='correlation coefficient', col=color_keys, names.arg=barlabs)
grid()
box()

dev.off()

