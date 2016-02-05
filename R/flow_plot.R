library(maptools)
library(gplots)
library(grDevices)



CAs = readShapeSpatial('../data/ChiCA_gps/ChiCaGPS')
cr = coordinates(CAs)
ids = as.numeric(as.character(CAs$AREA_NUMBE))


tf = read.csv('TF.csv', header=F)
tf_ord <- list()
for (i in 1:77) {
	tf_ord[[i]] = tf[ids[i],]
}
m <- matrix(unlist(tf_ord), nrow=77, byrow=F)
	
	

for (i in 1:77) {
	for (j in 1:77) {
		if (m[i,j] < 1000)
			m[i,j] = 0
	}
}

n = as.network.matrix(m, matrix.type='adjacency')


pdf(file='taxi-flow-ca.pdf', width=7, height=7)
par(mai=c(0,0,0,0))
plot(CAs, border='darkgreen')
plot(n, coord=cr, vertex.cex=0.5, vertex.col='blue', usearrows=F, new=F)
dev.off()



count = function( t, m ) {
	cnt = 0
	for (i in m) {
		if (i >= t)
			cnt = cnt + 1
	}
	return (cnt)
}