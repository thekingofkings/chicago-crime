
library(maptools)

CAs = readShapeSpatial('../data/ChiCA_gps/ChiCaGPS')

pdf('id-graph.pdf')
plot(CAs, border='blue')
cr = coordinates(CAs)
#text(cr, labels=as.character(1:77))
points(cr)
dev.off()
