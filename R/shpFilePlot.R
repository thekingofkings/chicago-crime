
library(maptools)

CAs = readShapeSpatial('../data/ChiCA_gps/ChiCaGPS')
ids = as.numeric(as.character(CAs$AREA_NUMBE))


pdf('id-graph.pdf')
plot(CAs, border='darkgreen')
cr = coordinates(CAs)
text(cr, labels=ids)
#points(cr)
dev.off()
