source('Megamunge_jitter.R')
datadir <- 'TrainingData'

Megamunge(file.path(datadir, 'Chuck Grip.csv'))

dir(datadir, pattern = 'csv')

library(purrr)
pdf('test.pdf')
map(dir(datadir, pattern='csv', full.names = T), Megamunge)
dev.off()
