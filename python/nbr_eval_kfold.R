

library(MASS)
options(warn=-1)


y_train <- read.csv('Y_train.csv', header=FALSE)
y_test <- read.csv('Y_test.csv', header=FALSE)
f_train <- read.csv('f_train.csv')
f_test <- read.csv('f_test.csv')



dat <- data.frame(y_train, f_train)
mod <- glm.nb( data=dat )
ybar <- predict(mod, newdata=f_test, type=c('response') )

r = abs(ybar - y_test)
cat(mean(r$V1))



