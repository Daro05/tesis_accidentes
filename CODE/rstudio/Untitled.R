rm(list=ls()) 
setwd("~")


library(ggplot2)
library(tidyverse)
library(dplyr)
library(lattice)
library(C50)
library(caret)
library(gmodels)
library(irr)
library(psych)
library(kknn)
library(randomForest)
library(e1071)
library(caTools)
library(mlbench)
library(ranger)
library(ROCR)
library(pROC)
library(kknn)
library(ada)
library(rpart)
library(MASS)
library(class)
library(party)
library(mccr)


setwd("/Users/daniel.rodriguez/Documents/ACC/ACC_PROOF/ACC1/final_datasets")

train100 <- read.csv("train100_fin.csv", sep = ",")
test100 <- read.csv("test100_fin.csv", sep = ",")

#Borramos las columnas innecesarias 

train100 <- dplyr::select(train100, -c("X.1","X","Y","ACC_TOTALES"))
test100 <- dplyr::select(test100, -c("X.1","X","Y","ACC_TOTALES"))


train100$SINIESTRO <- factor(train100$SINIESTRO, levels = c(0,1))
test100$SINIESTRO <- factor(test100$SINIESTRO, levels = c(0,1))

areaROC <- function(prediccion, real){
  pred <- prediction(prediccion, real)
  auc <- performance(pred, "auc")
  return(attributes(auc)$y.values[[1]])
}


muestra <- sample(1:94510, 18902)
ttesting <- train100[muestra,]
tentrenamiento <- train100[-muestra,]

modelo <- C5.0(SINIESTRO~.,data = tentrenamiento)
pred <- predict(modelo, ttesting[,-68], type = "prob")
#score <- attributes(pred)$probabilities[,1]
#clase <- ttesting[,68]

rocCurve.tree <- roc(ttesting[,68], pred[,2], levels = c(0,1), direction = "<")
plot(rocCurve.tree)
auc(rocCurve.tree)




n100 <- dim(train100)[1]
## Vamos a generar el modelo dejando un grupo para prueba y las demas para entrenamiento
v.mcc100 <- rep(0,10)
#Hacemos validación cruzada 10 veces para ver que el error se estabiliza
for(i in 1:10){
  mcci100 <- 0
  #Esta instrucción genera los k = 10 grupos (Folds)
  grupos100 <- createFolds(1:n100, 10)
  #Este ciclo es el que hace la validacion cruzada con 10 grupos (Folds)
  for (k in 1:10){
    muestra100 <- grupos100[[k]]
    ttesting100 <- train100[muestra100,]
    tentrenamiento100 <- train100[-muestra100,]
    mod_100_C50 <- C5.0(SINIESTRO~.,data = tentrenamiento100,  winnow = FALSE)
    pred100 <- predict(mod_100_C50, ttesting100[,-68])
    #rocCurve.tree <- roc(ttesting[,68], pred[,2], levels = c(0,1), direction = "<")
    #r_auc <- auc(rocCurve.tree)
    mcc1 <- mccr(pred100, ttesting100[,68])
    mcci100 <- mcci100+mcc1
   
  }
  v.mcc100[i] <- mcci100/10
}


muestra <- sample(1:94510, 18902)
ttesting <- train100[muestra,]
tentrenamiento <- train100[-muestra,]
mod_100_C50 <- C5.0(SINIESTRO~.,data = tentrenamiento100, trials = 20, winnow = FALSE)
pred100 <- predict(mod_100_C50, test100[,-68])
pred_p100 <- predict(mod_100_C50, test100[,-68], type = "prob")
rocCurve.tree <- roc(test100[,68], pred_p100[,2], levels = c(0,1), direction = "<")
r_auc <- auc(rocCurve.tree)
MC100 <- table(test100[,68], pred)
acierto100 <- (sum(diag(MC100)))/sum(MC100)
error100 <- 1-acierto100
round(confusionMatrix(table(pred100, ttesting100[,68]), positive="SI")$overall[2], 2)








library(caret)
library(C50)
library(mlbench)
fitControl <- trainControl(method = "repeatedcv",
                           number = 10,
                           repeats = 10, returnResamp="all")
# Choose the features and classes
#muestra100 <- sample(1:94510, 18902)
#y <- train100[muestra100,]
#x <- train100[-muestra100,]
#grid <- expand.grid( .winnow = c(TRUE,FALSE), .trials=c(1,5,10,15,20), .model="tree" )
mdl<- train(train100[,-68],train100[,68],trControl=fitControl,method="C5.0",verbose=FALSE)



n100 <- dim(train100)[1]
## Vamos a generar el modelo dejando un grupo para prueba y las demas para entrenamiento
v.error.kg100 <- rep(0,10)
v.kap.100 <- rep(0,10)
#Hacemos validación cruzada 10 veces para ver que el error se estabiliza
for(i in 1:10){
  errori100 <- 0
  kapi100 <- 0
  #Esta instrucción genera los k = 10 grupos (Folds)
  grupos100 <- createFolds(1:n100, 10)
  #Este ciclo es el que hace la validacion cruzada con 10 grupos (Folds)
  for (k in 1:10){
    muestra100 <- grupos100[[k]]
    ttesting100 <- train100[muestra100,]
    tentrenamiento100 <- train100[-muestra100,]
    mod_100_C50 <- C5.0(SINIESTRO~.,data = tentrenamiento100)
    pred100 <- predict(mod_100_C50, ttesting100[,-68])
    MC100 <- table(ttesting100[,68], pred100)
    #Porcentaje de buena clasificación y de error
    acierto100 <- sum(diag(MC100))/sum(MC100)
    error100 <- 1-acierto100
    errori100 <- errori100 + error100
    kap100 <- confusionMatrix(pred100, ttesting100[,68])$overall[2] # report official Kappa
    kapi100 <- kapi100+kap100
  }
  v.error.kg100[i] <- errori100/10
  v.kap.100[i] <- kapi100/10
}


n100 <- dim(train100)[1]
## Vamos a generar el modelo dejando un grupo para prueba y las demas para entrenamiento
v.error.kg100 <- rep(0,10)
v.kap.100 <- rep(0,10)
#Hacemos validación cruzada 10 veces para ver que el error se estabiliza
for(i in 1:10){
  errori100 <- 0
  kapi100 <- 0
  #Esta instrucción genera los k = 10 grupos (Folds)
  grupos100 <- createFolds(1:n100, 10)
  #Este ciclo es el que hace la validacion cruzada con 10 grupos (Folds)
  for (k in 1:10){
    muestra100 <- grupos100[[k]]
    ttesting100 <- train100[muestra100,]
    tentrenamiento100 <- train100[-muestra100,]
    mod_100_C50 <- C5.0(SINIESTRO~.,data = tentrenamiento100)
    pred100 <- predict(mod_100_C50, ttesting100[,-68])
    #Matriz de Confusion
    MC100 <- table(ttesting100[,68], pred100)
    #Porcentaje de buena clasificación y de error
    acierto100 <- sum(diag(MC100))/sum(MC100)
    error100 <- 1-acierto100
    errori100 <- errori100 + error100
    kap100 <- confusionMatrix(pred100, ttesting100[,68]), positive="1")$overall[2] # report official Kappa
    kapi100 <- kapi100+kap100
  }
  v.error.kg100[i] <- errori100/10
  v.kap.100[i] <- kapi100/10
}

set.seed(123)
muestra100 <- sample(1:94510, 18902)
ttesting100 <- train100[muestra100,]
tentrenamiento100 <- train100[-muestra100,]
mod_100_C50 <- C5.0(SINIESTRO~.,data = tentrenamiento100, trials = 60)
pred100 <- predict(mod_100_C50, test100[,-68])
pred_p100 <- predict(mod_100_C50, test100[,-68], type = "prob")
#Matriz de Confusion
MC100 <- table(test100[,68], pred100)
#Porcentaje de buena clasificación y de error
acierto100 <- sum(diag(MC100))/sum(MC100)
error100 <- 1-acierto100
kap100 <- confusionMatrix(pred100, test100[,68])$overall[2] # report official Kappa
mcc1 <- mccr(pred100, test100[,68])
rocCurve.tree <- roc(test100[,68], pred_p100[,2], levels = c(0,1), direction = "<")
r_auc <- auc(rocCurve.tree)


grid <- expand.grid(.winnow = c(TRUE,FALSE), 
                    .trials=c(10,50,80,100), .model="tree" )


C5.0Control(
  subset = TRUE,
  bands = 0,
  winnow = FALSE,
  noGlobalPruning = FALSE,
  CF = 0.25,
  minCases = 2,
  fuzzyThreshold = FALSE,
  sample = 0,
  seed = sample.int(4096, size = 1) - 1L,
  earlyStopping = TRUE,
  label = "outcome"
)

mdl<- train(x=train100[,-68],y=train100[,68],tuneGrid=grid,trControl=fitControl,method="C5.0",verbose=FALSE)
