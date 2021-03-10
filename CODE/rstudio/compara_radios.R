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

setwd("/Users/daniel.rodriguez/Documents/ACC/ACC_PROOF/ACC1/final_datasets")

train50  <- read.csv("train50_fin.csv", sep = ",")
train100 <- read.csv("train100_fin.csv", sep = ",")
train200 <- read.csv("train200_fin.csv", sep = ",")
train300 <- read.csv("train300_fin.csv", sep = ",")
train500 <- read.csv("train500_fin.csv", sep = ",")

train50  <- dplyr::select(train50,   -c("X.1","X","Y"))
train100 <- dplyr::select(train100,  -c("X.1","X","Y"))
train200 <- dplyr::select(train200,  -c("X.1","X","Y"))
train300 <- dplyr::select(train300,  -c("X.1","X","Y"))
train500 <- dplyr::select(train500,  -c("X.1","X","Y"))

train50$SINIESTRO  <- factor(train50$SINIESTRO,  levels = c(0,1),labels = c("NO", "SI"))
train100$SINIESTRO <- factor(train100$SINIESTRO, levels = c(0,1),labels = c("NO", "SI"))
train200$SINIESTRO <- factor(train200$SINIESTRO, levels = c(0,1),labels = c("NO", "SI"))
train300$SINIESTRO <- factor(train300$SINIESTRO, levels = c(0,1),labels = c("NO", "SI"))
train500$SINIESTRO <- factor(train500$SINIESTRO, levels = c(0,1),labels = c("NO", "SI"))

set.seed(123)
n50 <- dim(train50)[1]
## Vamos a generar el modelo dejando un grupo para prueba y las demas para entrenamiento
v.error.kg50 <- rep(0,10)
v.kap.50 <- rep(0,10)
#Hacemos validación cruzada 10 veces para ver que el error se estabiliza
for(i in 1:10){
  errori50 <- 0
  kapi50 <- 0
  #Esta instrucción genera los k = 10 grupos (Folds)
  grupos50 <- createFolds(1:n50, 10)
  #Este ciclo es el que hace la validacion cruzada con 10 grupos (Folds)
  for (k in 1:10){
    muestra50 <- grupos50[[k]]
    ttesting50 <- train50[muestra50,]
    tentrenamiento50 <- train50[-muestra50,]
    mod_50_C50 <- C5.0(SINIESTRO~.,data = tentrenamiento50, rules = TRUE)
    pred50 <- predict(mod_50_C50, ttesting50[,-68])
    #Matriz de Confusion
    MC50 <- table(ttesting50[,68], pred50)
    #Porcentaje de buena clasificación y de error
    acierto50 <- sum(diag(MC50))/sum(MC50)
    error50 <- 1-acierto50
    errori50 <- errori50 + error50
    kap50 <- round(confusionMatrix(table(pred50, ttesting50[,68]), positive="SI")$overall[2], 2) # report official Kappa
    kapi50 <- kapi50+kap50
  }
  v.error.kg50[i] <- errori50/10
  v.kap.50[i] <- kapi50/10
}


set.seed(123)
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
    mod_100_C50 <- C5.0(SINIESTRO~.,data = tentrenamiento100, rules = TRUE)
    pred100 <- predict(mod_100_C50, ttesting100[,-68])
    #Matriz de Confusion
    MC100 <- table(ttesting100[,68], pred100)
    #Porcentaje de buena clasificación y de error
    acierto100 <- sum(diag(MC100))/sum(MC100)
    error100 <- 1-acierto100
    errori100 <- errori100 + error100
    kap100 <- round(confusionMatrix(table(pred100, ttesting100[,68]), positive="SI")$overall[2], 2) # report official Kappa
    kapi100 <- kapi100+kap100
  }
  v.error.kg100[i] <- errori100/10
  v.kap.100[i] <- kapi100/10
}


set.seed(123)
n200 <- dim(train200)[1]
## Vamos a generar el modelo dejando un grupo para prueba y las demas para entrenamiento
v.error.kg200 <- rep(0,10)
v.kap.200 <- rep(0,10)
#Hacemos validación cruzada 10 veces para ver que el error se estabiliza
for(i in 1:10){
  errori200 <- 0
  kapi200 <- 0
  #Esta instrucción genera los k = 10 grupos (Folds)
  grupos200 <- createFolds(1:n200, 10)
  #Este ciclo es el que hace la validacion cruzada con 10 grupos (Folds)
  for (k in 1:10){
    muestra200 <- grupos200[[k]]
    ttesting200 <- train200[muestra200,]
    tentrenamiento200 <- train200[-muestra200,]
    mod_200_C50 <- C5.0(SINIESTRO~.,data = tentrenamiento200, rules = TRUE)
    pred200 <- predict(mod_200_C50, ttesting200[,-68])
    #Matriz de Confusion
    MC200 <- table(ttesting200[,68], pred200)
    #Porcentaje de buena clasificación y de error
    acierto200 <- sum(diag(MC200))/sum(MC200)
    error200 <- 1-acierto200
    errori200 <- errori200 + error200
    kap200 <- round(confusionMatrix(table(pred200, ttesting200[,68]), positive="SI")$overall[2], 2) # report official Kappa
    kapi200 <- kapi200+kap200
  }
  v.error.kg200[i] <- errori200/10
  v.kap.200[i] <- kapi200/10
}


set.seed(123)
n300 <- dim(train300)[1]
## Vamos a generar el modelo dejando un grupo para prueba y las demas para entrenamiento
v.error.kg300 <- rep(0,10)
v.kap.300 <- rep(0,10)
#Hacemos validación cruzada 10 veces para ver que el error se estabiliza
for(i in 1:10){
  errori300 <- 0
  kapi300 <- 0
  #Esta instrucción genera los k = 10 grupos (Folds)
  grupos300 <- createFolds(1:n300, 10)
  #Este ciclo es el que hace la validacion cruzada con 10 grupos (Folds)
  for (k in 1:10){
    muestra300 <- grupos300[[k]]
    ttesting300 <- train300[muestra300,]
    tentrenamiento300 <- train300[-muestra300,]
    mod_300_C50 <- C5.0(SINIESTRO~.,data = tentrenamiento300, rules = TRUE)
    pred300 <- predict(mod_300_C50, ttesting300[,-68])
    #Matriz de Confusion
    MC300 <- table(ttesting300[,68], pred300)
    #Porcentaje de buena clasificación y de error
    acierto300 <- sum(diag(MC300))/sum(MC300)
    error300 <- 1-acierto300
    errori300 <- errori300 + error300
    kap300 <- round(confusionMatrix(table(pred300, ttesting300[,68]), positive="SI")$overall[2], 2) # report official Kappa
    kapi300 <- kapi300+kap300
  }
  v.error.kg300[i] <- errori300/10
  v.kap.300[i] <- kapi300/10
}

set.seed(123)
n500 <- dim(train500)[1]
## Vamos a generar el modelo dejando un grupo para prueba y las demas para entrenamiento
v.error.kg500 <- rep(0,10)
v.kap.500 <- rep(0,10)
#Hacemos validación cruzada 10 veces para ver que el error se estabiliza
for(i in 1:10){
  errori500 <- 0
  kapi500 <- 0
  #Esta instrucción genera los k = 10 grupos (Folds)
  grupos500 <- createFolds(1:n500, 10)
  #Este ciclo es el que hace la validacion cruzada con 10 grupos (Folds)
  for (k in 1:10){
    muestra500 <- grupos500[[k]]
    ttesting500 <- train500[muestra500,]
    tentrenamiento500 <- train500[-muestra500,]
    mod_500_C50 <- C5.0(SINIESTRO~.,data = tentrenamiento500, rules = TRUE)
    pred500 <- predict(mod_500_C50, ttesting500[,-68])
    #Matriz de Confusion
    MC500 <- table(ttesting500[,68], pred500)
    #Porcentaje de buena clasificación y de error
    acierto500 <- sum(diag(MC500))/sum(MC500)
    error500 <- 1-acierto500
    errori500 <- errori500 + error500
    kap500 <- round(confusionMatrix(table(pred500, ttesting500[,68]), positive="SI")$overall[2], 2) # report official Kappa
    kapi500 <- kapi500+kap500
  }
  v.error.kg500[i] <- errori500/10
  v.kap.500[i] <- kapi500/10
}