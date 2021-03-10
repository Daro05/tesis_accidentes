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

setwd("/Users/daniel.rodriguez/Documents/ACC/ACC_PROOF/ACC1/final_datasets")

train100 <- read.csv("train100_fin.csv", sep = ",")
test100 <- read.csv("test100_fin.csv", sep = ",")

#Borramos las columnas innecesarias 

train100 <- dplyr::select(train100, -c("X.1","X","Y","ACC_TOTALES"))
test100 <- dplyr::select(test100, -c("X.1","X","Y","ACC_TOTALES"))

train100$SINIESTRO <- factor(train100$SINIESTRO, levels = c(0,1))
test100$SINIESTRO <- factor(test100$SINIESTRO, levels = c(0,1))

#Generamos el árbol de decisiòn
set.seed(123)
mod_100_C50 <- C5.0(SINIESTRO~.,data = train100, trials = 30)
mod_100_C50
summary(mod_100_C50)
pred100 <- predict(mod_100_C50, test100[,-68])
pred_p100 <- predict(mod_100_C50, test100[,-68], type = "prob")
rocCurve.tree <- roc(test100[,68], pred_p100[,2], levels = c(0,1), direction = "<")
r_auc <- auc(rocCurve.tree)
MC100 <- table(test100[,68], pred)
acierto100 <- (sum(diag(MC100)))/sum(MC100)
error100 <- 1-acierto100
round(confusionMatrix(table(pred100, ttesting100[,68]), positive="SI")$overall[2], 2)



set.seed(123)


ctrl = C5.0Control(bands = 10)
rules_100_C50 <- C5.0(SINIESTRO~.,data = train100, trials = 30,  rules = TRUE, control = ctrl)
rules_100_C50
summary(rules_100_C50)
