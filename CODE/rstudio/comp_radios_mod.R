rm(list=ls()) 
setwd("~")

#install.packages(c('FactoMineR','factoextra'))
#install.packages('mccr')
#install.packages(c('MASS', 'class'))
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

train50 <- read.csv("train50_fin.csv", sep = ",")
train100 <- read.csv("train100_fin.csv", sep = ",")
train200 <- read.csv("train200_fin.csv", sep = ",")
train300 <- read.csv("train300_fin.csv", sep = ",")
train500 <- read.csv("train500_fin.csv", sep = ",")

test50 <- read.csv("test50_fin.csv", sep = ",")
test100 <- read.csv("test100_fin.csv", sep = ",")
test200 <- read.csv("test200_fin.csv", sep = ",")
test300 <- read.csv("test300_fin.csv", sep = ",")
test500 <- read.csv("test500_fin.csv", sep = ",")

#Info dataset
#str(train100)
#str(test100)

#Borramos las columnas innecesarias 
train50  <- dplyr::select(train50,  -c("X.1","X","Y"))
train100 <- dplyr::select(train100, -c("X.1","X","Y","ACC_TOTALES"))
train200 <- dplyr::select(train200, -c("X.1","X","Y"))
train300 <- dplyr::select(train300, -c("X.1","X","Y","ACC_TOTALES"))
train500 <- dplyr::select(train500, -c("X.1","X","Y","ACC_TOTALES"))

test50  <- dplyr::select(test50,  -c("X.1","X","Y"))
test100 <- dplyr::select(test100, -c("X.1","X","Y","ACC_TOTALES"))
test200 <- dplyr::select(test200, -c("X.1","X","Y"))
test300 <- dplyr::select(test300, -c("X.1","X","Y","ACC_TOTALES"))
test500 <- dplyr::select(test500, -c("X.1","X","Y","ACC_TOTALES"))


train50$SINIESTRO  <- factor(train50$SINIESTRO,  levels = c(0,1),labels = c("NO", "SI"))
train100$SINIESTRO <- factor(train100$SINIESTRO, levels = c(0,1),labels = c("NO", "SI"))
train200$SINIESTRO <- factor(train200$SINIESTRO, levels = c(0,1),labels = c("NO", "SI"))
train300$SINIESTRO <- factor(train300$SINIESTRO, levels = c(0,1),labels = c("NO", "SI"))
train500$SINIESTRO <- factor(train500$SINIESTRO, levels = c(0,1),labels = c("NO", "SI"))


test50$SINIESTRO  <- factor(test50$SINIESTRO,  levels=c(0,1),labels = c("NO","SI"))
test100$SINIESTRO <- factor(test100$SINIESTRO, levels=c(0,1),labels = c("NO","SI"))
test200$SINIESTRO <- factor(test200$SINIESTRO, levels=c(0,1),labels = c("NO","SI"))
test300$SINIESTRO <- factor(test300$SINIESTRO, levels=c(0,1),labels = c("NO","SI"))
test500$SINIESTRO <- factor(test500$SINIESTRO, levels=c(0,1),labels = c("NO","SI"))

round(prop.table(table(train50$SINIESTRO))*100, digits = 1)
round(prop.table(table(train100$SINIESTRO))*100, digits = 1)
round(prop.table(table(train200$SINIESTRO))*100, digits = 1)
round(prop.table(table(train300$SINIESTRO))*100, digits = 1)
round(prop.table(table(train500$SINIESTRO))*100, digits = 1)


table(train50$SINIESTRO)
table(train100$SINIESTRO)
table(train200$SINIESTRO)
table(train300$SINIESTRO)
table(train500$SINIESTRO)


#Un solo modelo para 50
set.seed(123)
mod_50_C50 <- C5.0(SINIESTRO~.,data = train50)
mod_50_C50
summary(mod_50_C50)

#Un solo modelo para 100
set.seed(123)
mod_100_C100 <- C5.0(SINIESTRO~.,data = train100)
mod_100_C100
summary(mod_100_C100)

#Un solo modelo para 200
set.seed(123)
mod_200_C200 <- C5.0(SINIESTRO~.,data = train200)
mod_200_C200
summary(mod_200_C200)

#Un solo modelo para 300
set.seed(123)
mod_300_C300 <- C5.0(SINIESTRO~.,train300)
mod_300_C300

#Un solo modelo para 500
set.seed(123)
mod_500_C500 <- C5.0(SINIESTRO~.,data = train500)
mod_500_C500
summary(mod_500_C500)









