library(randomForest)
library(C50)
library(class)
library(miscTools)
library(caret)
library(ROCR)
library(pROC)
library(rpart)
library(rpart.plot)
library(ggplot2)
library(RColorBrewer)
library(ModelMetrics)
library(robustbase)
library(kernlab)
library(e1071)
#install.packages("e1071")
setwd("C:/Users/gokul/Documents/Projects/Online_News_Popularity")
getwd()
onlinedata <- read.csv("OnlineNewsPopularity1.csv", header = T)
#summary(onlinedata)
onlinedata <-  onlinedata[!onlinedata$n_unique_tokens==701,]
#summary(onlinedata)
onlinedata <- subset(onlinedata, select = -c(url,timedelta,n_tokens_content, self_reference_max_shares, self_reference_avg_sharess, title_subjectivity, n_tokens_title,  num_keywords,num_self_hrefs, title_sentiment_polarity, min_positive_polarity, max_negative_polarity, n_non_stop_words, n_non_stop_unique_tokens, n_unique_tokens, LDA_01, LDA_02, LDA_03, LDA_04, LDA_00, shares,is_weekend, weekday_is_sunday, weekday_is_saturday, weekday_is_monday, weekday_is_tuesday, weekday_is_wednesday, weekday_is_thursday, weekday_is_friday, log_share))
#summary(onlinedata)
hist(onlinedata$Classify_2)
onlinedata$Classify_2 <- as.factor(onlinedata$Classify_2)


#Perform scaling of attributes
for(i in ncol(onlinedata)-3)
{
  onlinedata[,i] <- scale(onlinedata[,i], center = TRUE, scale = TRUE)
}
#summary(onlinedata)
str(onlinedata)
set.seed(100)
samplesets <- sample(2,nrow(onlinedata), replace = TRUE, prob = c(0.8,0.2))
onlineC50 <- C5.0(Classify_2 ~., onlinedata[samplesets == 1,], trials = 5)
#summary(onlineC50)
#testing/predicting
onlineC50.pred <- predict(onlineC50, onlinedata[samplesets == 2,], type = "class")
onlineC50.prob <- predict(onlineC50, onlinedata[samplesets == 2,], type = "prob")
#Confusion Matrix
table(onlineC50.pred, onlinedata[samplesets == 2,]$Classify_2)
caret::confusionMatrix(onlineC50.pred, onlinedata[samplesets == 2,]$Classify_2)

#Building ROC Curve
#onlineC50.roc <- roc(onlinedata[samplesets == 2,]$Classify_2,onlineC50.prob[,2], plot = TRUE)
onlineC50.roc <- multiclass.roc(onlinedata[samplesets == 2,]$Classify_2,as.numeric(onlineC50.pred), levels = 4)

plot(onlineC50.roc, xlab())
plot(onlineC50.roc)
onlineC50.roc <- multiclass.roc(onlinedata[samplesets == 2,]$Classify_2,onlineC50.prob[,2], levels = 4, plot = TRUE )






