data <- read.table("C:/Users/gokul/Documents/Projects/Online_News_Popularity/OnlineNewsPopularity.csv", header = TRUE)
#C:\Users\gokul\Documents\Projects\Online_News_Popularity

onlinedata['url'] <- NULL
onlinedata['log_shares'] <-   sapply(onlinedata[,60],FUN = log10)