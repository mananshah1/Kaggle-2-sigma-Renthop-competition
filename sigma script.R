#Reading in JSon file

install.packages(jsonlite)

library(rjson)

library(jsonlite)

install.packages('xgboost')

install.packages('MLmetrics')

install.packages('tidytext')

install.packages('caret')

library(xgboost)
library(MLmetrics)
library(tidytext)
library(reshape2)
seed = 1985
set.seed(seed)

install.packages('RJSONIO')

library(RJSONIO)

mydf <- fromJSON('C:\\kaggle2sigma\\train.JSON')

# unlist every variable except `photos` and `features` and convert to tibble
vars <- setdiff(names( mydf ), c("photos", "features"))


library(purrr)
library(dplyr)
library(caret)


data <- map_at(mydf, vars, unlist) %>% tibble::as_tibble(.)

str(data)

train <- data

summary(train)

class(data$features)

features2 <- rbind(data$features)

features3 <- t(features2)

features4 <- as.data.frame(features3)

gsub('c(', '', features4)

features_all <- lapply(as.matrix(features4), function(x)x)

features <- train$features
photos <- train$features

train$features <- NULL
train$photos <- NULL 

feat_unique <- paste(unique(features_all), collapse = ' ')

rm(features)
rm(photos)

ny_center <- geocode("new york", source = "google")

map <- get_googlemap(
  zoom = 12,
  # Use Alternate New York City Center Coords
  center = ny_center %>% as.numeric,
  maptype = "satellite",
  sensor = FALSE)
ggmap(map)

diffaddress <-  sigmadf[nchar((sigmadf$street_address) != nchar(sigmadf$display_address )),]

diffaddress <- train[nchar((train$street_address) != nchar(train$display_address )),]

substr(sigmadf$street_address,1,1)

install.packages("RgoogleMaps")

library(RgoogleMaps)

library(qmap)

install.packages('ggmap')

library(ggmap)

install.packages('qmap')

map <- get_map(location= 'newyork', zoom = 12, source= "google")

high <- subset(train, interest_level == 'high')

str(high)

summary(train)

class(high)



library(ggplot2)

library(ggmap)

summary(data$price)

data[is.na(data)] <- 0 

summary(data$latitude)

#Here is the code for the plots

ggmap(map) + geom_point(data = data, aes(x = as.numeric(longitude), y = as.numeric(latitude) , color= factor(data$interest_level) ), size=3, alpha=0.5 )
p<-  ggmap(map)+  geom_point(data = data, aes(x = as.numeric(longitude), y = as.numeric(latitude) , color= data$price )) +
scale_colour_gradient(limits = c(0, 6000))

p
