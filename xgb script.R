##############################################
#This is a starter script to use XGBoost in R for the competition
#Portions of the script are based off SRK's python starter:
#(https://www.kaggle.com/sudalairajkumar/two-sigma-connect-rental-listing-inquiries/xgb-starter-in-python)
#and Dan J's scripts: 
#https://www.kaggle.com/danjordan/two-sigma-connect-rental-listing-inquiries/how-to-correctly-load-data-into-r
#https://www.kaggle.com/danjordan/two-sigma-connect-rental-listing-inquiries/text-analysis-classifying-listings-with-features

#########################
##Load Data
# Load packages and data
library(lubridate)
library(dplyr)
library(jsonlite)
library(caret)
library(purrr)
library(xgboost)
library(MLmetrics)
library(tidytext)
library(reshape2)
seed = 1985
set.seed(seed)



train <- fromJSON('C:\\kaggle 2 sigma\\train.JSON')
test <- fromJSON('C:\\kaggle 2 sigma\\test.JSON')

# unlist every variable except `photos` and `features` and convert to tibble

#Train
vars <- setdiff(names(train), c("photos", "features"))
train <- map_at(train, vars, unlist) %>% tibble::as_tibble(.)
train_id <-train$listing_id

#Test
vars <- setdiff(names(test), c("photos", "features"))
test <- map_at(test, vars, unlist) %>% tibble::as_tibble(.)
test_id <-test$listing_id

#add feature for length of features
train$feature_count <- lengths(train$features)
test$feature_count <- lengths(test$features)
train$photo_count <- lengths(train$photos)
test$photo_count <- lengths(test$photos)

#Add fill for listings lacking any features
train[unlist(map(train$features,is_empty)),]$features = 'Nofeat'
test[unlist(map(test$features,is_empty)),]$features = 'Nofeat'


#add dummy interest level for test
test$interest_level <- 'none'


#combine train and test data
train_test <- rbind(train,test)

description <- train_test$description

#features to use
feat <- c("bathrooms","bedrooms","building_id", "created","latitude", "description",
          "listing_id","longitude","manager_id", "price", "features",
          "display_address", "street_address","feature_count","photo_count", "interest_level")

train_test = train_test[,names(train_test) %in% feat]

############################
##Process Word features

feature = data.frame(feature = tolower(unlist(train_test$features))) %>% # convert all features to lower case
  group_by(feature) %>%
  summarise(feature_count = n()) %>%
  arrange(desc(feature_count)) %>%
  filter(feature_count >= 50)

word_remove = c('allowed', 'building','center', 'space','2','2br','bldg','24',
                '3br','1','ft','3','7','1br','hour','bedrooms','bedroom','true',
                'stop','size','blk','4br','4','sq','0862','1.5','373','16','3rd','block',
                'st','01','bathrooms')

#create sparse matrix for word features
word_sparse<-train_test[,names(train_test) %in% c("features","listing_id")]






#Create word features
word_sparse <- word_sparse %>%
  filter(map(features, is_empty) != TRUE) %>%
  tidyr::unnest(features) %>%
  unnest_tokens(word, features)


data("stop_words")

#remove stop words and other words
word_sparse = word_sparse[!(word_sparse$word %in% stop_words$word),]
word_sparse = word_sparse[!(word_sparse$word %in% word_remove),]

#get most common features and use (in this case top 150)

stop_words



feature2 <- as.data.frame(c('fee', 'war', 'internet', 'swimming', 'dogs', 'cats', 'fitness', 'swimming', 'war', 'outdoor', 'roof'))

feature2$frequency <- 0

colnames(feature2)[1] <- 'feature'
colnames(feature2)[2] <- 'feature_count'


feature3 <- rbind(feature,feature2)

word_sparse = word_sparse[word_sparse$word %in% feature3$feature,]

str(word_sparse)

word_sparse$word = as.factor(word_sparse$word)
word_sparse<-dcast(word_sparse, listing_id ~ word,length, value.var = "word")

#merge word features back into main data frame
train_test<-merge(train_test,word_sparse, by = "listing_id", sort = FALSE,all.x=TRUE)

###############
##Non-word features

#convert building and manager id to integer
train_test$building_id<-as.integer(factor(train_test$building_id))
train_test$manager_id<-as.integer(factor(train_test$manager_id))

#lat long convert 0s

outliers_addrs <- train_test[train_test$longitude == 0 |
                               train_test$latitude == 0, ]$street_address
outliers_addrs

outliers_ny <- paste(outliers_addrs, ", new york")

outliers_addrs <- data.frame("street_address" = outliers_addrs)

library(jsonlite)
library(dplyr)
library(ggplot2)
library(magrittr)
library(ggmap)
library(knitr)

coords <- sapply(outliers_ny,
                 function(x) geocode(x, source = "google")) %>%
  t %>%
  data.frame %>%
  cbind(outliers_addrs, .)

rownames(coords) <- 1:nrow(coords)
# Display table
kable(coords)  



train_test[train_test$longitude == 0,]$longitude <- coords$lon
train_test[train_test$latitude == 0,]$latitude <- coords$lat

train_test$latitude <- as.numeric(train_test$latitude)
train_test$longitude <- as.numeric(train_test$longitude)

summary(train_test$longitude)

#convert street and display address to integer
train_test$display_address<-as.integer(factor(train_test$display_address))
train_test$street_address<-as.integer(factor(train_test$street_address))


#convert date
train_test$created<-ymd_hms(train_test$created)
train_test$month<- month(train_test$created)
train_test$day<- day(train_test$created)
train_test$hour<- hour(train_test$created)
train_test$created = NULL


##Length of description in words
train_test$description_len<-sapply(strsplit(train_test$description, "\\s+"), length)
train_test$description = NULL

#price to bedroom ratio
train_test$bed_price <- train_test$price/train_test$bedrooms
train_test[which(is.infinite(train_test$bed_price)),]$bed_price = train_test[which(is.infinite(train_test$bed_price)),]$price

#add sum of rooms and price per room
train_test$room_sum <- train_test$bedrooms + train_test$bathrooms
train_test$room_diff <- train_test$bedrooms - train_test$bathrooms
train_test$room_price <- train_test$price/train_test$room_sum
train_test$bed_ratio <- train_test$bedrooms/train_test$room_sum
train_test[which(is.infinite(train_test$room_price)),]$room_price = train_test[which(is.infinite(train_test$room_price)),]$price



#log transform features, these features aren't normally distributed
train_test$photo_count <- log(train_test$photo_count + 1)
train_test$feature_count <- log(train_test$feature_count + 1)
train_test$price <- log(train_test$price + 1)
train_test$room_price <- log(train_test$room_price + 1)
train_test$bed_price <- log(train_test$bed_price + 1)

train_test$features <- NULL

summa#split train test
train <- train_test[train_test$listing_id %in%train_id,]
test <- train_test[train_test$listing_id %in%test_id,]

#Convert labels to integers
train$interest_level<-as.integer(factor(train$interest_level))
y <- train$interest_level
y = y - 1
train$interest_level = NULL
test$interest_level = NULL

##################
#Parameters for XGB

xgb_params = list(
  colsample_bytree= 0.7,
  subsample = 0.7,
  eta = 0.1,
  objective= 'multi:softprob',
  max_depth= 4,
  min_child_weight= 1,
  eval_metric= "mlogloss",
  num_class = 3,
  seed = seed
)


summary(test)

test[is.na(test)] <- 0


#convert xgbmatrix
dtest <- xgb.DMatrix(data.matrix(test))

#create folds
kfolds<- 10
folds<-createFolds(y, k = kfolds, list = TRUE, returnTrain = FALSE)
fold <- as.numeric(unlist(folds[1]))

x_train<-train[-fold,] #Train set
x_val<-train[fold,] #Out of fold validation set

y_train<-y[-fold]
y_val<-y[fold]

x_train[is.na(x_train)] <- 0

str(x_train)

x_val[is.na(x_val)] <- 0


#convert to xgbmatrix
dtrain = xgb.DMatrix(as.matrix(x_train), label=y_train)
dval = xgb.DMatrix(as.matrix(x_val), label=y_val)

#perform training
gbdt = xgb.train(params = xgb_params,
                 data = dtrain,
                 nrounds =475,
                 watchlist = list(train = dtrain, val=dval),
                 print_every_n = 25,
                 early_stopping_rounds=50)

allpredictions =  (as.data.frame(matrix(predict(gbdt,dtest), nrow=dim(test), byrow=TRUE)))



######################
##Generate Submission
allpredictions = cbind (allpredictions, test$listing_id)
names(allpredictions)<-c("high","low","medium","listing_id")
allpredictions=allpredictions[,c(1,3,2,4)]
write.csv(allpredictions,paste0(Sys.Date(),"-LATLONGUPDATED-20Fold-Seed",seed,".csv"),row.names = FALSE)


####################################
###Generate Feature Importance Plot
imp <- xgb.importance(names(train),model = gbdt)

library(ggplot2)

xgb.ggplot.importance(imp)


--------------
  
  
###  improvements

hist(train_test$building_id)

library(stringr)
description[1]

#caps letters in description

train_test$descALLCAPS <- str_count(description, "\\b[A-Z]{2,}\\b")

## bag of words

class(description)

desc <- as.data.frame(description, stringsAsFactors = FALSE)
str(desc)

desc$description[2]


library(tm)

CorpusHeadline = Corpus(VectorSource(train_test2$description))

# You can go through all of the standard pre-processing steps like we did in Unit 5:

CorpusHeadline = tm_map(CorpusHeadline, tolower)

# Remember this extra line is needed after running the tolower step:

# CorpusHeadline = tm_map(CorpusHeadline, PlainTextDocument)

CorpusHeadline = tm_map(CorpusHeadline, removePunctuation)

CorpusHeadline = tm_map(CorpusHeadline, removeWords, stopwords("english"))

CorpusHeadline = tm_map(CorpusHeadline, stemDocument)

dtm = TermDocumentMatrix(CorpusHeadline)

sparse = removeSparseTerms(dtm, 0.95)

HeadlineWords = as.data.frame(as.matrix(sparse))

trans <- t(sparse)

HeadlineWords2 = as.data.frame(as.matrix(trans))

train_test$descfreq <- rowSums(HeadlineWords2)

# Let's make sure our variable names are okay for R:

colnames(HeadlineWords) = make.names(colnames(HeadlineWords))

HeadlineWords$count <- rowSums(HeadlineWords)

summary(HeadlineWords$count)

HeadlineWords <- HeadlineWords[order(HeadlineWords$count, decreasing = TRUE),]

top20desc <- HeadlineWords[,c(1,124012)]

library(data.table)
setDT(top20desc, keep.rownames = TRUE)[]



table(train_test$manhattan)

train2 <- fromJSON('C:\\kaggle 2 sigma\\train.JSON')
test2 <- fromJSON('C:\\kaggle 2 sigma\\test.JSON')



# unlist every variable except `photos` and `features` and convert to tibble

#Train
vars <- setdiff(names(train2), c("photos", "features"))
train2 <- map_at(train2, vars, unlist) %>% tibble::as_tibble(.)
train_id <-train2$listing_id

#Test
vars <- setdiff(names(test2), c("photos", "features"))
test2 <- map_at(test2, vars, unlist) %>% tibble::as_tibble(.)
test_id <-test2$listing_id

test2$interest_level <- 'none'

train_test2 <- rbind(train2,test2)

#clean out display address 

library(tm)

Corpusdisplay = Corpus(VectorSource(train_test2$display_address))

# You can go through all of the standard pre-processing steps like we did in Unit 5:

Corpusdisplay = tm_map(Corpusdisplay, tolower)

# Remember this extra line is needed after running the tolower step:

# CorpusHeadline = tm_map(CorpusHeadline, PlainTextDocument)

Corpusdisplay = tm_map(Corpusdisplay, removePunctuation)

Corpusdisplay = tm_map(Corpusdisplay, removeWords, stopwords("english"))

Corpusdisplay = tm_map(Corpusdisplay, removeWords,c('street', 'avenue', 'place', 'st', 'east', 'e','west', 'w', 'ave')   )

# CorpusHeadline = tm_map(CorpusHeadline, stemDocument)

dtm = TermDocumentMatrix(Corpusdisplay)

sparse = removeSparseTerms(dtm, 0.995)

trans = t(sparse)

Headlineaddress = as.data.frame(as.matrix(sparse))
Headlineaddress2 = as.data.frame(as.matrix(trans))

train_test <- cbind(train_test, Headlineaddress2)

train_test <- cbind(train_test, HeadlineWords2)

train_test$description <- train_test2$description




train_test$kitchen <- as.integer(as.logical(grepl('kitchen',ignore.case= TRUE  , train_test$description ) ))

train_test$manhattan <- as.integer(as.logical(grepl('manhattan',ignore.case= TRUE  , train_test$description ) ))

train_test$broker <- as.integer(as.logical(grepl('broker',ignore.case= TRUE  , train_test$description ) ))

train_test$stainless <- as.integer(as.logical(grepl('stainless',ignore.case= TRUE  , train_test$description ) ))

train_test$renovated <- as.integer(as.logical(grepl('renovated',ignore.case= TRUE  , train_test$description ) ))

train_test$call <- as.integer(as.logical(grepl('call',ignore.case= TRUE  , train_test$description ) ))

train_test$email <- as.integer(as.logical(grepl('email',ignore.case= TRUE  , train_test$description ) ))

train_test$park <- as.integer(as.logical(grepl('park',ignore.case= TRUE  , train_test$description ) ))


train_test$subway <- as.integer(as.logical(grepl('subway',ignore.case= TRUE  , train_test$description ) ))

train_test$excl <- length(gregexpr("[.?!*_]", train_test$description)[[1]])

str(train_test)

table(train_test$stainless)

#RF try

train$interest_level <- train2$interest_level

summary(train$interest_level)

train$interest_level <- as.factor(train$interest_level)

library(randomForest)

train[is.na(train)] <- 0

output.forest <- randomForest(interest_level ~ ., 
                              data = train,ntree=100)


predicted <- predict(output.forest,test,type="prob")


allpredictionsRF = as.data.frame(cbind (predicted, test$listing_id))
names(allpredictionsRF)<-c("high","low","medium","listing_id")
allpredictionsRF=allpredictionsRF[,c(1,3,2,4)]


write.csv(allpredictionsRF,'sample_submissionRF.csv',row.names=FALSE)

str(train_test2)

summary(train_test2$latitude)


###h20 package

library(data.table)
library(jsonlite)
library(h2o)
library(lubridate)
h2o.init(nthreads = -1, max_mem_size="8g")

train_h20 <- as.h2o(train, destination_frame = "train.hex")

test_h20 <- as.h2o(test, destination_frame = "test.hex")


varnames <- setdiff(colnames(train), "interest_level")
gbm1 <- h2o.gbm(x = varnames
                ,y = "interest_level"
                ,training_frame = train_h20
                ,distribution = "multinomial"
                ,model_id = "gbm1"
                #,nfolds = 5
                ,ntrees = 750
                ,learn_rate = 0.05
                ,max_depth = 7
                ,min_rows = 20
                ,sample_rate = 0.7
                ,col_sample_rate = 0.7
                #   ,stopping_rounds = 5
                #   ,stopping_metric = "logloss"
                #   ,stopping_tolerance = 0
                ,seed=321
)

preds <- as.data.table(h2o.predict(gbm1, test_h20))


testPreds <- data.table(listing_id = unlist(test$listing_id), preds[,.(high, medium, low)])
write.csv(testPreds, "submission_h20.csv", row.names = FALSE)


#SENTIMENT ANALYSIS

install.packages('syuzhet')
install.packages('DT')

library(syuzhet)
library(DT)

sentiment <- get_nrc_sentiment(description)
datatable(head(sentiment))

train_test$positive <- sentiment$positive

train_test$negative <- sentiment$negative

train_test$anger <- sentiment$anger

train_test$anticipation <- sentiment$anticipation

train_test$disguist <- sentiment$disgust 

train_test$fear <- sentiment$fear 

train_test$joy <- sentiment$joy

train_test$sadness <- sentiment$sadness

train_test$surprise <- sentiment$surprise

train_test$trust <- sentiment$trust

# exploration

manager_counts <- as.data.frame(table(train_test$manager_id))

colnames(manager_counts)[1] <- 'manager_id'

train_test3 <- merge(manager_counts, train_test, by= 'manager_id', all= FALSE)
