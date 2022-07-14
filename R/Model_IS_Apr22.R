# CPSC 685: Final Project
# Model building for the Stanford Large Movie Review dataset
# Author: Isaiah Steinke
# Last Modified: April 22, 2021
# Written, tested, and debugged in R v.4.0.3

# Load required libraries.
library(quanteda) # v.2.1.2
library(readtext) # v.0.80
library(SnowballC) # v.0.7.0
library(rpart) # v.4.1-15
library(ranger) # v.0.12.1
library(caret) # v.6.0-86

# Training set: Import and processing.
neg.train <- readtext("train/neg/*.txt") # negative reviews
neg.train$class_var <- as.factor("neg") # set class as negative
neg.train$text <- gsub("<br />", " ", neg.train$text) # replace HMTL
neg.train$text <- gsub("  ", " ", neg.train$text) # replace double spaces
pos.train <- readtext("train/pos/*.txt") # positive reviews
pos.train$class_var <- as.factor("pos") # set class as positive
pos.train$text <- gsub("<br />", " ", pos.train$text) # replace HMTL
pos.train$text <- gsub("  ", " ", pos.train$text) # replace double spaces

# Test set: Import and processing.
neg.test <- readtext("test/neg/*.txt") # negative reviews
neg.test$class_var <- as.factor("neg") # set class as negative
neg.test$text <- gsub("<br />", " ", neg.test$text) # replace HMTL
neg.test$text <- gsub("  ", " ", neg.test$text) # replace double spaces
pos.test <- readtext("test/pos/*.txt") # positive reviews
pos.test$class_var <- as.factor("pos") # set class as positive
pos.test$text <- gsub("<br />", " ", pos.test$text) # replace HMTL
pos.test$text <- gsub("  ", " ", pos.test$text) # replace double spaces

# Create corpora.
corpus.train <- corpus(neg.train) + corpus(pos.train)
corpus.test <- corpus(neg.test) + corpus(pos.test)

# Tokenize corpora; remove punctuation, numbers, symbols, and stopwords;
# perform stemming.
tokens.train <- tokens(corpus.train, what = "word", remove_punct = TRUE,
                       remove_symbols = TRUE, remove_numbers = TRUE)
tokens.train <- tokens_select(tokens.train,
                              pattern = stopwords(language = "en"),
                              selection = "remove")
tokens.train <- tokens_wordstem(tokens.train, language = "en")
tokens.test <- tokens(corpus.test, what = "word", remove_punct = TRUE,
                      remove_symbols = TRUE, remove_numbers = TRUE)
tokens.test <- tokens_select(tokens.test,
                              pattern = stopwords(language = "en"),
                              selection = "remove")
tokens.test <- tokens_wordstem(tokens.test, language = "en")

# Create document-feature matrices (DFMs); lower-case all tokens.
dfm.train <- dfm(tokens.train, tolower = TRUE)
dfm.test <- dfm(tokens.test, tolower = TRUE)

# Compute TF-IDFs.
tfidf.train <- dfm_tfidf(dfm.train, scheme_tf = "prop",
                         scheme_df = "inverse")
tfidf.test <- dfm_tfidf(dfm.test, scheme_tf = "prop",
                        scheme_df = "inverse")

# Get the names of the top 1000 features according to the TF-IDFs and
# reduce the training- and test-set TF-IDFs to use the same features.
top.features <- names(topfeatures(tfidf.train, 1000))
tfidf.train.top <- dfm_match(tfidf.train, top.features)
tfidf.test.top <- dfm_match(tfidf.test, top.features)

# Create dataframes for the tree models.
df.train <- convert(tfidf.train.top, to = "data.frame")
df.test <- convert(tfidf.test.top, to = "data.frame")

# Append class_var to both dataframes; drop doc_id variable.
df.train$class_var <- corpus.train$class_var
df.train <- subset(df.train, select = -doc_id)
df.test$class_var <- corpus.test$class_var
df.test <- subset(df.test, select = -doc_id)

# Fix the column names so the model doesn't produce errors.
names(df.train) <- make.names(names(df.train))
names(df.test) <- make.names(names(df.test))

# Clean up the environment. We'll keep the initial data read in
# from all the text files.
rm(corpus.train, corpus.test, dfm.train, dfm.test, tfidf.train,
   tfidf.test, tfidf.train.top, tfidf.test.top)

# ===================================================================
# Decision Tree
# ===================================================================
start.time <- proc.time() # start timer
dt.model <- rpart(class_var ~ ., data = df.train, method = "class")
proc.time() - start.time # calculate runtime

# Takes 49.61 s to build the decision tree model. Let's take a look
# at the output.
summary(dt.model)

# Variable importance
# bad   wast  worst  great   bore   poor     aw stupid   noth 
#  35     17     15      9      6      5      5      4      4  

# There's some slight differences than before, most likely due to
# the changes in removing stopwords and stemming.

# Plot tree.
plot(dt.model)
text(dt.model)

# Finally, let's see how it does on the test set.
dt.preds <- predict(dt.model, newdata = df.test, type = "class")
conf.matrix <- table(dt.preds, df.test$class_var) # confusion matrix

# dt.preds  neg  pos
#      neg 8499 2564
#      pos 4001 9936

# Performance metrics
TP <- conf.matrix[2, 2] # true positives
TN <- conf.matrix[1, 1] # true negatives
FP <- conf.matrix[2, 1] # false positives
FN <- conf.matrix[1, 2] # false negatives
Acc <- (TP + TN)/sum(conf.matrix) # accuracy: 0.7374
Pre <- TP/(TP + FP) # precision: 0.7129
Rec <- TP/(TP + FN) # recall: 0.7949
F1 <- (2*Pre*Rec)/(Pre + Rec) # F1 score: 0.7517

# ===================================================================
# Random Forests
# ===================================================================
# We'll need to at least create a validation set to determine the
# best hyperparameter values. Let's split off 20% for the validation
# set.
set.seed(7767)
indexes <- createDataPartition(df.train$class_var, times = 1,
                                      p = 0.8, list = FALSE)
df.train.v2 <- df.train[indexes, ] # new training set
df.valid <- df.train[-indexes, ] # validation set

# Given the runtimes needed to generate a random forest, we'll
# tune the number of trees and the number of predictors (mtry)
# over a coarse grid of values.
ntrees <- c(100, 250, 500, 1000)
mtry.values <- c(5, 7, 10, 15, 20, 25, 50, 100, 200)

# Initialize vectors to store results.
No_Trees <- rep(0, length(ntrees)*length(mtry.values))
No_Preds <- rep(0, length(ntrees)*length(mtry.values))
TP <- rep(0, length(ntrees)*length(mtry.values)) # true positives
TN <- rep(0, length(ntrees)*length(mtry.values)) # true negatives
FP <- rep(0, length(ntrees)*length(mtry.values)) # false positives
FN <- rep(0, length(ntrees)*length(mtry.values)) # false negatives
Acc <- rep(0, length(ntrees)*length(mtry.values)) # accuracy
Pre <- rep(0, length(ntrees)*length(mtry.values)) # precision
Rec <- rep(0, length(ntrees)*length(mtry.values)) # recall
F1 <- rep(0, length(ntrees)*length(mtry.values)) # F1 score

# Loop for running all combinations of hyperparameters. Be sure to
# set num.threads in the ranger function appropriately before
# executing this code!
start.time <- proc.time() # start timer
lc <- 1 # loop counter
for (i in 1:length(ntrees)){
  for (j in 1: length(mtry.values)){
    set.seed(305067)
    rf.model <- ranger(class_var ~ ., data = df.train.v2,
                       num.trees = ntrees[i], mtry = mtry.values[j],
                       importance = "impurity", num.threads = 12)
    rf.preds <- predict(rf.model, data = df.valid, type = "response")
    conf.matrix <- table(rf.preds$predictions, df.valid$class_var)
    No_Trees[lc] <- ntrees[i]
    No_Preds[lc] <- mtry.values[j]
    TP[lc] <- conf.matrix[2, 2]
    TN[lc] <- conf.matrix[1, 1]
    FP[lc] <- conf.matrix[2, 1]
    FN[lc] <- conf.matrix[1, 2]
    Acc[lc] <- (TP[lc] + TN[lc])/sum(conf.matrix)
    Pre[lc] <- TP[lc]/(TP[lc] + FP[lc])
    Rec[lc] <- TP[lc]/(TP[lc] + FN[lc])
    F1[lc] <- (2*Pre[lc]*Rec[lc])/(Pre[lc] + Rec[lc])
    lc <- lc + 1
  }
}
proc.time() - start.time # calculate loop runtime
rf.results <- data.frame(No_Trees, No_Preds, TP, TN, FP, FN, Acc,
                         Pre, Rec, F1)

# The runtime for the loop is ~23.6 min on my computer.
# The best results are when No_Trees is 1000 and No_Preds is 5.

# Output results.
write.csv(rf.results, "RFParamResults.csv", row.names = FALSE)

# Build the random forests model with the best results using the
# training set and calculate its performance on the test set.
set.seed(305067)
rf.model <- ranger(class_var ~ ., data = df.train.v2,
                   num.trees = 1000, mtry = 5,
                   importance = "impurity", num.threads = 12)
rf.preds <- predict(rf.model, data = df.test, type = "response")
conf.matrix <- table(rf.preds$predictions, df.test$class_var)
TP <- conf.matrix[2, 2] # true positives
TN <- conf.matrix[1, 1] # true negatives
FP <- conf.matrix[2, 1] # false positives
FN <- conf.matrix[1, 2] # false negatives
Acc <- (TP + TN)/sum(conf.matrix) # accuracy: 0.8534
Pre <- TP/(TP + FP) # precision: 0.8478
Rec <- TP/(TP + FN) # recall: 0.8616
F1 <- (2*Pre*Rec)/(Pre + Rec) # F1 score: 0.8546

# This similar to the performance on the validation set, which
# was 83%-87% for all performance metrics.

# Look at variable importance.
rf.model$variable.importance # raw data
rf.vi <- rf.model$variable.importance # store in vector
rf.vi <- sort(rf.vi, decreasing = TRUE) # order values
barchart(rf.vi[25:1], xlab = "Variable Importance") # plot top 25 as bars
dotchart(rf.vi[25:1], xlab = "Variable Importance") # plot top 25 as dots

# Clean up the environment (delete loop variables).
rm(lc, i, j, No_Preds, No_Trees, mtry.values, ntrees, rf.vi)

# ===================================================================
# IMDb Dataset
# ===================================================================
# Import data.
IMDb <- read.csv("final_reviews.csv", header = TRUE)

# Preprocess data. For consistency with the Stanford set, on which
# our models are trained, we will remove reviews that have ratings
# of 5 or 6, which are neutral reviews. Also, remove reviews for
# which the rating is "NA."
IMDb.del <- c(which(is.na(IMDb$rating)), which(IMDb$rating == 5),
              which(IMDb$rating == 6))
IMDb <- IMDb[-IMDb.del, ]

# Add class variable.
IMDb$class_var <- ifelse(IMDb$rating <= 4, "neg", "pos")

# Create a corpus and compute DFs and TF-IDFs using same procedure
# as the Stanford set.
corpus.IMDb <- corpus(IMDb$review)
tokens.IMDb <- tokens(corpus.IMDb, what = "word", remove_punct = TRUE,
                      remove_symbols = TRUE, remove_numbers = TRUE)
tokens.IMDb <- tokens_select(tokens.IMDb,
                             pattern = stopwords(language = "en"),
                             selection = "remove")
tokens.IMDb <- tokens_wordstem(tokens.IMDb, language = "en")
dfm.IMDb <- dfm(tokens.IMDb, tolower = TRUE)
tfidf.IMDb <- dfm_tfidf(dfm.IMDb, scheme_tf = "prop",
                        scheme_df = "inverse")
tfidf.IMDb.top <- dfm_match(tfidf.IMDb, top.features)

# Create dataframes for analysis with the tree models for each
# year, 2019 & 2020.
df.IMDb <- convert(tfidf.IMDb.top, to = "data.frame")
df.IMDb$class_var <- as.factor(IMDb$class_var)
df.IMDb <- subset(df.IMDb, select = -doc_id)
names(df.IMDb) <- make.names(names(df.IMDb))
df.IMDb$year_var <- IMDb$year
df.IMDb2019 <- df.IMDb[which(df.IMDb$year_var == 2019), ]
df.IMDb2020 <- df.IMDb[which(df.IMDb$year_var == 2020), ]
df.IMDb2019 <- subset(df.IMDb2019, select = -year_var)
df.IMDb2020 <- subset(df.IMDb2020, select = -year_var)

# Clean up the environment.
rm(corpus.IMDb, dfm.IMDb, tfidf.IMDb, tfidf.IMDb.top)

# Statistics for actual negative and positive reviews
# ** 2019 **
length(which(df.IMDb2019$class_var == "neg")) # 422
length(which(df.IMDb2019$class_var == "pos")) # 641
length(which(df.IMDb2019$class_var == "neg"))/
  length(df.IMDb2019$class_var) # 39.70%
length(which(df.IMDb2019$class_var == "pos"))/
  length(df.IMDb2019$class_var) # 60.30%

# **2020 **
length(which(df.IMDb2020$class_var == "neg")) # 465
length(which(df.IMDb2020$class_var == "pos")) # 583
length(which(df.IMDb2020$class_var == "neg"))/
  length(df.IMDb2020$class_var) # 44.37%
length(which(df.IMDb2020$class_var == "pos"))/
  length(df.IMDb2020$class_var) # 55.63%

# -------------------------------------------------------------------

# Check performance with the decision tree model.
# ** 2019 **
dt.preds <- predict(dt.model, newdata = df.IMDb2019, type = "class")
conf.matrix <- table(dt.preds, df.IMDb2019$class_var)

# dt.preds neg pos
#      neg 238 118
#      pos 184 523

# Performance metrics
TP <- conf.matrix[2, 2] # true positives
TN <- conf.matrix[1, 1] # true negatives
FP <- conf.matrix[2, 1] # false positives
FN <- conf.matrix[1, 2] # false negatives
Acc <- (TP + TN)/sum(conf.matrix) # accuracy: 0.7159
Pre <- TP/(TP + FP) # precision: 0.7397
Rec <- TP/(TP + FN) # recall: 0.8159
F1 <- (2*Pre*Rec)/(Pre + Rec) # F1 score: 0.7760

# Statistics for predicted negative and positive reviews
length(which(dt.preds == "neg")) # 356
length(which(dt.preds == "pos")) # 707
length(which(dt.preds == "neg"))/length(dt.preds) # 33.49%
length(which(dt.preds == "pos"))/length(dt.preds) # 66.51%

# ** 2020 **
dt.preds <- predict(dt.model, newdata = df.IMDb2020, type = "class")
conf.matrix <- table(dt.preds, df.IMDb2020$class_var)

# dt.preds neg pos
#      neg 254 101
#      pos 211 482

# Performance metrics
TP <- conf.matrix[2, 2] # true positives
TN <- conf.matrix[1, 1] # true negatives
FP <- conf.matrix[2, 1] # false positives
FN <- conf.matrix[1, 2] # false negatives
Acc <- (TP + TN)/sum(conf.matrix) # accuracy: 0.7023
Pre <- TP/(TP + FP) # precision: 0.6955
Rec <- TP/(TP + FN) # recall: 0.8268
F1 <- (2*Pre*Rec)/(Pre + Rec) # F1 score: 0.7555

# Statistics for predicted negative and positive reviews
length(which(dt.preds == "neg")) # 355
length(which(dt.preds == "pos")) # 693
length(which(dt.preds == "neg"))/length(dt.preds) # 33.87%
length(which(dt.preds == "pos"))/length(dt.preds) # 66.13%

# -------------------------------------------------------------------

# Check performance with the best random forests model.
# ** 2019 **
rf.preds <- predict(rf.model, data = df.IMDb2019, type = "response")
conf.matrix <- table(rf.preds$predictions, df.IMDb2019$class_var)

#      neg pos
#  neg 346  92
#  pos  76 549

# Performance metrics
TP <- conf.matrix[2, 2] # true positives
TN <- conf.matrix[1, 1] # true negatives
FP <- conf.matrix[2, 1] # false positives
FN <- conf.matrix[1, 2] # false negatives
Acc <- (TP + TN)/sum(conf.matrix) # accuracy: 0.8420
Pre <- TP/(TP + FP) # precision: 0.8784
Rec <- TP/(TP + FN) # recall: 0.8565
F1 <- (2*Pre*Rec)/(Pre + Rec) # F1 score: 0.8673

# Statistics for predicted negative and positive reviews
length(which(rf.preds$predictions == "neg")) # 438
length(which(rf.preds$predictions == "pos")) # 625
length(which(rf.preds$predictions == "neg"))/
  length(rf.preds$predictions) # 41.20%
length(which(rf.preds$predictions == "pos"))/
  length(rf.preds$predictions) # 58.80%

# ** 2020 **
rf.preds <- predict(rf.model, data = df.IMDb2020, type = "response")
conf.matrix <- table(rf.preds$predictions, df.IMDb2020$class_var)

#      neg pos
#  neg 367 102
#  pos  98 481

# Performance metrics
TP <- conf.matrix[2, 2] # true positives
TN <- conf.matrix[1, 1] # true negatives
FP <- conf.matrix[2, 1] # false positives
FN <- conf.matrix[1, 2] # false negatives
Acc <- (TP + TN)/sum(conf.matrix) # accuracy: 0.8092
Pre <- TP/(TP + FP) # precision: 0.8307
Rec <- TP/(TP + FN) # recall: 0.8250
F1 <- (2*Pre*Rec)/(Pre + Rec) # F1 score: 0.8279

# Statistics for predicted negative and positive reviews
length(which(rf.preds$predictions == "neg")) # 469
length(which(rf.preds$predictions == "pos")) # 579
length(which(rf.preds$predictions == "neg"))/
  length(rf.preds$predictions) # 44.75%
length(which(rf.preds$predictions == "pos"))/
  length(rf.preds$predictions) # 55.25%