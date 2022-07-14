# CPSC 685: Final Project
# Model building for the Stanford Large Movie Review dataset
# Author: Isaiah Steinke
# Last Modified: April 23, 2021
# Written, tested, and debugged in R v.4.0.3

# Load required libraries.
library(quanteda) # v.2.1.2
library(readtext) # v.0.80
library(SnowballC) # v.0.7.0
library(caret) # v.6.0-86
library(rpart) # v.4.1-15
library(ranger) # v.0.12.1
library(e1071) # v.1.7-6

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

# Create dataframes for the models.
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
# SVM
# ===================================================================
# Create a validation set at 20% of the training set.
set.seed(7767)
indexes <- createDataPartition(df.train$class_var, times = 1,
                               p = 0.8, list = FALSE)
df.train.v2 <- df.train[indexes, ] # new training set
df.valid <- df.train[-indexes, ] # validation set

# ** Linear kernel **
# Linear kernel only needs the cost parameter tuned.
cost_vals <- c(0.01, 0.1, 1, 10, 100, 1000)

# Initialize vectors to store results.
Cost_param <- rep(0, length(cost_vals))
TP <- rep(0, length(cost_vals))
TN <- rep(0, length(cost_vals))
FP <- rep(0, length(cost_vals))
FN <- rep(0, length(cost_vals))
Acc <- rep(0, length(cost_vals))
Pre <- rep(0, length(cost_vals))
Rec <- rep(0, length(cost_vals))
F1 <- rep(0, length(cost_vals))

# Loop for running all combinations of hyperparameters.
start.time <- proc.time() # start timer
for (i in 1:length(cost_vals)){
  svm.model <- svm(class_var ~ ., data = df.train.v2, kernel = "linear",
                   cost = cost_vals[i], scale = FALSE)
  svm.preds <- predict(svm.model, newdata = df.valid)
  conf.matrix <- table(svm.preds, df.valid$class_var)
  Cost_param[i] <- cost_vals[i]
  TP[i] <- conf.matrix[2, 2]
  TN[i] <- conf.matrix[1, 1]
  FP[i] <- conf.matrix[2, 1]
  FN[i] <- conf.matrix[1, 2]
  Acc[i] <- (TP[i] + TN[i])/sum(conf.matrix)
  Pre[i] <- TP[i]/(TP[i] + FP[i])
  Rec[i] <- TP[i]/(TP[i] + FN[i])
  F1[i] <- (2*Pre[i]*Rec[i])/(Pre[i] + Rec[i])
}
proc.time() - start.time # calculate loop runtime
svm.results <- data.frame(Cost_param, TP, TN, FP, FN, Acc, Pre, Rec, F1)

# The runtime for this loop is ~40.76 min on my computer.
# The best results are when cost is 100.

# Output results.
write.csv(svm.results, "SVMLinResults.csv", row.names = FALSE)

# ** Radial kernel **
# Need to tune cost and gamma hyperparameters. We'll use our
# limited experience with the linear kernel to choose a small
# number of values for cost.
cost_vals <- c(1, 10, 100)
gamma_vals <- c(0.5, 1, 3, 5)

# Initialize vectors to store results.
Cost_param <- rep(0, length(cost_vals)*length(gamma_vals))
Gamma_param <- rep(0, length(cost_vals)*length(gamma_vals))
TP <- rep(0, length(cost_vals)*length(gamma_vals))
TN <- rep(0, length(cost_vals)*length(gamma_vals))
FP <- rep(0, length(cost_vals)*length(gamma_vals))
FN <- rep(0, length(cost_vals)*length(gamma_vals))
Acc <- rep(0, length(cost_vals)*length(gamma_vals))
Pre <- rep(0, length(cost_vals)*length(gamma_vals))
Rec <- rep(0, length(cost_vals)*length(gamma_vals))
F1 <- rep(0, length(cost_vals)*length(gamma_vals))

# Loop for running all combinations of hyperparameters.
start.time <- proc.time() # start timer
lc <- 1 # loop counter
for (i in 1:length(cost_vals)){
  for (j in 1: length(gamma_vals)){
    svm.model <- svm(class_var ~ ., data = df.train.v2, kernel = "radial",
                     cost = cost_vals[i], gamma = gamma_vals[j],
                     scale = FALSE)
    svm.preds <- predict(svm.model, newdata = df.valid)
    conf.matrix <- table(svm.preds, df.valid$class_var)
    Cost_param[lc] <- cost_vals[i]
    Gamma_param[lc] <- gamma_vals[j]
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
svm.results <- data.frame(Cost_param, Gamma_param, TP, TN, FP, FN,
                          Acc, Pre, Rec, F1)

# The runtime for the loop is ~63.6 min on my computer.
# The best results are when cost is 10 and gamma is 1.

# Output results.
write.csv(svm.results, "SVMRadResults.csv", row.names = FALSE)

# Clean up the environment.
rm(i, j, lc, cost_vals, gamma_vals, Cost_param, Gamma_param,
   start.time)

# ===================================================================
# Build best models for testing with IMDb dataset.
# ===================================================================
# Decision tree. Previous scripts built the decision tree using all
# of the training set data. For fair comparison, we'll train the
# decision tree on the same training set as the random forests and
# SVM models. This results in some slight changes in variable
# importance and the resulting tree.
dt.model <- rpart(class_var ~ ., data = df.train.v2, method = "class")

# Variable importance
# bad  worst   wast  great   poor     aw   bore   noth stupid 
#  36     18     15      9      6      5      5      4      3

# Plot tree.
plot(dt.model)
text(dt.model)

# Random forests: num.trees = 1000 and mtry = 5.
set.seed(305067)
rf.model <- ranger(class_var ~ ., data = df.train.v2,
                   num.trees = 1000, mtry = 5,
                   importance = "impurity", num.threads = 12)

# SVM: radial kernel, cost = 10, and gamma = 1.
svm.model <- svm(class_var ~ ., data = df.train.v2, kernel = "radial",
                 cost = 10, gamma = 1, scale = FALSE)

# ===================================================================
# Determine performance metrics on the test set for all models.
# ===================================================================
# Decision tree.
dt.preds <- predict(dt.model, newdata = df.test, type = "class")
conf.matrix <- table(dt.preds, df.test$class_var)
TP <- conf.matrix[2, 2]
TN <- conf.matrix[1, 1]
FP <- conf.matrix[2, 1]
FN <- conf.matrix[1, 2]
Acc <- (TP + TN)/sum(conf.matrix) # accuracy: 0.7378
Pre <- TP/(TP + FP) # precision: 0.7140
Rec <- TP/(TP + FN) # recall: 0.7934
F1 <- (2*Pre*Rec)/(Pre + Rec) # F1 score: 0.7516

# Random forests.
rf.preds <- predict(rf.model, data = df.test, type = "response")
conf.matrix <- table(rf.preds$predictions, df.test$class_var)
TP <- conf.matrix[2, 2]
TN <- conf.matrix[1, 1]
FP <- conf.matrix[2, 1]
FN <- conf.matrix[1, 2]
Acc <- (TP + TN)/sum(conf.matrix) # accuracy: 0.8527
Pre <- TP/(TP + FP) # precision: 0.8493
Rec <- TP/(TP + FN) # recall: 0.8577
F1 <- (2*Pre*Rec)/(Pre + Rec) # F1 score: 0.8534

# SVM
svm.preds <- predict(svm.model, newdata = df.test)
conf.matrix <- table(svm.preds, df.test$class_var)
TP <- conf.matrix[2, 2]
TN <- conf.matrix[1, 1]
FP <- conf.matrix[2, 1]
FN <- conf.matrix[1, 2]
Acc <- (TP + TN)/sum(conf.matrix) # accuracy: 0.8618
Pre <- TP/(TP + FP) # precision: 0.8462
Rec <- TP/(TP + FN) # recall: 0.8845
F1 <- (2*Pre*Rec)/(Pre + Rec) # F1 score: 0.8649

# ===================================================================
# IMDb Dataset: Unfiltered/Unsupervised
# ===================================================================
# Import data.
IMDb <- read.csv("final_reviews.csv", header = TRUE)

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

# Create dataframes for analysis with the models for each
# year, 2019 & 2020.
df.IMDb <- convert(tfidf.IMDb.top, to = "data.frame")
df.IMDb <- subset(df.IMDb, select = -doc_id)
names(df.IMDb) <- make.names(names(df.IMDb))
df.IMDb$year_var <- IMDb$year
df.IMDb2019 <- df.IMDb[which(df.IMDb$year_var == 2019), ]
df.IMDb2020 <- df.IMDb[which(df.IMDb$year_var == 2020), ]
df.IMDb2019 <- subset(df.IMDb2019, select = -year_var)
df.IMDb2020 <- subset(df.IMDb2020, select = -year_var)

# Clean up the environment.
rm(corpus.IMDb, dfm.IMDb, tfidf.IMDb, tfidf.IMDb.top)

# -------------------------------------------------------------------

# Check performance with the decision tree model.
# ** 2019 **
dt.preds <- predict(dt.model, newdata = df.IMDb2019, type = "class")

# Statistics for predicted negative and positive reviews
length(which(dt.preds == "neg")) # 439
length(which(dt.preds == "pos")) # 810
length(which(dt.preds == "neg"))/length(dt.preds) # 35.15%
length(which(dt.preds == "pos"))/length(dt.preds) # 64.85%

# ** 2020 **
dt.preds <- predict(dt.model, newdata = df.IMDb2020, type = "class")

# Statistics for predicted negative and positive reviews
length(which(dt.preds == "neg")) # 440
length(which(dt.preds == "pos")) # 809
length(which(dt.preds == "neg"))/length(dt.preds) # 35.23%
length(which(dt.preds == "pos"))/length(dt.preds) # 64.77%

# -------------------------------------------------------------------

# Check performance with the best random forests model.
# ** 2019 **
rf.preds <- predict(rf.model, data = df.IMDb2019, type = "response")

# Statistics for predicted negative and positive reviews
length(which(rf.preds$predictions == "neg")) # 541
length(which(rf.preds$predictions == "pos")) # 708
length(which(rf.preds$predictions == "neg"))/
  length(rf.preds$predictions) # 43.31%
length(which(rf.preds$predictions == "pos"))/
  length(rf.preds$predictions) # 56.69%

# ** 2020 **
rf.preds <- predict(rf.model, data = df.IMDb2020, type = "response")

# Statistics for predicted negative and positive reviews
length(which(rf.preds$predictions == "neg")) # 578
length(which(rf.preds$predictions == "pos")) # 671
length(which(rf.preds$predictions == "neg"))/
  length(rf.preds$predictions) # 46.28%
length(which(rf.preds$predictions == "pos"))/
  length(rf.preds$predictions) # 53.72%

# -------------------------------------------------------------------

# Check performance with the best SVM model.
# ** 2019 **
svm.preds <- predict(svm.model, newdata = df.IMDb2019)

# Statistics for predicted negative and positive reviews
length(which(svm.preds == "neg")) # 560
length(which(svm.preds == "pos")) # 689
length(which(svm.preds == "neg"))/length(svm.preds) # 44.84%
length(which(svm.preds == "pos"))/length(svm.preds) # 55.16%

# ** 2020 **
svm.preds <- predict(svm.model, newdata = df.IMDb2020)

# Statistics for predicted negative and positive reviews
length(which(svm.preds == "neg")) # 606
length(which(svm.preds == "pos")) # 643
length(which(svm.preds == "neg"))/length(svm.preds) # 48.52%
length(which(svm.preds == "pos"))/length(svm.preds) # 51.48%

# ===================================================================
# IMDb Dataset: Filtered
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

# Create dataframes for analysis with the models for each
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

# Statistics for predicted negative and positive reviews
length(which(dt.preds == "neg")) # 358
length(which(dt.preds == "pos")) # 705
length(which(dt.preds == "neg"))/length(dt.preds) # 33.68%
length(which(dt.preds == "pos"))/length(dt.preds) # 66.32%

# ** 2020 **
dt.preds <- predict(dt.model, newdata = df.IMDb2020, type = "class")

# Statistics for predicted negative and positive reviews
length(which(dt.preds == "neg")) # 355
length(which(dt.preds == "pos")) # 693
length(which(dt.preds == "neg"))/length(dt.preds) # 33.87%
length(which(dt.preds == "pos"))/length(dt.preds) # 66.13%

# -------------------------------------------------------------------

# Check performance with the best random forests model.
# ** 2019 **
rf.preds <- predict(rf.model, data = df.IMDb2019, type = "response")

# Statistics for predicted negative and positive reviews
length(which(rf.preds$predictions == "neg")) # 433
length(which(rf.preds$predictions == "pos")) # 630
length(which(rf.preds$predictions == "neg"))/
  length(rf.preds$predictions) # 40.73%
length(which(rf.preds$predictions == "pos"))/
  length(rf.preds$predictions) # 59.27%

# ** 2020 **
rf.preds <- predict(rf.model, data = df.IMDb2020, type = "response")

# Statistics for predicted negative and positive reviews
length(which(rf.preds$predictions == "neg")) # 462
length(which(rf.preds$predictions == "pos")) # 586
length(which(rf.preds$predictions == "neg"))/
  length(rf.preds$predictions) # 44.08%
length(which(rf.preds$predictions == "pos"))/
  length(rf.preds$predictions) # 55.92%

# -------------------------------------------------------------------

# Check performance with the best SVM model.
# ** 2019 **
svm.preds <- predict(svm.model, newdata = df.IMDb2019)

# Statistics for predicted negative and positive reviews
length(which(svm.preds == "neg")) # 455
length(which(svm.preds == "pos")) # 608
length(which(svm.preds == "neg"))/length(svm.preds) # 42.80%
length(which(svm.preds == "pos"))/length(svm.preds) # 57.20%

# ** 2020 **
svm.preds <- predict(svm.model, newdata = df.IMDb2020)

# Statistics for predicted negative and positive reviews
length(which(svm.preds == "neg")) # 484
length(which(svm.preds == "pos")) # 564
length(which(svm.preds == "neg"))/length(svm.preds) # 46.18%
length(which(svm.preds == "pos"))/length(svm.preds) # 53.82%

# ===================================================================
# Hypothesis Test Calculations
# ===================================================================
# We'll utilize the data for the SVM since that has the best
# performance.

# We'll utilize the test for the difference between two population
# proportions. For a two-sided test, i.e., H0: p1 - p2 = 0 and
# Ha: p1 - p2 != 0. For α = 0.05, the critical z value is 1.96.

# IMDb data, unfiltered/unsupervised.
p1 <- 0.4484 # proportion of negative reviews, 2019
n1 <- 1249 # sample size, 2019
p2 <- 0.4852 # proportion of negative reviews, 2020
n2 <- 1249 # sample size, 2020
z <- (p1 - p2)/sqrt((p1*(1 - p1)/n1) + (p2*(1 - p2)/n2)) # test stat
pnorm(z, 0, 1)*2 # p-value: 0.0651

# Since |z| = 1.845 < 1.96, we do not reject H0.

# IMDb data, filtered, predicted by SVM.
p1 <- 0.428 # proportion of negative reviews, 2019
n1 <- 1063 # sample size, 2019
p2 <- 0.4618 # proportion of negative reviews, 2020
n2 <- 1048 # sample size, 2020
z <- (p1 - p2)/sqrt((p1*(1 - p1)/n1) + (p2*(1 - p2)/n2)) # test stat
pnorm(z, 0, 1)*2 # p-value: 0.1180

# Since |z| = 1.563 < 1.96, we do not reject H0.

# IMDb data, filtered, actual values.
p1 <- 0.397 # proportion of negative reviews, 2019
n1 <- 1063 # sample size, 2019
p2 <- 0.4437 # proportion of negative reviews, 2020
n2 <- 1048 # sample size, 2020
z <- (p1 - p2)/sqrt((p1*(1 - p1)/n1) + (p2*(1 - p2)/n2)) # test stat
pnorm(z, 0, 1)*2 # p-value: 0.0298

# Since |z| = 2.176 > 1.96, we reject H0.