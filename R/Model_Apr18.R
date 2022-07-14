# CPSC 685: Final Project
# Dataset preprocessing and initial model building
# for the Stanford Large Movie Review dataset
# Author: Isaiah Steinke
# Last Modified: April 18, 2021
# Written, tested, and debugged in R v.4.0.3

# Required libraries
library(quanteda) # v.2.1.2
library(readtext) # v.0.80
library(rpart) # v.4.1-15

# Training set: Import and processing
neg.train <- readtext("train/neg/*.txt") # negative reviews
neg.train$class_var <- as.factor("neg") # set class as negative
neg.train$text <- gsub("<br />", " ", neg.train$text) # replace HMTL
neg.train$text <- gsub("  ", " ", neg.train$text) # replace double spaces
pos.train <- readtext("train/pos/*.txt") # positive reviews
pos.train$class_var <- as.factor("pos") # set class as positive
pos.train$text <- gsub("<br />", " ", pos.train$text) # replace HMTL
pos.train$text <- gsub("  ", " ", pos.train$text) # replace double spaces

# Test set: Import and processing
neg.test <- readtext("test/neg/*.txt") # negative reviews
neg.test$class_var <- as.factor("neg") # set class as negative
neg.test$text <- gsub("<br />", " ", neg.test$text) # replace HMTL
neg.test$text <- gsub("  ", " ", neg.test$text) # replace double spaces
pos.test <- readtext("test/pos/*.txt") # positive reviews
pos.test$class_var <- as.factor("pos") # set class as positive
pos.test$text <- gsub("<br />", " ", pos.test$text) # replace HMTL
pos.test$text <- gsub("  ", " ", pos.test$text) # replace double spaces

# ===================================================================
# The following code will calculate the length of each review and add
# it to the various sets. It can be skipped since review lengths don't
# seem to vary much for positive/negative reviews.
neg.train$rev_len <- sapply(strsplit(neg.train$text, " "), length)
pos.train$rev_len <- sapply(strsplit(pos.train$text, " "), length)
neg.test$rev_len <- sapply(strsplit(neg.test$text, " "), length)
pos.test$rev_len <- sapply(strsplit(pos.test$text, " "), length)

# Alternative code snippet for calculating review lengths. This
# code seems to count a few words extra given my counts of a few
# reviews. I'm going to keep this here in case it's useful in the
# future.
lengths(gregexpr("\\W+", neg.train$text[1]))

# Plot histograms for comparison; given the long tails in the
# histograms, adjust for the same number of bins between 0 and
# 1000 words in the histograms.
par(mfrow = c(1, 2))
hist(neg.train$rev_len, breaks = 40, main = "Training Set: Negative Reviews",
     xlim = c(0, 1000), xlab = "Length of Reviews")
hist(pos.train$rev_len, breaks = 40, main = "Training Set: Positive Reviews",
     xlim = c(0, 1000), xlab = "Length of Reviews")
hist(neg.test$rev_len, breaks = 20, main = "Test Set: Negative Reviews",
     xlim = c(0, 1000), xlab = "Length of Reviews")
hist(pos.test$rev_len, breaks = 40, main = "Test Set: Positive Reviews",
     xlim = c(0, 1000), xlab = "Length of Reviews")
par(mfrow = c(1, 1))

# Distributions look similar between negative and positive reviews for
# both the training and test sets. So, review length may not be a 
# useful feature.
# ===================================================================

# Create corpora
corpus.train <- corpus(neg.train) + corpus(pos.train)
corpus.test <- corpus(neg.test) + corpus(pos.test)

# Tokenize corpora; remove punctuation, numbers, and symbols.
tokens.train <- tokens(corpus.train, what = "word", remove_punct = TRUE,
                     remove_symbols = TRUE, remove_numbers = TRUE)
tokens.test <- tokens(corpus.test, what = "word", remove_punct = TRUE,
                     remove_symbols = TRUE, remove_numbers = TRUE)

# Create document-feature matrices (DFMs); perform stemming, remove
# stopwords, and lower-case all tokens.
dfm.train <- dfm(tokens.train, tolower = TRUE, stem = TRUE,
               remove = stopwords(language = "en"))
dfm.test <- dfm(tokens.test, tolower = TRUE, stem = TRUE,
               remove = stopwords(language = "en"))

# Check top features in the DFMs.
topfeatures(dfm.train, 100)
topfeatures(dfm.test, 100)

# Get the names of the top features in the DFMs and reduce the
# training- and test-set DFMs to use the same features. This code
# should only be executed if you plan to build models using the
# top features according to document frequencies (DFs).
top.features <- names(topfeatures(dfm.train, 2500))
dfm.train.top <- dfm_select(dfm.train, pattern = top.features,
                            selection = "keep")
dfm.test.top <- dfm_select(dfm.test, pattern = top.features,
                           selection = "keep")

# Compute TF-IDFs.
tfidf.train <- dfm_tfidf(dfm.train, scheme_tf = "prop",
                       scheme_df = "inverse")
tfidf.test <- dfm_tfidf(dfm.test, scheme_tf = "prop",
                       scheme_df = "inverse")

# Check the top features according to TF-IDF.
topfeatures(tfidf.train, 100)
topfeatures(tfidf.test, 100)

# Get the names of the top features according to the TF-IDFs and
# reduce the training- and test-set TF-IDFs to use the same features.
# This code should only be executed if you plan to build models 
# using the top features according to TF-IDFs.
top.features <- names(topfeatures(tfidf.train, 2500))
tfidf.train.top <- dfm_select(tfidf.train, pattern = top.features,
                              selection = "keep")
tfidf.test.top <- dfm_select(tfidf.test, pattern = top.features,
                             selection = "keep")

# Create a decision-tree model using the TF-IDFs.
# Create dataframes.
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

# Build a decision tree model.
start.time <- proc.time() # start timer
dt.model <- rpart(class_var ~ ., data = df.train, method = "class")
proc.time() - start.time # calculate runtime

# Takes ~115 s to build the decision tree model. Let's take a look
# at the output.
summary(dt.model)

# Variable importance
# bad   wast  worst  great   bore   poor     aw   noth stupid 
# 35     17     15      8      6      5      5      4      3 

# Check the ranks of these variables
which(top.features == "bad") # 6
which(top.features == "wast") # 105
which(top.features == "worst") # 84
which(top.features == "great") # 15
which(top.features == "bore") # 114
which(top.features == "poor") # 158
which(top.features == "aw") # 202
which(top.features == "noth") # 78
which(top.features == "stupid") # 188

# These results suggest that we might not need to use the top
# 2500 features. Maybe 1000 is sufficient.

# Plot tree.
plot(dt.model)
text(dt.model)

# Finally, let's see how it does on the test set.
dt.preds <- predict(dt.model, newdata = df.test, type = "class")
conf.matrix <- table(dt.preds, df.test$class_var) # confusion matrix

# dt.preds  neg  pos
#      neg 8541 2579
#      pos 3959 9921

# Performance metrics
TP <- conf.matrix[2, 2] # no. of true positives
TN <- conf.matrix[1, 1] # no. of true negatives
FP <- conf.matrix[2, 1] # no. of false positives
FN <- conf.matrix[1, 2] # no. of false negatives
Acc <- (TP + TN)/sum(conf.matrix) # accuracy
Pre <- TP/(TP + FP) # precision
Rec <- TP/(TP + FN) # recall
F1 <- (2*Pre*Rec)/(Pre + Rec) # F1 score

# Acc = 0.7385, Pre = 0.7148, Rec = 0.7937, F1 = 0.7521
# This seems fairly good, and there's still plenty of room for
# improvement.