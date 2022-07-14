# CPSC 685: Final Project
# Initial test code for reading in data from the
# Stanford Large Movie Review dataset
# Author: Isaiah Steinke
# Last Modified: March 26, 2021
# Written, tested, and debugged in R v.4.0.3

# Required libraries
library(quanteda) # v.2.1.2
library(readtext) # v.0.80

# As a preliminary test, just read in all of the negative
# reviews (12,500 text files) from the training set.
neg.text <- readtext("train/neg/*.txt")
test.corpus <- corpus(neg.text) # create corpus
summary(test.corpus) # by default, only shows first 100 docs
test.tokens <- tokens(test.corpus, remove_punct = TRUE) # create tokens