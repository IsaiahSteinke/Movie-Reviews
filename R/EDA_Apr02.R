# CPSC 685: Final Project
# Exploratory data analysis of the Stanford Large Movie Review dataset
# Author: Isaiah Steinke
# Last Modified: April 2, 2021
# Written, tested, and debugged in R v.4.0.3

# Required libraries
library(quanteda) # v.2.1.2
library(readtext) # v.0.80

# Read in all negative/positive reviews from our training set
neg.text <- readtext("train/neg/*.txt") # negative reviews
pos.text <- readtext("train/pos/*.txt") # positive reviews

# Replace <br /> characters in reviews with blank spaces since
# this is just a line break in HTML.
neg.text$text <- gsub("<br />", " ", neg.text$text)
pos.text$text <- gsub("<br />", " ", pos.text$text)

# Alternative method using tidyverse package "stringr"
library(stringr) # v.1.4.0
neg.text$text <- str_replace_all(neg.text$text, "<br />", " ")
pos.text$text <- str_replace_all(pos.text$text, "<br />", " ")

# Create corpora
neg.corpus <- corpus(neg.text) # create corpus of negative reviews
pos.corpus <- corpus(pos.text) # create corpus of positive reviews

# Take a look at quanteda's built-in stopword list
stopwords(language = "en")

# Surprisingly, it's a short list of only 175 words. It's possible
# that words such as "no," "nor," "not," and the contractions of "not"
# may be useful words to have when classifying reviews. Since the list is
# short, let's just customize it. I've removed "no," "nor," "not," and
# the contractions of "not." To be fair, we'll also remove the 
# corresponding "uncontracted" words, e.g., remove "are" and "aren't."
my.stopwords <- stopwords(language = "en")[c(1:37, 43:45, 49,
                                             53, 57:80, 99:164, 168:175)]

# It should be relatively easy to add words to this list in the future
# if we decide that other movie-specific terms should be removed.

# Let's go ahead and tokenize both corpora. I'll remove punctuation,
# numbers, and symbols.
neg.tokens <- tokens(neg.corpus, what = "word", remove_punct = TRUE,
                     remove_symbols = TRUE, remove_numbers = TRUE)
pos.tokens <- tokens(pos.corpus, what = "word", remove_punct = TRUE,
                     remove_symbols = TRUE, remove_numbers = TRUE)

# Since the dfm function can do stemming, remove stopwords, and
# lower-case all tokens, I'll just call that and create the
# document features (i.e., the DFs).
neg.dfm <- dfm(neg.tokens, tolower = TRUE, stem = TRUE,
               remove = my.stopwords)
pos.dfm <- dfm(pos.tokens, tolower = TRUE, stem = TRUE,
               remove = my.stopwords)

# These are huge matrices (~6M elements)!
dim(neg.dfm) # 47863 words/stems
dim(pos.dfm) # 49037 words/stems

# Let's look at the top 50 features in the positive/negative reviews.
topfeatures(neg.dfm, 50)
topfeatures(pos.dfm, 50)

# There's quite a bit of overlap. Some of the stopwords we retained
# from quanteda's list are there as well as some others that we
# thought might be similar between positive/negative reviews:
# film and movi, most predominantly. There's also a few others like
# play, show, charact, and perform.

# Let's try some wordclouds of the 250 most frequent terms.
textplot_wordcloud(neg.dfm, min_size = 1, max_size = 6,
                   max_words = 250, random_order = FALSE,
                   rotation = 0.25, 
                   color = RColorBrewer::brewer.pal(8,"Dark2"))
textplot_wordcloud(pos.dfm, min_size = 1, max_size = 6,
                   max_words = 250, random_order = FALSE,
                   rotation = 0.25, 
                   color = RColorBrewer::brewer.pal(8,"Dark2"))

# Looks like "no" and "not" are more prominent in the negative reviews but not
# in the positive reviews. Contractions with "not" appear in the positive
# reviews frequently. Maybe it would be better to only remove "no" and "not"
# from the stopword list.
my.stopwords <- stopwords(language = "en")[c(1:164, 166, 168:175)]
neg.dfm <- dfm(neg.tokens, tolower = TRUE, stem = TRUE,
               remove = my.stopwords)
pos.dfm <- dfm(pos.tokens, tolower = TRUE, stem = TRUE,
               remove = my.stopwords)
topfeatures(neg.dfm, 50)
topfeatures(pos.dfm, 50)
textplot_wordcloud(neg.dfm, min_size = 1, max_size = 5,
                   max_words = 250, random_order = FALSE,
                   rotation = 0.25, 
                   color = RColorBrewer::brewer.pal(8,"Dark2"))
textplot_wordcloud(pos.dfm, min_size = 1, max_size = 5,
                   max_words = 250, random_order = FALSE,
                   rotation = 0.25, 
                   color = RColorBrewer::brewer.pal(8,"Dark2"))

# Again, quite a bit of overlap in the top 50 features/DFs.
# "No" and "not" appear in both sets of reviews in the 
# top 50, oddly enough.

# Let's take a look at the TF-IDFs using quanteda's function.
neg.tfidf <- dfm_tfidf(neg.dfm, scheme_tf = "prop",
                       scheme_df = "inverse")
pos.tfidf <- dfm_tfidf(pos.dfm, scheme_tf = "prop",
                       scheme_df = "inverse")
topfeatures(neg.tfidf, 50)
topfeatures(pos.tfidf, 50)

# There looks to be less overlap with the TF-IDFs than the DFs. "No" and
# "not" still appear in the top 50 in both corpora. We might have a good
# case that some movie-specific terms should be removed, as "movi" and
# "film" are top 2 in both. Alternatively, we could let the algorithms
# for the models sort that out for us.