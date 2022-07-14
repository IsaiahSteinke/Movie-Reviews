# Classification of Movie Reviews on IMDb
This repository contains the work I did for a group project during my Master's program. We utilized two datasets to classify movie reviews on IMDb as either positive (a rating of 7–10 stars) or negative (a rating of 1–4 stars). One of these sets is a "benchmark" dataset, and the other is a new dataset that we created by scraping reviews of movies released in 2019 and 2020 from IMDb. The new dataset was then analyzed to see if overall movie sentiment had changed owing to the COVID-19 pandemic.

## Data
The benchmark dataset is the [Large Movie Review Dataset](http://ai.stanford.edu/~amaas/data/sentiment/) compiled by researchers at Stanford (not uploaded to this repository because of its large size). This dataset was already split into training and test sets, each containing 25,000 reviews.

The other dataset was scraped from IMDb and is available in the `Data` subdirectory. This dataset contains 1,249 reviews from 2019 (pre-pandemic) and 1,249 from 2020.

## Model Building
We built machine learning models based on decision trees, random forests, and support vector machines (SVMs). The benchmark dataset was used to build the models after the data were preprocessed using standard natural language processing (NLP) techniques into term frequency-inverse document frequency (TF-IDF) features. Owing to the limitations of our available hardware, we only used the top 1,000 features with the highest TF-IDFs. Thus, models were built using a dataset of 25,000 reviews with 1,000 features.

## Main Results
Using the actual data scraped for movie reviews in 2019 and 2020, we find a statistically signficant difference in sentiment (movie reviews became more negative). However, if we use the classification results from our models, we are not able to detect a statistically significant difference for a change in sentiment. This could be due to the relatively low accuracy of our models (~86% for our SVM model).

Additional details about the processing of the data, model building, and our results are given in the paper that we submitted to the 2021 IEEE ICMLA conference in the `ICMLA` directory.
