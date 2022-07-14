# Histograms of Scraped IMDb Reviews
# Author: Isaiah Steinke
# Last Modified: August 6, 2021
# Written, tested, and debugged in R v. 4.1.0

# Load required libraries
library(ggplot2) # v. 3.3.5
library(ggpubr)  # v. 0.4.0

# Import data
IMDb <- read.csv("final_reviews.csv", header = TRUE)

# Separate data by year
IMDb_2019 <- IMDb[which(IMDb$year == 2019), ]
IMDb_2020 <- IMDb[which(IMDb$year == 2020), ]

# Calculate how many movies have no rating
length(which(is.na(IMDb_2019$rating))) # 29 movies
length(which(is.na(IMDb_2020$rating))) # 28 movies

# Histograms
hist19 <- ggplot(IMDb_2019, aes(x = rating)) +
  geom_histogram(binwidth = 1, fill = "white", color = "black", size = 1.2) +
  xlab("Rating") + ylab("Frequency") +
  scale_x_continuous(breaks = c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)) +
  scale_y_continuous(limits = c(0, 250)) +
  theme_bw(base_size = 18) +
  theme(panel.grid.minor = element_blank(),
        panel.border = element_rect(size = 1.5))

hist20 <- ggplot(IMDb_2020, aes(x = rating)) +
  geom_histogram(binwidth = 1, fill = "white", color = "black", size = 1.2) +
  xlab("Rating") + ylab("Frequency") +
  scale_x_continuous(breaks = c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)) +
  scale_y_continuous(limits = c(0, 250)) +
  theme_bw(base_size = 18) +
  theme(panel.grid.minor = element_blank(),
        panel.border = element_rect(size = 1.5))

ggarrange(hist19, hist20, ncol = 1, nrow = 2,
          labels = c("(a) 2019", "(b) 2020"),
          label.x = 0.065, label.y = 0.975,
          font.label = list(size = 18))
