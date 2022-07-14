# Revised figures for the ICMLA paper
# Author: Isaiah Steinke
# Last Updated: July 26, 2021
# Written, tested, and debugged in R v. 4.1.0

# Load required libraries
library(ggplot2) # v. 3.3.5
library(ggpubr)  # v. 0.4.0

# Read in data
dt <- read.csv("DT_VarImp.csv", header = TRUE)
rf <- read.csv("RF_VarImp.csv", header = TRUE)

# Plots with values of variable importance
dt_plt <- ggplot(dt, aes(y = reorder(Word, VarImp), x = VarImp)) +
            geom_bar(stat = "identity", fill = "gray", color = "black") +
            theme_bw(base_size = 18) +
            xlab("Variable Importance") + ylab(NULL) +
            scale_x_continuous(limits = c(0, 1100),
                               breaks = c(0, 200, 400, 600, 800, 1000)) +
            geom_text(aes(label = as.integer(VarImp), hjust = -0.4))

rf_plt <- ggplot(rf, aes(y = reorder(Word, VarImp), x = VarImp)) +
            geom_bar(stat = "identity", fill = "gray", color = "black") +
            theme_bw(base_size = 18) +
            xlab("Variable Importance") + ylab(NULL) +
            scale_x_continuous(limits = c(0, 700),
                               breaks = c(0, 100, 200, 300, 400, 500, 600, 700)) +
            geom_text(aes(label = as.integer(VarImp), hjust = -0.4))

ggarrange(dt_plt, rf_plt, ncol = 1, nrow = 2,
          labels = c("(a)", "(b)"),
          label.x = 0.9, label.y = 0.22,
          font.label = list(size = 18))

# The values for variable importance seem unnecessary. So, create
# plots without these values.
dt_plt <- ggplot(dt, aes(y = reorder(Word, VarImp), x = VarImp)) +
  geom_bar(stat = "identity", fill = "gray", color = "black") +
  theme_bw(base_size = 18) +
  xlab("Variable Importance") + ylab(NULL) +
  scale_x_continuous(limits = c(0, 1100),
                     breaks = c(0, 200, 400, 600, 800, 1000))

rf_plt <- ggplot(rf, aes(y = reorder(Word, VarImp), x = VarImp)) +
  geom_bar(stat = "identity", fill = "gray", color = "black") +
  theme_bw(base_size = 18) +
  xlab("Variable Importance") + ylab(NULL) +
  scale_x_continuous(limits = c(0, 700),
                     breaks = c(0, 100, 200, 300, 400, 500, 600, 700))

ggarrange(dt_plt, rf_plt, ncol = 1, nrow = 2,
          labels = c("(a)", "(b)"),
          label.x = 0.9, label.y = 0.22,
          font.label = list(size = 18))