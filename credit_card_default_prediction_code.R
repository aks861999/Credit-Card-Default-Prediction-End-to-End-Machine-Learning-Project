# ============================================================================
# ML2 Project: Credit Card Default Prediction - Initial Data Analysis
# Dataset: Default of Credit Card Clients (Taiwan)
# Source: Kaggle/UCI Machine Learning Repository
# Date: December 3, 2025
# ============================================================================

# Load required libraries
library(tidyverse)   # For data manipulation and visualization
library(skimr)       # For comprehensive data summary
library(corrplot)    # For correlation visualization
library(gridExtra)   # For arranging multiple plots

# ============================================================================
# 1. LOAD THE DATA
# ============================================================================

# Load the dataset
# Note: Adjust the file path as needed
credit_data <- read.csv("D:/New_Downloads/UCI_Credit_Card.csv/UCI_Credit_Card.csv")

# Display first few rows
head(credit_data)

# Display structure of the dataset
str(credit_data)

# ============================================================================
# 2. BASIC DATA EXPLORATION
# ============================================================================

# Dataset dimensions
cat("Number of observations:", nrow(credit_data), "\n")
cat("Number of variables:", ncol(credit_data), "\n")

# Check for missing values
cat("\nMissing values per variable:\n")
colSums(is.na(credit_data))

# Summary statistics
summary(credit_data)

# More detailed summary using skimr
skim(credit_data)

# ============================================================================
# 3. TARGET VARIABLE ANALYSIS
# ============================================================================

# Check target variable distribution
cat("\nTarget Variable Distribution:\n")
table(credit_data$default.payment.next.month)
prop.table(table(credit_data$default.payment.next.month)) * 100

# Visualize target variable distribution
ggplot(credit_data, aes(x = factor(default.payment.next.month))) +
  geom_bar(fill = c("steelblue", "coral"), alpha = 0.8) +
  geom_text(stat = 'count', aes(label = after_stat(count)), vjust = -0.5) +
  labs(title = "Distribution of Credit Card Default",
       x = "Default Status (0 = No, 1 = Yes)",
       y = "Count") +
  theme_minimal()

# ============================================================================
# 4. VARIABLE TYPE CLASSIFICATION
# ============================================================================

# Convert categorical variables to factors
credit_data$SEX <- factor(credit_data$SEX, 
                          levels = c(1, 2), 
                          labels = c("Male", "Female"))

credit_data$EDUCATION <- factor(credit_data$EDUCATION,
                                levels = c(0, 1, 2, 3, 4, 5, 6),
                                labels = c("Unknown", "Graduate", "University", 
                                           "High School", "Others", "Unknown", "Unknown"))

credit_data$MARRIAGE <- factor(credit_data$MARRIAGE,
                               levels = c(0, 1, 2, 3),
                               labels = c("Unknown", "Married", "Single", "Others"))

credit_data$default.payment.next.month <- factor(credit_data$default.payment.next.month,
                                                 levels = c(0, 1),
                                                 labels = c("No Default", "Default"))

# ============================================================================
# 5. DESCRIPTIVE ANALYSIS OF KEY VARIABLES
# ============================================================================

# Age distribution
p1 <- ggplot(credit_data, aes(x = AGE)) +
  geom_histogram(bins = 30, fill = "steelblue", alpha = 0.7) +
  labs(title = "Age Distribution", x = "Age", y = "Frequency") +
  theme_minimal()

# Credit limit distribution
p2 <- ggplot(credit_data, aes(x = LIMIT_BAL)) +
  geom_histogram(bins = 50, fill = "coral", alpha = 0.7) +
  labs(title = "Credit Limit Distribution", x = "Credit Limit (NT$)", y = "Frequency") +
  theme_minimal()

# Gender distribution
p3 <- ggplot(credit_data, aes(x = SEX, fill = default.payment.next.month)) +
  geom_bar(position = "dodge", alpha = 0.8) +
  labs(title = "Default by Gender", x = "Gender", y = "Count") +
  scale_fill_manual(values = c("steelblue", "coral")) +
  theme_minimal()

# Education distribution
p4 <- ggplot(credit_data, aes(x = EDUCATION, fill = default.payment.next.month)) +
  geom_bar(position = "dodge", alpha = 0.8) +
  labs(title = "Default by Education", x = "Education Level", y = "Count") +
  scale_fill_manual(values = c("steelblue", "coral")) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# Arrange plots
grid.arrange(p1, p2, p3, p4, ncol = 2)

# ============================================================================
# 6. CORRELATION ANALYSIS (NUMERIC VARIABLES ONLY)
# ============================================================================

# Select numeric variables for correlation analysis
numeric_vars <- credit_data %>%
  select(LIMIT_BAL, AGE, starts_with("BILL_AMT"), starts_with("PAY_AMT"))

# Calculate correlation matrix
cor_matrix <- cor(numeric_vars, use = "complete.obs")

# Visualize correlation matrix
corrplot(cor_matrix, method = "color", type = "upper", 
         tl.cex = 0.7, tl.col = "black",
         title = "Correlation Matrix of Numeric Variables",
         mar = c(0,0,1,0))

# ============================================================================
# 7. DEFAULT ANALYSIS BY KEY FACTORS
# ============================================================================

# Default rate by age group
credit_data %>%
  mutate(age_group = cut(AGE, breaks = c(20, 30, 40, 50, 60, 100),
                         labels = c("20-30", "30-40", "40-50", "50-60", "60+"))) %>%
  group_by(age_group, default.payment.next.month) %>%
  summarise(count = n(), .groups = "drop") %>%
  group_by(age_group) %>%
  mutate(percentage = count / sum(count) * 100) %>%
  filter(default.payment.next.month == "Default") %>%
  ggplot(aes(x = age_group, y = percentage)) +
  geom_col(fill = "coral", alpha = 0.8) +
  geom_text(aes(label = sprintf("%.1f%%", percentage)), vjust = -0.5) +
  labs(title = "Default Rate by Age Group",
       x = "Age Group", y = "Default Rate (%)") +
  theme_minimal()

# ============================================================================
# 8. DATA QUALITY CHECK
# ============================================================================

cat("\n=== Data Quality Summary ===\n")
cat("Total observations:", nrow(credit_data), "\n")
cat("Complete cases:", sum(complete.cases(credit_data)), "\n")
cat("Missing values:", sum(is.na(credit_data)), "\n")
cat("Duplicate rows:", sum(duplicated(credit_data)), "\n")

# Check for outliers in credit limit
cat("\nCredit Limit Statistics:\n")
cat("Min:", min(credit_data$LIMIT_BAL), "\n")
cat("Max:", max(credit_data$LIMIT_BAL), "\n")
cat("Mean:", mean(credit_data$LIMIT_BAL), "\n")
cat("Median:", median(credit_data$LIMIT_BAL), "\n")
