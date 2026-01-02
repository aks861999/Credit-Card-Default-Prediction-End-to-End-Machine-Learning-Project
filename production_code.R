# ==============================================================================
# ML2 Project: Credit Card Default Prediction
# ==============================================================================
# Course: Machine Learning 2
# Institution: Berliner Hochschule für Technik (BHT)
# Dataset: Default of Credit Card Clients (Taiwan, 2005)
# Methods: Random Forest (ML1) vs Support Vector Machine (ML2)
# Date: January 2026
# ==============================================================================

# ==============================================================================
# 0. INITIAL SETUP
# ==============================================================================

# Clear environment
rm(list = ls())

# Set seed for reproducibility
set.seed(42)

# Suppress warnings for cleaner output
options(warn = -1)

# ==============================================================================
# 1. LOAD REQUIRED LIBRARIES
# ==============================================================================

# Check and install required packages
required_packages <- c(
  "tidyverse",      # Data manipulation and visualization
  "caret",          # ML workflow and cross-validation
  "randomForest",   # Random Forest implementation
  "e1071",          # SVM implementation
  "DALEX",          # Model interpretability (XAI)
  "pROC",           # ROC curves and AUC
  "corrplot",       # Correlation matrix visualization
  "gridExtra",      # Multiple plot arrangements
  "scales",         # Scale functions for plots
  "reshape2",       # Data reshaping
  "ggthemes",       # Additional ggplot themes
  "RColorBrewer",   # Color palettes
  "viridis",        # Color scales
  "pdp"             # Partial dependence plots
)

# Install missing packages
new_packages <- required_packages[!(required_packages %in% installed.packages()[,"Package"])]
if(length(new_packages)) install.packages(new_packages, dependencies = TRUE)

# Load libraries
invisible(lapply(required_packages, library, character.only = TRUE))

cat("✓ All libraries loaded successfully\n\n")

# ==============================================================================
# 2. DATA LOADING AND INITIAL EXPLORATION
# ==============================================================================

cat("="*80, "\n")
cat("SECTION A: DATA LOADING AND DESCRIPTIVE ANALYSIS\n")
cat("="*80, "\n\n")

# Load dataset
# Note: Adjust file path as needed
data_path <- "D:/New_Downloads/UCI_Credit_Card.csv/UCI_Credit_Card.csv"


if (!file.exists(data_path)) {
  stop("Error: Data file not found. Please ensure UCI_Credit_Card.csv is in the working directory.")
}

credit_data <- read.csv(data_path, stringsAsFactors = FALSE)

cat("✓ Dataset loaded successfully\n")
cat(sprintf("  Dimensions: %d rows × %d columns\n\n", nrow(credit_data), ncol(credit_data)))

# Display structure
cat("Dataset Structure:\n")
str(credit_data)
cat("\n")

# First few rows
cat("First 5 rows:\n")
print(head(credit_data, 5))
cat("\n")

# ==============================================================================
# 3. DATA PREPROCESSING
# ==============================================================================

cat("Starting data preprocessing...\n")

# Create a copy for preprocessing
credit_clean <- credit_data

# Remove ID column (not predictive)
if ("ID" %in% colnames(credit_clean)) {
  credit_clean <- credit_clean %>% select(-ID)
  cat("✓ ID column removed\n")
}

# Check for missing values
missing_count <- sum(is.na(credit_clean))
cat(sprintf("✓ Missing values: %d (%.2f%%)\n", missing_count, missing_count/prod(dim(credit_clean))*100))

# Check for duplicates
duplicate_count <- sum(duplicated(credit_clean))
cat(sprintf("✓ Duplicate rows: %d\n", duplicate_count))

# Convert target variable to factor
credit_clean$default.payment.next.month <- factor(
  credit_clean$default.payment.next.month,
  levels = c(0, 1),
  labels = c("No_Default", "Default")
)

# Convert categorical variables to factors
credit_clean$SEX <- factor(
  credit_clean$SEX,
  levels = c(1, 2),
  labels = c("Male", "Female")
)

credit_clean$EDUCATION <- factor(
  credit_clean$EDUCATION,
  levels = c(0, 1, 2, 3, 4, 5, 6),
  labels = c("Unknown", "Graduate", "University", "HighSchool", "Others", "Unknown", "Unknown")
)

# Combine unknown education levels
levels(credit_clean$EDUCATION)[levels(credit_clean$EDUCATION) == "Unknown"] <- "Others"
credit_clean$EDUCATION <- droplevels(credit_clean$EDUCATION)

credit_clean$MARRIAGE <- factor(
  credit_clean$MARRIAGE,
  levels = c(0, 1, 2, 3),
  labels = c("Others", "Married", "Single", "Others")
)

# Combine marriage "Others" levels
credit_clean$MARRIAGE <- fct_collapse(credit_clean$MARRIAGE, Others = c("Others"))

cat("✓ Categorical variables converted to factors\n")

# Summary statistics
cat("\n")
cat("Summary Statistics:\n")
print(summary(credit_clean))
cat("\n")

# ==============================================================================
# 4. EXPLORATORY DATA ANALYSIS (EDA)
# ==============================================================================

cat("="*80, "\n")
cat("PERFORMING EXPLORATORY DATA ANALYSIS\n")
cat("="*80, "\n\n")

# Create output directory for plots
dir.create("plots", showWarnings = FALSE)

# 4.1 Target Variable Distribution
cat("Target Variable Distribution:\n")
target_table <- table(credit_clean$default.payment.next.month)
target_prop <- prop.table(target_table) * 100
print(target_table)
cat(sprintf("\nNo Default: %.2f%%\nDefault: %.2f%%\n\n", target_prop[1], target_prop[2]))

# Plot 1: Target variable distribution
p1 <- ggplot(credit_clean, aes(x = default.payment.next.month, fill = default.payment.next.month)) +
  geom_bar(alpha = 0.8) +
  geom_text(stat = 'count', aes(label = after_stat(count)), vjust = -0.5, size = 5) +
  scale_fill_manual(values = c("#4575b4", "#d73027")) +
  labs(
    title = "Distribution of Credit Card Default",
    subtitle = sprintf("No Default: %.1f%% | Default: %.1f%%", target_prop[1], target_prop[2]),
    x = "Default Status",
    y = "Count",
    fill = "Status"
  ) +
  theme_minimal(base_size = 12) +
  theme(legend.position = "bottom")

ggsave("plots/01_target_distribution.png", p1, width = 8, height = 6, dpi = 300)

# 4.2 Numeric Variables Distribution
numeric_vars <- credit_clean %>%
  select(LIMIT_BAL, AGE, starts_with("BILL_AMT"), starts_with("PAY_AMT"))

# Plot 2: Age distribution
p2 <- ggplot(credit_clean, aes(x = AGE)) +
  geom_histogram(bins = 30, fill = "#4575b4", alpha = 0.7, color = "white") +
  geom_vline(aes(xintercept = mean(AGE)), color = "red", linetype = "dashed", size = 1) +
  labs(
    title = "Age Distribution",
    subtitle = sprintf("Mean: %.1f years | Median: %.0f years", mean(credit_clean$AGE), median(credit_clean$AGE)),
    x = "Age (years)",
    y = "Frequency"
  ) +
  theme_minimal(base_size = 12)

ggsave("plots/02_age_distribution.png", p2, width = 8, height = 6, dpi = 300)

# Plot 3: Credit limit distribution
p3 <- ggplot(credit_clean, aes(x = LIMIT_BAL)) +
  geom_histogram(bins = 50, fill = "#d73027", alpha = 0.7, color = "white") +
  geom_vline(aes(xintercept = median(LIMIT_BAL)), color = "blue", linetype = "dashed", size = 1) +
  scale_x_continuous(labels = scales::comma) +
  labs(
    title = "Credit Limit Distribution",
    subtitle = sprintf("Median: NT$ %s", format(median(credit_clean$LIMIT_BAL), big.mark = ",")),
    x = "Credit Limit (NT$)",
    y = "Frequency"
  ) +
  theme_minimal(base_size = 12)

ggsave("plots/03_credit_limit_distribution.png", p3, width = 8, height = 6, dpi = 300)

# 4.3 Categorical Variables by Default Status
# Plot 4: Gender vs Default
p4 <- credit_clean %>%
  group_by(SEX, default.payment.next.month) %>%
  summarise(count = n(), .groups = "drop") %>%
  group_by(SEX) %>%
  mutate(percentage = count / sum(count) * 100) %>%
  ggplot(aes(x = SEX, y = percentage, fill = default.payment.next.month)) +
  geom_col(position = "dodge", alpha = 0.8) +
  geom_text(aes(label = sprintf("%.1f%%", percentage)), 
            position = position_dodge(width = 0.9), vjust = -0.5, size = 3.5) +
  scale_fill_manual(values = c("#4575b4", "#d73027")) +
  labs(
    title = "Default Rate by Gender",
    x = "Gender",
    y = "Percentage (%)",
    fill = "Status"
  ) +
  theme_minimal(base_size = 12) +
  theme(legend.position = "bottom")

ggsave("plots/04_gender_default.png", p4, width = 8, height = 6, dpi = 300)

# Plot 5: Education vs Default
p5 <- credit_clean %>%
  group_by(EDUCATION, default.payment.next.month) %>%
  summarise(count = n(), .groups = "drop") %>%
  group_by(EDUCATION) %>%
  mutate(percentage = count / sum(count) * 100) %>%
  ggplot(aes(x = EDUCATION, y = percentage, fill = default.payment.next.month)) +
  geom_col(position = "dodge", alpha = 0.8) +
  geom_text(aes(label = sprintf("%.1f%%", percentage)), 
            position = position_dodge(width = 0.9), vjust = -0.5, size = 3) +
  scale_fill_manual(values = c("#4575b4", "#d73027")) +
  labs(
    title = "Default Rate by Education Level",
    x = "Education",
    y = "Percentage (%)",
    fill = "Status"
  ) +
  theme_minimal(base_size = 12) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        legend.position = "bottom")

ggsave("plots/05_education_default.png", p5, width = 8, height = 6, dpi = 300)

# Plot 6: Marriage vs Default
p6 <- credit_clean %>%
  group_by(MARRIAGE, default.payment.next.month) %>%
  summarise(count = n(), .groups = "drop") %>%
  group_by(MARRIAGE) %>%
  mutate(percentage = count / sum(count) * 100) %>%
  ggplot(aes(x = MARRIAGE, y = percentage, fill = default.payment.next.month)) +
  geom_col(position = "dodge", alpha = 0.8) +
  geom_text(aes(label = sprintf("%.1f%%", percentage)), 
            position = position_dodge(width = 0.9), vjust = -0.5, size = 3.5) +
  scale_fill_manual(values = c("#4575b4", "#d73027")) +
  labs(
    title = "Default Rate by Marital Status",
    x = "Marital Status",
    y = "Percentage (%)",
    fill = "Status"
  ) +
  theme_minimal(base_size = 12) +
  theme(legend.position = "bottom")

ggsave("plots/06_marriage_default.png", p6, width = 8, height = 6, dpi = 300)

# 4.4 Correlation Analysis
cat("Computing correlation matrix for numeric variables...\n")

# Select only numeric columns for correlation
numeric_cols <- credit_clean %>%
  select(LIMIT_BAL, AGE, starts_with("PAY_"), starts_with("BILL_AMT"), starts_with("PAY_AMT")) %>%
  select(where(is.numeric))

cor_matrix <- cor(numeric_cols, use = "complete.obs")

# Plot 7: Correlation heatmap
png("plots/07_correlation_matrix.png", width = 12, height = 10, units = "in", res = 300)
corrplot(cor_matrix, 
         method = "color", 
         type = "upper", 
         tl.cex = 0.8, 
         tl.col = "black",
         title = "Correlation Matrix of Numeric Variables",
         mar = c(0,0,2,0),
         col = colorRampPalette(c("#4575b4", "white", "#d73027"))(200))
dev.off()

cat("✓ EDA completed and plots saved to 'plots/' directory\n\n")

# ==============================================================================
# 5. DATA SPLITTING (60% Train / 20% Validation / 20% Test)
# ==============================================================================

cat("="*80, "\n")
cat("DATA SPLITTING\n")
cat("="*80, "\n\n")

# Create stratified splits
set.seed(42)

# First split: 80% (train+val) and 20% test
trainval_index <- createDataPartition(credit_clean$default.payment.next.month, 
                                      p = 0.80, 
                                      list = FALSE)
trainval_data <- credit_clean[trainval_index, ]
test_data <- credit_clean[-trainval_index, ]

# Second split: Split trainval into 75% train and 25% validation (giving 60-20-20)
train_index <- createDataPartition(trainval_data$default.payment.next.month, 
                                   p = 0.75, 
                                   list = FALSE)
train_data <- trainval_data[train_index, ]
val_data <- trainval_data[-train_index, ]

cat(sprintf("Training set:   %d samples (%.1f%%)\n", 
            nrow(train_data), nrow(train_data)/nrow(credit_clean)*100))
cat(sprintf("Validation set: %d samples (%.1f%%)\n", 
            nrow(val_data), nrow(val_data)/nrow(credit_clean)*100))
cat(sprintf("Test set:       %d samples (%.1f%%)\n\n", 
            nrow(test_data), nrow(test_data)/nrow(credit_clean)*100))

# Verify class distribution in each set
cat("Class distribution in splits:\n")
cat("Training:  ", sprintf("%.2f%% / %.2f%%", 
                           prop.table(table(train_data$default.payment.next.month))[1]*100,
                           prop.table(table(train_data$default.payment.next.month))[2]*100), "\n")
cat("Validation:", sprintf("%.2f%% / %.2f%%", 
                           prop.table(table(val_data$default.payment.next.month))[1]*100,
                           prop.table(table(val_data$default.payment.next.month))[2]*100), "\n")
cat("Test:      ", sprintf("%.2f%% / %.2f%%", 
                           prop.table(table(test_data$default.payment.next.month))[1]*100,
                           prop.table(table(test_data$default.payment.next.month))[2]*100), "\n\n")

# ==============================================================================
# 6. RANDOM FOREST MODEL (ML1 METHOD)
# ==============================================================================

cat("="*80, "\n")
cat("SECTION C: RANDOM FOREST MODEL TRAINING (ML1)\n")
cat("="*80, "\n\n")

# 6.1 Hyperparameter Tuning
cat("Starting Random Forest hyperparameter tuning...\n")

# Calculate class weights to handle imbalance
class_weights <- table(train_data$default.payment.next.month)
class_weights <- max(class_weights) / class_weights
cat("Class weights:", paste(names(class_weights), "=", round(class_weights, 2), collapse = ", "), "\n\n")

# Define hyperparameter grid
rf_grid <- expand.grid(
  ntree = c(300, 500),
  mtry = c(4, 6, 8),
  nodesize = c(5, 10)
)

# Initialize results storage
rf_tuning_results <- data.frame()

# Manual grid search with progress tracking
cat("Testing", nrow(rf_grid), "hyperparameter combinations...\n")

start_time <- Sys.time()

for (i in 1:nrow(rf_grid)) {
  cat(sprintf("  [%d/%d] ntree=%d, mtry=%d, nodesize=%d... ", 
              i, nrow(rf_grid), rf_grid$ntree[i], rf_grid$mtry[i], rf_grid$nodesize[i]))
  
  rf_temp <- randomForest(
    default.payment.next.month ~ .,
    data = train_data,
    ntree = rf_grid$ntree[i],
    mtry = rf_grid$mtry[i],
    nodesize = rf_grid$nodesize[i],
    classwt = class_weights,
    importance = TRUE
  )
  
  # Predict on validation set
  val_pred <- predict(rf_temp, val_data, type = "class")
  val_prob <- predict(rf_temp, val_data, type = "prob")[, 2]
  
  # Calculate metrics
  cm <- confusionMatrix(val_pred, val_data$default.payment.next.month, positive = "Default")
  roc_obj <- roc(val_data$default.payment.next.month, val_prob, levels = c("No_Default", "Default"))
  
  rf_tuning_results <- rbind(rf_tuning_results, data.frame(
    ntree = rf_grid$ntree[i],
    mtry = rf_grid$mtry[i],
    nodesize = rf_grid$nodesize[i],
    Accuracy = cm$overall["Accuracy"],
    Precision = cm$byClass["Precision"],
    Recall = cm$byClass["Recall"],
    F1 = cm$byClass["F1"],
    AUC = as.numeric(auc(roc_obj))
  ))
  
  cat(sprintf("AUC: %.4f\n", as.numeric(auc(roc_obj))))
}

end_time <- Sys.time()
cat(sprintf("\n✓ Tuning completed in %.1f seconds\n\n", as.numeric(difftime(end_time, start_time, units = "secs"))))

# Display tuning results
cat("Random Forest Tuning Results:\n")
print(rf_tuning_results %>% arrange(desc(AUC)))
cat("\n")

# Select best hyperparameters based on AUC
best_rf_params <- rf_tuning_results[which.max(rf_tuning_results$AUC), ]
cat("Best Random Forest Parameters:\n")
print(best_rf_params)
cat("\n")

# 6.2 Train Final Random Forest Model
cat("Training final Random Forest model with best parameters...\n")

start_time <- Sys.time()

rf_final <- randomForest(
  default.payment.next.month ~ .,
  data = train_data,
  ntree = best_rf_params$ntree,
  mtry = best_rf_params$mtry,
  nodesize = best_rf_params$nodesize,
  classwt = class_weights,
  importance = TRUE,
  keep.forest = TRUE
)

end_time <- Sys.time()
cat(sprintf("✓ Model trained in %.1f seconds\n\n", as.numeric(difftime(end_time, start_time, units = "secs"))))

print(rf_final)
cat("\n")

# 6.3 Validation Set Performance
cat("Random Forest - Validation Set Performance:\n")
rf_val_pred <- predict(rf_final, val_data, type = "class")
rf_val_prob <- predict(rf_final, val_data, type = "prob")[, 2]

rf_val_cm <- confusionMatrix(rf_val_pred, val_data$default.payment.next.month, positive = "Default")
print(rf_val_cm)
cat("\n")

# Save Random Forest model
saveRDS(rf_final, "rf_model.rds")
cat("✓ Random Forest model saved to 'rf_model.rds'\n\n")

# ==============================================================================
# 7. SUPPORT VECTOR MACHINE MODEL (ML2 METHOD)
# ==============================================================================

cat("="*80, "\n")
cat("SECTION C: SUPPORT VECTOR MACHINE TRAINING (ML2)\n")
cat("="*80, "\n\n")

# 7.1 Data Scaling (Critical for SVM!)
cat("Scaling numeric features for SVM...\n")

# Identify numeric columns (exclude target and categorical)
numeric_features <- names(train_data)[sapply(train_data, is.numeric)]

# Create preprocessing object
preproc <- preProcess(train_data[, numeric_features], method = c("center", "scale"))

# Apply scaling
train_scaled <- train_data
train_scaled[, numeric_features] <- predict(preproc, train_data[, numeric_features])

val_scaled <- val_data
val_scaled[, numeric_features] <- predict(preproc, val_data[, numeric_features])

test_scaled <- test_data
test_scaled[, numeric_features] <- predict(preproc, test_data[, numeric_features])

cat("✓ Features scaled using standardization (mean=0, sd=1)\n\n")

# 7.2 SVM Hyperparameter Tuning
cat("Starting SVM hyperparameter tuning...\n")

# Define hyperparameter grid for RBF kernel
svm_grid <- expand.grid(
  kernel = c("radial"),
  cost = c(0.1, 1, 10),
  gamma = c(0.001, 0.01, 0.1)
)

# Calculate class weights
svm_class_weights <- as.numeric(class_weights)
names(svm_class_weights) <- levels(train_scaled$default.payment.next.month)

# Initialize results storage
svm_tuning_results <- data.frame()

cat("Testing", nrow(svm_grid), "hyperparameter combinations...\n")

start_time <- Sys.time()

for (i in 1:nrow(svm_grid)) {
  cat(sprintf("  [%d/%d] kernel=%s, cost=%.1f, gamma=%.3f... ", 
              i, nrow(svm_grid), svm_grid$kernel[i], svm_grid$cost[i], svm_grid$gamma[i]))
  
  svm_temp <- svm(
    default.payment.next.month ~ .,
    data = train_scaled,
    kernel = as.character(svm_grid$kernel[i]),
    cost = svm_grid$cost[i],
    gamma = svm_grid$gamma[i],
    class.weights = svm_class_weights,
    probability = TRUE
  )
  
  # Predict on validation set
  val_pred <- predict(svm_temp, val_scaled)
  val_prob <- attr(predict(svm_temp, val_scaled, probability = TRUE), "probabilities")[, "Default"]
  
  # Calculate metrics
  cm <- confusionMatrix(val_pred, val_scaled$default.payment.next.month, positive = "Default")
  roc_obj <- roc(val_scaled$default.payment.next.month, val_prob, levels = c("No_Default", "Default"))
  
  svm_tuning_results <- rbind(svm_tuning_results, data.frame(
    kernel = as.character(svm_grid$kernel[i]),
    cost = svm_grid$cost[i],
    gamma = svm_grid$gamma[i],
    Accuracy = cm$overall["Accuracy"],
    Precision = cm$byClass["Precision"],
    Recall = cm$byClass["Recall"],
    F1 = cm$byClass["F1"],
    AUC = as.numeric(auc(roc_obj))
  ))
  
  cat(sprintf("AUC: %.4f\n", as.numeric(auc(roc_obj))))
}

end_time <- Sys.time()
cat(sprintf("\n✓ Tuning completed in %.1f seconds\n\n", as.numeric(difftime(end_time, start_time, units = "secs"))))

# Display tuning results
cat("SVM Tuning Results:\n")
print(svm_tuning_results %>% arrange(desc(AUC)))
cat("\n")

# Select best hyperparameters
best_svm_params <- svm_tuning_results[which.max(svm_tuning_results$AUC), ]
cat("Best SVM Parameters:\n")
print(best_svm_params)
cat("\n")

# 7.3 Train Final SVM Model
cat("Training final SVM model with best parameters...\n")

start_time <- Sys.time()

svm_final <- svm(
  default.payment.next.month ~ .,
  data = train_scaled,
  kernel = as.character(best_svm_params$kernel),
  cost = best_svm_params$cost,
  gamma = best_svm_params$gamma,
  class.weights = svm_class_weights,
  probability = TRUE
)

end_time <- Sys.time()
cat(sprintf("✓ Model trained in %.1f seconds\n\n", as.numeric(difftime(end_time, start_time, units = "secs"))))

print(svm_final)
cat("\n")

# 7.4 Validation Set Performance
cat("SVM - Validation Set Performance:\n")
svm_val_pred <- predict(svm_final, val_scaled)
svm_val_prob <- attr(predict(svm_final, val_scaled, probability = TRUE), "probabilities")[, "Default"]

svm_val_cm <- confusionMatrix(svm_val_pred, val_scaled$default.payment.next.month, positive = "Default")
print(svm_val_cm)
cat("\n")

# Save SVM model
saveRDS(svm_final, "svm_model.rds")
cat("✓ SVM model saved to 'svm_model.rds'\n\n")

# ==============================================================================
# 8. TEST SET EVALUATION (FINAL COMPARISON)
# ==============================================================================

cat("="*80, "\n")
cat("SECTION C: TEST SET EVALUATION (FINAL COMPARISON)\n")
cat("="*80, "\n\n")

cat("⚠️  IMPORTANT: Using test set for FINAL model comparison only\n\n")

# 8.1 Random Forest on Test Set
rf_test_pred <- predict(rf_final, test_data, type = "class")
rf_test_prob <- predict(rf_final, test_data, type = "prob")[, 2]
rf_test_cm <- confusionMatrix(rf_test_pred, test_data$default.payment.next.month, positive = "Default")

cat("Random Forest - Test Set Performance:\n")
print(rf_test_cm)
cat("\n")

# 8.2 SVM on Test Set
svm_test_pred <- predict(svm_final, test_scaled)
svm_test_prob <- attr(predict(svm_final, test_scaled, probability = TRUE), "probabilities")[, "Default"]
svm_test_cm <- confusionMatrix(svm_test_pred, test_scaled$default.payment.next.month, positive = "Default")

cat("SVM - Test Set Performance:\n")
print(svm_test_cm)
cat("\n")

# 8.3 ROC Curves
rf_roc <- roc(test_data$default.payment.next.month, rf_test_prob, levels = c("No_Default", "Default"))
svm_roc <- roc(test_scaled$default.payment.next.month, svm_test_prob, levels = c("No_Default", "Default"))

# Plot 8: ROC Curves Comparison
png("plots/08_roc_curves_comparison.png", width = 10, height = 8, units = "in", res = 300)
plot(rf_roc, col = "#d73027", lwd = 2, main = "ROC Curves Comparison - Test Set", cex.main = 1.5)
plot(svm_roc, col = "#4575b4", lwd = 2, add = TRUE)
legend("bottomright", 
       legend = c(sprintf("Random Forest (AUC = %.4f)", auc(rf_roc)),
                  sprintf("SVM (AUC = %.4f)", auc(svm_roc))),
       col = c("#d73027", "#4575b4"), 
       lwd = 2, 
       cex = 1.2)
abline(a = 0, b = 1, lty = 2, col = "gray50")
dev.off()

# 8.4 Performance Comparison Table
comparison_table <- data.frame(
  Metric = c("Accuracy", "Precision", "Recall", "Specificity", "F1-Score", "AUC-ROC"),
  Random_Forest = c(
    rf_test_cm$overall["Accuracy"],
    rf_test_cm$byClass["Precision"],
    rf_test_cm$byClass["Recall"],
    rf_test_cm$byClass["Specificity"],
    rf_test_cm$byClass["F1"],
    as.numeric(auc(rf_roc))
  ),
  SVM = c(
    svm_test_cm$overall["Accuracy"],
    svm_test_cm$byClass["Precision"],
    svm_test_cm$byClass["Recall"],
    svm_test_cm$byClass["Specificity"],
    svm_test_cm$byClass["F1"],
    as.numeric(auc(svm_roc))
  )
)

comparison_table$Difference <- comparison_table$Random_Forest - comparison_table$SVM

cat("\n")
cat("="*80, "\n")
cat("PERFORMANCE COMPARISON TABLE (TEST SET)\n")
cat("="*80, "\n")
print(comparison_table, digits = 4)
cat("\n")

# Save comparison table
write.csv(comparison_table, "model_comparison.csv", row.names = FALSE)
cat("✓ Comparison table saved to 'model_comparison.csv'\n\n")

# 8.5 Statistical Test: McNemar's Test
mcnemar_table <- table(
  RF = rf_test_pred,
  SVM = svm_test_pred
)

mcnemar_test <- mcnemar.test(mcnemar_table)
cat("McNemar's Test for Model Comparison:\n")
print(mcnemar_test)
cat("\nInterpretation: ", 
    ifelse(mcnemar_test$p.value < 0.05, 
           "Significant difference between models (p < 0.05)",
           "No significant difference between models (p >= 0.05)"), "\n\n")

# ==============================================================================
# 9. MODEL INTERPRETATION (XAI) - SECTION D
# ==============================================================================

cat("="*80, "\n")
cat("SECTION D: EXPLAINABLE AI (XAI) - MODEL INTERPRETATION\n")
cat("="*80, "\n\n")

# 9.1 Random Forest Variable Importance
cat("Random Forest - Variable Importance:\n")

# Extract importance
rf_importance <- importance(rf_final)
rf_importance_df <- data.frame(
  Variable = rownames(rf_importance),
  MeanDecreaseAccuracy = rf_importance[, "MeanDecreaseAccuracy"],
  MeanDecreaseGini = rf_importance[, "MeanDecreaseGini"]
)
rf_importance_df <- rf_importance_df %>% arrange(desc(MeanDecreaseGini))

print(head(rf_importance_df, 10))
cat("\n")

# Plot 9: Variable Importance
p9 <- rf_importance_df %>%
  top_n(15, MeanDecreaseGini) %>%
  ggplot(aes(x = reorder(Variable, MeanDecreaseGini), y = MeanDecreaseGini)) +
  geom_col(fill = "#d73027", alpha = 0.8) +
  coord_flip() +
  labs(
    title = "Random Forest - Variable Importance",
    subtitle = "Top 15 Features by Mean Decrease in Gini",
    x = "Variable",
    y = "Mean Decrease in Gini"
  ) +
  theme_minimal(base_size = 12)

ggsave("plots/09_rf_variable_importance.png", p9, width = 10, height = 8, dpi = 300)

# 9.2 DALEX Explainers
cat("Creating DALEX explainers for XAI analysis...\n")

# Prepare data for DALEX (numeric features only)
X_train <- train_data %>% select(-default.payment.next.month)
y_train <- as.numeric(train_data$default.payment.next.month) - 1

X_test <- test_data %>% select(-default.payment.next.month)
y_test <- as.numeric(test_data$default.payment.next.month) - 1

# Custom predict functions
predict_rf <- function(model, newdata) {
  predict(model, newdata, type = "prob")[, 2]
}

predict_svm <- function(model, newdata) {
  # Scale the data first
  newdata_scaled <- newdata
  newdata_scaled[, numeric_features] <- predict(preproc, newdata[, numeric_features])
  prob <- attr(predict(model, newdata_scaled, probability = TRUE), "probabilities")
  prob[, "Default"]
}

# Create explainers
explainer_rf <- DALEX::explain(
  model = rf_final,
  data = X_test,
  y = y_test,
  predict_function = predict_rf,
  label = "Random Forest",
  verbose = FALSE
)

explainer_svm <- DALEX::explain(
  model = svm_final,
  data = X_test,
  y = y_test,
  predict_function = predict_svm,
  label = "SVM",
  verbose = FALSE
)

cat("✓ DALEX explainers created\n\n")

# 9.3 Variable Importance using DALEX (Model-Agnostic)
cat("Computing model-agnostic variable importance...\n")

vi_rf <- model_parts(explainer_rf, type = "difference")
vi_svm <- model_parts(explainer_svm, type = "difference")

# Plot 10: Variable Importance Comparison
png("plots/10_variable_importance_comparison.png", width = 12, height = 8, units = "in", res = 300)
plot(vi_rf, vi_svm, max_vars = 15)
dev.off()

cat("✓ Variable importance plot saved\n\n")

# 9.4 Partial Dependence Plots
cat("Creating Partial Dependence Plots for top features...\n")

top_features <- head(rf_importance_df$Variable, 5)

# Plot 11-15: PDP for top 5 features
for (i in 1:min(5, length(top_features))) {
  feature <- as.character(top_features[i])
  
  cat(sprintf("  PDP for %s...\n", feature))
  
  pdp_rf <- model_profile(explainer_rf, variables = feature, type = "partial")
  pdp_svm <- model_profile(explainer_svm, variables = feature, type = "partial")
  
  png(sprintf("plots/%d_pdp_%s.png", 10+i, feature), width = 10, height = 6, units = "in", res = 300)
  plot(pdp_rf, pdp_svm, title = sprintf("Partial Dependence Plot: %s", feature))
  dev.off()
}

cat("✓ Partial Dependence Plots saved\n\n")

# 9.5 Individual Predictions (LIME-style)
cat("Creating individual prediction explanations...\n")

# Select representative cases
test_indices <- c(
  which(rf_test_pred == "Default" & test_data$default.payment.next.month == "Default")[1],  # TP
  which(rf_test_pred == "No_Default" & test_data$default.payment.next.month == "No_Default")[1],  # TN
  which(rf_test_pred == "Default" & test_data$default.payment.next.month == "No_Default")[1],  # FP
  which(rf_test_pred == "No_Default" & test_data$default.payment.next.month == "Default")[1]   # FN
)

case_labels <- c("True Positive", "True Negative", "False Positive", "False Negative")

for (i in 1:length(test_indices)) {
  if (!is.na(test_indices[i])) {
    cat(sprintf("  Explaining case: %s (row %d)\n", case_labels[i], test_indices[i]))
    
    bd_rf <- predict_parts(explainer_rf, new_observation = X_test[test_indices[i], ], type = "break_down")
    
    png(sprintf("plots/%d_breakdown_%s.png", 15+i, gsub(" ", "_", case_labels[i])), 
        width = 10, height = 8, units = "in", res = 300)
    plot(bd_rf, title = sprintf("Prediction Breakdown: %s", case_labels[i]))
    dev.off()
  }
}

cat("✓ Individual prediction explanations saved\n\n")

# ==============================================================================
# 10. CONFUSION MATRIX VISUALIZATIONS
# ==============================================================================

cat("Creating confusion matrix visualizations...\n")

# Function to plot confusion matrix
plot_confusion_matrix <- function(cm, title, filename) {
  cm_table <- as.data.frame(cm$table)
  
  p <- ggplot(cm_table, aes(x = Reference, y = Prediction, fill = Freq)) +
    geom_tile(color = "white", size = 1) +
    geom_text(aes(label = Freq), size = 8, color = "white", fontface = "bold") +
    scale_fill_gradient(low = "#4575b4", high = "#d73027", name = "Count") +
    labs(title = title, x = "Actual", y = "Predicted") +
    theme_minimal(base_size = 14) +
    theme(
      plot.title = element_text(hjust = 0.5, face = "bold"),
      axis.text = element_text(size = 12),
      legend.position = "right"
    )
  
  ggsave(filename, p, width = 8, height = 6, dpi = 300)
}

# Plot 20-21: Confusion matrices
plot_confusion_matrix(rf_test_cm, "Random Forest - Test Set Confusion Matrix", 
                      "plots/20_rf_confusion_matrix.png")
plot_confusion_matrix(svm_test_cm, "SVM - Test Set Confusion Matrix", 
                      "plots/21_svm_confusion_matrix.png")

cat("✓ Confusion matrix visualizations saved\n\n")

# ==============================================================================
# 11. FINAL SUMMARY AND REPORT
# ==============================================================================

cat("="*80, "\n")
cat("FINAL SUMMARY\n")
cat("="*80, "\n\n")

cat("Dataset Statistics:\n")
cat(sprintf("  Total observations: %d\n", nrow(credit_clean)))
cat(sprintf("  Training set: %d (%.1f%%)\n", nrow(train_data), nrow(train_data)/nrow(credit_clean)*100))
cat(sprintf("  Validation set: %d (%.1f%%)\n", nrow(val_data), nrow(val_data)/nrow(credit_clean)*100))
cat(sprintf("  Test set: %d (%.1f%%)\n", nrow(test_data), nrow(test_data)/nrow(credit_clean)*100))
cat(sprintf("  Features: %d\n\n", ncol(credit_clean) - 1))

cat("Best Hyperparameters:\n")
cat("  Random Forest:\n")
cat(sprintf("    ntree: %d\n", best_rf_params$ntree))
cat(sprintf("    mtry: %d\n", best_rf_params$mtry))
cat(sprintf("    nodesize: %d\n\n", best_rf_params$nodesize))

cat("  SVM:\n")
cat(sprintf("    kernel: %s\n", best_svm_params$kernel))
cat(sprintf("    cost: %.1f\n", best_svm_params$cost))
cat(sprintf("    gamma: %.3f\n\n", best_svm_params$gamma))

cat("Test Set Performance:\n")
cat(sprintf("  Random Forest AUC: %.4f\n", auc(rf_roc)))
cat(sprintf("  SVM AUC: %.4f\n\n", auc(svm_roc)))

cat("Model Comparison:\n")
better_model <- ifelse(auc(rf_roc) > auc(svm_roc), "Random Forest", "SVM")
cat(sprintf("  Better performing model: %s\n", better_model))
cat(sprintf("  AUC difference: %.4f\n", abs(auc(rf_roc) - auc(svm_roc))))
cat(sprintf("  Statistical significance: %s\n\n", 
            ifelse(mcnemar_test$p.value < 0.05, "Yes (p < 0.05)", "No (p >= 0.05)")))

cat("Output Files:\n")
cat("  ✓ Trained models: rf_model.rds, svm_model.rds\n")
cat("  ✓ Performance comparison: model_comparison.csv\n")
cat("  ✓ Visualizations: plots/ directory\n\n")

cat("="*80, "\n")
cat("ANALYSIS COMPLETE!\n")
cat("="*80, "\n\n")

cat("All results have been saved. You can now:\n")
cat("1. Review plots in the 'plots/' directory\n")
cat("2. Check model_comparison.csv for performance metrics\n")
cat("3. Load saved models using: readRDS('rf_model.rds') or readRDS('svm_model.rds')\n")
cat("4. Use these results to write your project report\n\n")

# ==============================================================================
# END OF SCRIPT
# ==============================================================================
