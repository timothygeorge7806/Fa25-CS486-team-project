library(randomForest)
library(caret)
library(e1071)

# Load Data
e1data <- read.csv("training_heart_disease_uci_cleaned.csv", header=TRUE)
e1data$num <- as.factor(e1data$num)


#Grid Search all values of NTREE, MTRY, CUTOFF
ntree_grid <- c(1000,2000,5000)
mtry_grid <- pmax(1, round(c(0.5, 1, 2) * sqrt(length(e1data))))
cutoff_grid <- list(0.3, 0.5, 0.7)

results <- data.frame(
  ntree = integer(),
  mtry = integer(),
  cutoff = numeric(),
  oob = numeric(),
  stringsAsFactors = FALSE
)

# Build and Train Model
for(n_tree in ntree_grid) {
  for(m_try in mtry_grid) {
    for(cut_off in cutoff_grid) {
      model1.rf <- randomForest(num ~. , data = e1data,
                                ntree = n_tree,
                                mtry = m_try,
                                CUTOFF = cut_off,
                                do.trace=100)
      oob_err <- tail(model1.rf$err.rate[,"OOB"],1)
      results <- rbind(results, data.frame(ntree = n_tree, mtry = m_try, cutoff = cut_off, oob = oob_err))
    }
  }
}

# Get best hyperparameters
best_values <- results[which.min(results$oob), ]
print(best_values)

# Recreate run-time best model
model1.rf <- randomForest(num ~. , data = e1data,
                          ntree = best_values$ntree,
                          mtry = best_values$mtry,
                          importance = TRUE,
                          CUTOFF = best_values$cutoff,
                          do.trace=100)
print(model1.rf)

# Calculate accuracy statistics
TN <- model1.rf$confusion[1,1]   # True Negatives
FP <- model1.rf$confusion[1,2]   # False Positives
FN <- model1.rf$confusion[2,1]   # False Negatives
TP <- model1.rf$confusion[2,2]   # True Positives

precision <- TP / (TP + FP)
recall <- TP / (TP + FN)
f1 <- 2 * (recall * precision) / (recall + precision)

cat("Precision:", round(precision, 4), "\n")
cat("Recall   :", round(recall, 4), "\n")
cat("F1 Score :", round(f1, 4), "\n")

# Rank top 10 features (Taken from ChatGPT)
importance_vals <- importance(model1.rf)
importance_df <- data.frame(Feature = rownames(importance_vals),
                            MDA = importance_vals[, "MeanDecreaseAccuracy"])

importance_df <- importance_df[order(-importance_df$MDA), ]

print(head(importance_df, 10))

# varImpPlot(model1.rf, n.var = 10)

#Save and verify trained model
saveRDS(model1.rf, "modelBEST.rds")
modelbest.rf <- readRDS("modelBEST.rds")
verification_data <- read.csv("verification_heart_disease_uci_cleaned.csv", header=TRUE)
verification_data$num <- as.factor(verification_data$num)
predictions <- predict(modelbest.rf, verification_data, type="prob")
print(predictions)

