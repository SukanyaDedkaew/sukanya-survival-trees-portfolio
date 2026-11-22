

# 0) ENVIRONMENT SETUP & PACKAGES

rm(list = ls())

required_packages <- c("dplyr", "survival", "rpart", "rpart.plot", "survminer", "ggplot2", "readr")
new_packages <- required_packages[!(required_packages %in% installed.packages()[, "Package"])]
if (length(new_packages) > 0) install.packages(new_packages)

library(dplyr)
library(survival)
library(rpart)
library(rpart.plot)
library(survminer)
library(ggplot2)
library(readr)

# Load dataset (Replace file.choose() with file path if running non-interactively)
motor_claim_df <- read.csv(file.choose(), stringsAsFactors = FALSE)


# 1) BASIC DATA INSPECTION

cat("Dataset Dimensions:\n")
print(dim(motor_claim_df))

cat("\nFirst Few Rows:\n")
print(head(motor_claim_df))

cat("\nColumn Names:\n")
print(colnames(motor_claim_df))


# 2) DATA PREPARATION & FEATURE ENGINEERING

motor_claim_df <- motor_claim_df %>%
  mutate(
    # Survival Variables
    time  = pmax(as.numeric(Exposure), 0.001), # Ensure time > 0 to avoid division errors
    event = ifelse(ClaimNb > 0, 1, 0),         # 1 if at least one claim, 0 otherwise
    
    # Factor Variables
    VehBrand = as.factor(VehBrand),
    VehGas   = as.factor(VehGas),
    Region   = as.factor(Region),
    Area     = as.factor(Area),
    
    # Numeric Variables
    VehAge     = as.numeric(VehAge),
    DrivAge    = as.numeric(DrivAge),
    BonusMalus = as.numeric(BonusMalus),
    Density    = as.numeric(Density),
    
    # Feature Engineering
    young_driver = ifelse(DrivAge < 25, 1, 0),
    old_vehicle  = ifelse(VehAge > 10, 1, 0)
    # Note: log_density excluded as Density is already utilized by the decision tree
  ) %>%
  na.omit()


# 3) TRAIN / TEST SPLIT & FACTOR LEVEL SYNCHRONIZATION

set.seed(123)
train_index <- sample(seq_len(nrow(motor_claim_df)), size = 0.8 * nrow(motor_claim_df))
train_data  <- motor_claim_df[train_index, ]
test_data   <- motor_claim_df[-train_index, ]

factor_cols <- c("VehBrand", "VehGas", "Region", "Area")
for (col in factor_cols) {
  test_data[[col]] <- factor(test_data[[col]], levels = levels(factor(train_data[[col]])))
}
test_data <- na.omit(test_data)


# 4) BUILD & PRUNE SURVIVAL TREE

surv_tree <- rpart(
  Surv(time, event) ~ Area + VehAge + DrivAge + BonusMalus + Density + young_driver + old_vehicle,
  data = train_data,
  method = "exp",
  control = rpart.control(minsplit = 3000, minbucket = 1000, cp = 0.00001, maxdepth = 4)
)

best_cp     <- surv_tree$cptable[which.min(surv_tree$cptable[, "xerror"]), "CP"]
pruned_tree <- prune(surv_tree, cp = best_cp)


# 5) MODEL PERFORMANCE METRICS

pred_risk <- as.numeric(predict(pruned_tree, newdata = test_data, type = "vector"))

# Calculate Concordance Index (C-index)
c_index <- concordance(Surv(time, event) ~ pred_risk, data = test_data, reverse = TRUE)$concordance

# Calculate Baseline Hazard from training dataset
baseline_hazard <- sum(train_data$event) / sum(train_data$time)

# Calculate Survival Probability S(t) at t = 1 Year
eval_t             <- 1.0
pred_survival_prob <- exp(-(pred_risk * baseline_hazard) * eval_t)

# Define actual survival status (1 = no claim, 0 = at least one claim)
actual_survival_status <- ifelse(test_data$event == 0, 1, 0)

# Calculate Brier Score (Integrated Brier Score proxy)
brier_score_survival <- mean((actual_survival_status - pred_survival_prob)^2)

cat("\n========== MODEL PERFORMANCE ==========\n")
cat("C-index           :", round(c_index, 4), "\n")
cat("Brier Score (IBS) :", round(brier_score_survival, 4), "\n")


# 6) MEDIAN SURVIVAL TIME (MST) & NODE SUMMARIZATION

cat("\n[PROCESSING] Calculating MST for each terminal node...\n")

train_data$node_index <- pruned_tree$where

node_stats <- train_data %>%
  group_by(node_index) %>%
  do({
    fit <- survfit(Surv(time, event) ~ 1, data = .)
    s   <- summary(fit)$table
    data.frame(
      node_obs    = as.numeric(s["records"]),
      node_events = as.numeric(s["events"]),
      median_time = ifelse(!is.na(s["median"]), s["median"], NA)
    )
  }) %>%
  ungroup()

tree_paths   <- path.rpart(pruned_tree, nodes = rownames(pruned_tree$frame), print.it = FALSE)
path_strings <- sapply(tree_paths, function(x) paste(x[-1], collapse = " AND "))

terminal_nodes_final <- as.data.frame(pruned_tree$frame) %>%
  mutate(
    node_id      = as.numeric(row.names(.)),
    node_row_idx = row_number()
  ) %>%
  filter(var == "<leaf>") %>%
  left_join(node_stats, by = c("node_row_idx" = "node_index")) %>%
  transmute(
    node_id,
    observations            = node_obs,
    events                  = node_events,
    predicted_relative_risk = yval,
    mst_median_time         = median_time,
    theoretical_mean_time   = 1 / (yval * baseline_hazard),
    decision_path           = path_strings[as.character(node_id)]
  )


# 7) EXPORT OUTPUTS & VISUALIZATIONS

if (!dir.exists("output")) {
  dir.create("output")
}

# Export Variable Importance
if (!is.null(pruned_tree$variable.importance)) {
  importance_df <- data.frame(
    Variable   = names(pruned_tree$variable.importance),
    Importance = as.numeric(pruned_tree$variable.importance)
  )
  write_csv(importance_df, "output/variable_importance.csv")
}

# Export Survival Tree Plot
png("output/survival_tree.png", width = 1400, height = 900)
rpart.plot(
  pruned_tree,
  type          = 3,
  extra         = 101,
  fallen.leaves = TRUE,
  box.palette   = "GnBu",
  main          = "Pruned Survival Tree with MST Analysis"
)
dev.off()

cat("\n========== [SUCCESS] ALL OUTPUTS SAVED IN 'output/' ==========\n")