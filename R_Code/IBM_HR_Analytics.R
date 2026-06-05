# =========================================================
# Setup Packages
# =========================================================
install.packages(c("survival","survminer","rpart","rpart.plot","party","partykit",
                   "survRM2","survcomp","dplyr", "grid", "ipred"))

library(survival)
library(survminer)
library(rpart)
library(rpart.plot)
library(party)
library(partykit)
library(survRM2)
library(survcomp)
library(dplyr)
library(grid)
library(ipred)

# =========================================================
# Load and Preprocess Dataset 
# Dataset Source : https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset/data 
df <- read.csv(file.choose(), header=TRUE)

# Define survival object: time = YearsAtCompany, event = Attrition status (1=Yes/Event, 0=No/Censor).
df <- df %>%
  mutate(
    event = ifelse(Attrition == "Yes", 1, 0),
    time = as.numeric(YearsAtCompany)
  )

# Adjust zero-time values to ensure stability in survival models.
df$time[df$time <= 0] <- 0.1  

# Convert categorical variables to factors for proper model interpretation.
categorical_vars <- c("BusinessTravel","Department","EducationField",
                      "Gender","JobRole","MaritalStatus","OverTime")
df[categorical_vars] <- lapply(df[categorical_vars], factor)

# =========================================================
# Model Selection and Formula Setup
# =========================================================
predictors <- c("Age", "MonthlyIncome", "DistanceFromHome", "JobSatisfaction",
                "EnvironmentSatisfaction", "WorkLifeBalance", "JobLevel",
                "TotalWorkingYears", "TrainingTimesLastYear", "YearsInCurrentRole",
                "YearsSinceLastPromotion", "YearsWithCurrManager")

formula_surv <- as.formula(
  paste("Surv(time, event) ~", paste(predictors, collapse = " + "))
)

# =========================================================
# rpart Survival Tree (Recursive Partitioning)
# =========================================================
df_rpart <- df %>% select(any_of(c(predictors, "time", "event")))
rpart_fit <- rpart(
  formula = formula_surv,
  data = df_rpart,
  method = "exp",
  control = rpart.control(cp = 0.01, minsplit = 15, minbucket = 5)
)

# Visualize terminal nodes and complexity parameter
rpart.plot(rpart_fit, extra=101, fallen.leaves=TRUE, cex=0.6)
plotcp(rpart_fit)

# Calculate risk scores per node (median predicted hazard proxy)
df$rpart_node <- rpart_fit$where
rpart_summary <- data.frame(node = unique(df$rpart_node)) %>%
  rowwise() %>%
  mutate(
    n_obs = sum(df$rpart_node == node),
    n_event = sum(df$rpart_node == node & df$event==1),
    n_censor = sum(df$rpart_node == node & df$event==0),
    median_time = median(df$time[df$rpart_node == node]),
    risk_score = median(predict(rpart_fit, type="vector")[df$rpart_node == node])
  )

# Calculate C-index to assess model discrimination performance
df <- df %>%
  left_join(rpart_summary %>% select(node, risk_score) %>% rename(rpart_risk = risk_score),
            by = c("rpart_node" = "node"))

valid_idx_rpart <- which(!is.na(df$rpart_risk) & is.finite(df$rpart_risk))
cindex_rpart <- concordance.index(
  x=df$rpart_risk[valid_idx_rpart],
  surv.time=df$time[valid_idx_rpart],
  surv.event=df$event[valid_idx_rpart]
)
cat("C-index rpart:", round(cindex_rpart$c.index, 3), "\n")

# =========================================================
# Conditional Survival Tree (ctree)
# =========================================================
# ctree uses non-parametric hypothesis testing for recursive partitioning.
ctree_fit <- ctree(
  formula_surv,
  data = df, 
  control = ctree_control(mincriterion = 0.9, minsplit = 30, minbucket = 15, maxdepth = 5)
)

# Visualize conditional inference tree structure
plot(ctree_fit, gp=gpar(fontsize=6), main="Conditional Survival Tree")

# Assign node IDs and calculate event-based risk scores
df$ctree_node <- predict(ctree_fit, type="node")
risk_per_node <- df %>%
  group_by(ctree_node) %>%
  summarise(risk_score_ctree = sum(event) / n()) %>%
  ungroup()

df <- df %>% left_join(risk_per_node, by = "ctree_node")

# =========================================================
# Bagging (Bootstrap Aggregation) for Improved Prediction
# =========================================================
# Use bagging to reduce variance and compute OOB (Out-of-Bag) error estimates.
bagging_data <- df %>% select(any_of(c(predictors, "time", "event")))

gbag_rpart <- bagging(
  Surv(time, event) ~ .,
  data = bagging_data,
  nbagg = 299,
  coob = TRUE,
  control = rpart.control(minsplit=15, minbucket=5, cp=0.01)
)
print(gbag_rpart)

# =========================================================
# Rule Extraction and Visualization
# =========================================================
# Extract logical rules from terminal nodes for business interpretation
rules <- partykit:::.list.rules.party(ctree_fit)
rules_df <- data.frame(
  node = as.integer(names(rules)),
  rule = unlist(rules),
  stringsAsFactors = FALSE
)