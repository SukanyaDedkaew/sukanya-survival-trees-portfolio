# =========================================================
# Setup and Packages
# =========================================================

install.packages(c(
  "rpart", "rpart.plot", "party", "survival", "survminer",
  "dplyr", "ipred", "survcomp", "randomForestSRC", "tidyr", "partykit","ggplot2", "ggparty"))

library(rpart)
library(rpart.plot)
library(party)
library(survival)
library(survminer)
library(dplyr)
library(ipred)
library(survcomp)
library(randomForestSRC)
library(tidyr)
library(partykit)
library(ggplot2)
library(ggparty)

# =========================================================
# Load Dataset
# https://www.kaggle.com/datasets/shashanknecrothapa/machine-failure-predictions/code
# =========================================================

df <- read.csv(file.choose(),header = TRUE) 
df

# Prepare variables for Survival Analysis
# time = Tool wear, event = Machine failure

df_model <- df %>%
  mutate(
    time = as.numeric(`Tool.wear..min.`) + 0.1,    
    event = as.numeric(Machine.failure),           
    Type = as.factor(Type),
    AirTemp = as.numeric(`Air.temperature..K.`),
    ProcessTemp = as.numeric(`Process.temperature..K.`),
    RotSpeed = as.numeric(`Rotational.speed..rpm.`),
    Torque = as.numeric(`Torque..Nm.`)
  ) %>%
  select(time, event, AirTemp, ProcessTemp, RotSpeed, Torque, Type) %>%

  # Clean: Cut the rows that contain "NA" / 
  filter(
    !is.na(time) & !is.infinite(time) &
      !is.na(event) & !is.infinite(event) &
      !is.na(AirTemp) & !is.infinite(AirTemp) &
      !is.na(ProcessTemp) & !is.infinite(ProcessTemp) &
      !is.na(RotSpeed) & !is.infinite(RotSpeed) &
      !is.na(Torque) & !is.infinite(Torque) &
      !is.na(Type)
  )

# Verify the information
summary(df_model)
cat("Missing values:\n")
print(colSums(is.na(df_model)))


# Event / No Event

df_model$event_factor <- factor(df_model$event, levels = c(0,1), labels = c("no event","event"))
cat("\n=== Event / No Event ===\n")
print(table(df_model$event_factor))
cat("=== Percent ===\n")
print(round(prop.table(table(df_model$event_factor))*100,2))

# =========================================================
# Split Train 70 / Test 30
# =========================================================

set.seed(123)
train_idx <- sample(1:nrow(df_model), size = 0.7 * nrow(df_model))
train_df <- df_model[train_idx, ]
test_df  <- df_model[-train_idx, ]


# Ensure factor levels match
test_df$Type <- factor(test_df$Type, levels = levels(train_df$Type))


# Single Survival Trees (rpart)

set.seed(123)
fit_rpart <- rpart(
  Surv(time, event) ~ AirTemp + ProcessTemp + RotSpeed + Torque + Type,
  data = df_model,
  method = "exp",
  control = rpart.control(cp = 0.01, minsplit = 30, minbucket = 10))


# Plot rpart

rpart.plot(fit_rpart,
           type = 3,
           extra = 101,
           fallen.leaves = TRUE,
           cex = 0.7)

# Terminal node summary rpart
df_model$rpart_node <- fit_rpart$where
rpart_nodes <- data.frame(node = unique(df_model$rpart_node)) %>%
  rowwise() %>%
  mutate(
    n_total = sum(df_model$rpart_node == node),
    n_event = sum(df_model$rpart_node == node & df_model$event==1),
    n_censor = sum(df_model$rpart_node == node & df_model$event==0),
    median_survival = median(df_model$time[df_model$rpart_node == node])
  )
print(rpart_nodes)

# Paths for each terminal node
cat("\n--- rpart Node Paths ---\n")
path.rpart(fit_rpart, nodes = rpart_nodes$node, print.it = TRUE)


# Terminal node Kaplan-Meier

df_model$node <- fit_rpart$where
km_fit <- survfit(Surv(time, event) ~ node, data = df_model)
ggsurvplot(km_fit, data = df_model,
           palette = "Dark2",
           legend.title = "Terminal Node",
           title = "KM by Terminal Node (rpart)")

# Performance rpart (C-index)

pred_rpart <- predict(fit_rpart, newdata=test_df, type="vector")
valid_idx <- complete.cases(pred_rpart, test_df$time, test_df$event)
cindex_rpart <- concordance.index(pred_rpart[valid_idx],
                                  surv.time=test_df$time[valid_idx],
                                  surv.event=test_df$event[valid_idx])
cat("C-index (rpart):", round(cindex_rpart$c.index,3), "\n")



# =========================================================
# Bagging Survival Tree (ipred)
# =========================================================

set.seed(123)
fit_bag <- bagging(
  time ~ AirTemp + ProcessTemp + RotSpeed + Torque + Type,
  data = df_model,
  nbagg = 100,
  coob = TRUE,
  control = rpart.control(minsplit=20, minbucket=10, maxdepth=6)
)

# Predict expected survival time

pred_bag <- as.numeric(predict(fit_bag, newdata=test_df, type="response"))
cindex_bag <- concordance.index(-pred_bag, surv.time=test_df$time, surv.event=test_df$event)
cat("C-index (Bagging):", round(cindex_bag$c.index,3), "\n\n")

# =========================================================
# Random Survival Forest (RSF)
# =========================================================
set.seed(123)
fit_rsf <- rfsrc(
  Surv(time, event) ~ AirTemp + ProcessTemp + RotSpeed + Torque + Type,
  data = df_model,
  ntree = 500,
  importance=TRUE,
  samptype="swor"
)
pred_rsf <- predict(fit_rsf, newdata = test_df)$predicted
cindex_rsf <- concordance.index(pred_rsf, surv.time = test_df$time, surv.event = test_df$event)
cat("C-index (RSF):", round(cindex_rsf$c.index,3), "\n\n")

# Variable importance RSF
var_imp <- data.frame(fit_rsf$importance) %>%
  rename(Importance = fit_rsf.importance) %>%
  rownames_to_column(var="Variable") %>%
  arrange(desc(Importance))
print(var_imp)

ggplot(var_imp, aes(x=reorder(Variable, Importance), y=Importance)) +
  geom_bar(stat="identity", fill="#4682B4") +
  coord_flip() +
  labs(title="RSF Variable Importance", x="Variable", y="Importance") +
  theme_minimal()

# =========================================================
# Performance Comparison
# =========================================================
performance <- data.frame(
  Model = c("rpart", "Bagging", "RSF"),
  C_index = c(cindex_rpart$c.index, cindex_bag$c.index, cindex_rsf$c.index)
)
print(performance)

ggplot(performance, aes(x = Model, y = C_index, fill = Model)) +
  geom_bar(stat = "identity", width = 0.6) +
  ylim(0, 1) +
  geom_text(aes(label = round(C_index, 3)), vjust = -0.5) +
  theme_minimal() +
  ggtitle("Survival Model Performance Comparison (C-index)") +
  theme(legend.position = "none")
