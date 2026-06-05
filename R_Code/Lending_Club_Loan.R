# Lending Club Loan Data at www.kaggle.com/datasets/adarshsng/lending-club-loan-data-csv?resource=download 

############################################################
# 0) CLEAR ENVIRONMENT
############################################################
rm(list = ls())

############################################################
# INSTALL & LOAD PACKAGES
############################################################

# Load packages

install.packages(c("dplyr","survival","rpart","rpart.plot","pec","prodlim","survminer","readr", "tidyr"))

library(dplyr)
library(survival)
library(rpart)
library(rpart.plot)
library(pec)       # สำหรับ Brier score
library(prodlim)   # สำหรับ Kaplan-Meier baselin
library(survminer)
library(ggplot2)
library(readr)
library(tidyr)

# Load a dataset
loan_df <- read.csv(file.choose(), stringsAsFactors = FALSE)

# 3️ Number of customers / rows
cat("Number of customers / records:", nrow(loan_df), "\n")

# 4️⃣Number of columns
cat("Number of columns:", ncol(loan_df), "\n")


# Identify all empty columns
empty_cols <- sapply(loan_df, function(x) all(is.na(x)))
cat("Empty columns found:\n")
print(names(loan_df)[empty_cols])

# Count of empty columns
cat("Total number of empty columns:", sum(empty_cols), "\n")


# Show amount rows and columns
dim(loan_df)

# Show data types
str(loan_df)

# Descriptive Statistic
summary(loan_df)

############################################################
# PREPARE DATA
############################################################

df_clean <- loan_df %>%
  filter(!is.na(loan_status),
         !is.na(term),
         !is.na(int_rate),
         !is.na(annual_inc),
         !is.na(dti),
         !is.na(installment),
         !is.na(revol_util),
         !is.na(delinq_2yrs),
         !is.na(inq_last_6mths),
         !is.na(open_acc),
         !is.na(total_acc),
         !is.na(pub_rec_bankruptcies)) %>%
  mutate(
    event = ifelse(loan_status %in% c("Charged Off","Default"), 1, 0),
    time = as.numeric(gsub("[^0-9]", "", term)),
    int_rate = as.numeric(gsub("%","", int_rate)),
    revol_util = as.numeric(gsub("%","", revol_util))
  )

############################################################
# SELECT VARIABLES
############################################################

df_model <- df_clean %>%
  dplyr::select(time, event,
                loan_amnt,
                int_rate,
                installment,
                annual_inc,
                dti,
                revol_util,
                delinq_2yrs,
                inq_last_6mths,
                open_acc,
                total_acc,
                pub_rec_bankruptcies,
                purpose) %>%
  na.omit()

df_model$purpose <- as.factor(df_model$purpose)



############################################################
# BUILD SURVIVAL TREE
############################################################

set.seed(123)

surv_tree <- rpart(
  Surv(time, event) ~ .,
  data = df_model,
  method = "exp",
  control = rpart.control(
    minsplit = 100,
    minbucket = 50,
    cp = 0.001,
    maxdepth = 5
  )
)

# Plot tree
rpart.plot(surv_tree, type=3, extra=101, fallen.leaves=TRUE, cex=0.7)

############################################################
# PRUNE TREE
############################################################

best_cp <- surv_tree$cptable[which.min(surv_tree$cptable[,"xerror"]), "CP"]
pruned_tree <- prune(surv_tree, cp = best_cp)

rpart.plot(pruned_tree, type=3, extra=101, fallen.leaves=TRUE, cex=0.8)

############################################################
# C-INDEX
############################################################

risk_score <- predict(pruned_tree, type="vector")
c_index <- survConcordance(Surv(time, event) ~ risk_score, data = df_model)$concordance
print(paste("C-index =", round(c_index,3)))


############################################################
# BRIER SCORE (FIXED: Tree + Kaplan-Meier by Node)
############################################################

df_model$risk_group <- as.factor(pruned_tree$where)

km_tree <- prodlim(Surv(time, event) ~ risk_group, data = df_model)

brier_model <- pec(
  object = list("TreeKM" = km_tree),
  formula = Surv(time, event) ~ risk_group,
  data = df_model,
  times = c(12,24,36,48,60),
  cens.model = "marginal",
  splitMethod = "BootCv",
  B = 20
)

print("Brier Score:")
print(brier_model$AppErr)

mean_brier <- mean(brier_model$AppErr$TreeKM)
print(paste("Mean Brier =", round(mean_brier,3)))


############################################################
# TERMINAL NODE SUMMARY
############################################################

df_model$leaf <- pruned_tree$where
df_model$mst  <- predict(pruned_tree, type = "vector")
total_obs <- nrow(df_model)

terminal_nodes <- df_model %>%
  group_by(leaf) %>%
  summarise(
    n_customers = n(),
    pct_observation = n()/total_obs,
    n_default = sum(event),
    pct_event = mean(event),
    mst = mean(mst),
    avg_loan_amnt = mean(loan_amnt),
    avg_int_rate = mean(int_rate),
    avg_installment = mean(installment),
    avg_annual_inc = mean(annual_inc),
    avg_dti = mean(dti),
    avg_revol_util = mean(revol_util),
    avg_delinq_2yrs = mean(delinq_2yrs),
    avg_inq_last_6mths = mean(inq_last_6mths),
    avg_open_acc = mean(open_acc),
    avg_total_acc = mean(total_acc),
    avg_pub_rec_bankruptcies = mean(pub_rec_bankruptcies)
  ) %>%
  arrange(desc(pct_event)) %>%
  mutate(
    grade = case_when(
      pct_event < 0.05 ~ "A",
      pct_event < 0.10 ~ "B",
      pct_event < 0.20 ~ "C",
      TRUE ~ "D"
    ),
    segment_name = paste0("Segment_", grade, "_", leaf)
  )


############################################################
# EXTRACT PATH ROOT -> TERMINAL NODE
############################################################

leaf_nodes <- as.numeric(row.names(pruned_tree$frame[pruned_tree$frame$var == "<leaf>", ]))

node_paths <- path.rpart(pruned_tree, nodes = leaf_nodes, print.it = FALSE)

path_df <- data.frame(
  leaf = as.numeric(names(node_paths)),
  path = sapply(node_paths, function(x) paste(c("ROOT", x), collapse = " -> "))
)

terminal_nodes <- terminal_nodes %>%
  left_join(path_df, by = "leaf")



############################################################
# EXPECTED LOSS + CREDIT POLICY
############################################################
df_model <- df_model %>%
  mutate(
    PD = predict(pruned_tree, type = "vector"),
    LGD = 0.5,
    EAD = loan_amnt,
    EL = PD * LGD * EAD
  )

df_model <- df_model %>%
  left_join(
    terminal_nodes %>% select(leaf, grade, segment_name),
    by = "leaf"
  )

df_model <- df_model %>%
  mutate(
    decision = case_when(
      grade == "A" ~ "Approve_LowRate",
      grade == "B" ~ "Approve_Standard",
      grade == "C" ~ "Approve_HighRate",
      grade == "D" ~ "Reject"
    )
  )

############################################################
# SIMULATION REVENUE + LOSS
############################################################
df_model <- df_model %>%
  mutate(
    revenue = case_when(
      decision == "Approve_LowRate" ~ loan_amnt * 0.02,
      decision == "Approve_Standard" ~ loan_amnt * 0.03,
      decision == "Approve_HighRate" ~ loan_amnt * 0.05,
      TRUE ~ 0
    ),
    loss = ifelse(decision == "Reject", 0, EL)
  )

############################################################
# BUSINESS IMPACT SUMMARY
############################################################
simulation_summary <- df_model %>%
  summarise(
    total_customers = n(),
    total_revenue = sum(revenue),
    total_loss = sum(loss),
    net_profit = total_revenue - total_loss
  )

baseline_summary <- df_model %>%
  summarise(
    baseline_loss = sum(EL)
  )

comparison <- data.frame(
  baseline_loss = baseline_summary$baseline_loss,
  new_loss = simulation_summary$total_loss,
  loss_reduction = baseline_summary$baseline_loss - simulation_summary$total_loss,
  net_profit = simulation_summary$net_profit
)

segment_summary <- df_model %>%
  group_by(segment_name, grade) %>%
  summarise(
    n_customers = n(),
    avg_PD = mean(PD),
    total_EL = sum(EL),
    total_revenue = sum(revenue),
    total_loss = sum(loss),
    net_profit = sum(revenue) - sum(loss)
  ) %>%
  arrange(desc(total_EL))

############################################################
# VISUALIZATION
############################################################
ggplot(terminal_nodes, aes(x = reorder(segment_name, pct_event), y = pct_event)) +
  geom_bar(stat="identity") +
  coord_flip() +
  labs(title="Default Rate by Segment", x="Segment", y="Default Rate")

ggplot(segment_summary, aes(x = reorder(segment_name, total_EL), y = total_EL)) +
  geom_bar(stat="identity") +
  coord_flip() +
  labs(title="Expected Loss by Segment", x="Segment", y="Expected Loss")

ggplot(segment_summary, aes(x = reorder(segment_name, net_profit), y = net_profit)) +
  geom_bar(stat="identity") +
  coord_flip() +
  labs(title="Net Profit by Segment", x="Segment", y="Net Profit")

############################################################
# EXPORT FILES
############################################################
write_csv(terminal_nodes, "terminal_nodes_summary.csv")
write_csv(path_df, "node_paths.csv")
write_csv(segment_summary, "segment_summary.csv")
write_csv(simulation_summary, "simulation_summary.csv")
write_csv(comparison, "comparison.csv")

############################################################
# PRINT RESULTS
############################################################
print("===== Terminal Nodes =====")
print(terminal_nodes)

print("===== Segment Summary =====")
print(segment_summary)

print("===== Simulation Summary =====")
print(simulation_summary)

print("===== Comparison =====")
print(comparison)