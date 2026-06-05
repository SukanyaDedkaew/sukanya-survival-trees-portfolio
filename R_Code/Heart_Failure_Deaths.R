# Import a dataset at https://archive.ics.uci.edu/dataset/519/heart+%20failure+clinical+records
heart_failure_clinical <-read.csv(file.choose(),header = TRUE)

# Show whole dataset 
heart_failure_clinical

# Show data types
str(heart_failure_clinical)

# Descriptive Statistic
summary(heart_failure_clinical)

# --- Survival Trees (Single Survival Trees) ---
# Loading 'rpart' for recursive partitioning and 'survival' for core survival analysis routines.

install.packages("rpart")
install.packages("survival")

library(rpart)
library(survival)

# Fit an rpart model for survival analysis.
# 'z' is the survival object created using Surv(time, DEATH_EVENT).
# method = "exp" specifies an exponential survival model.
z <-Surv(heart_failure_clinical$time,heart_failure_clinical$DEATH_EVENT)
z
class(z)



# Fit using all 11 predictor variables.
fit <-rpart(z~anaemia + creatinine_phosphokinase + diabetes + ejection_fraction + 
      high_blood_pressure + platelets + serum_creatinine + serum_sodium + gender + 
      smoking + age, data = heart_failure_clinical, method = "exp")
                                                                                                     
plot(fit, uniform = TRUE, margin = 0.1)
text(fit, use.n = TRUE, cex = 0.7)
print


# Summarize a fitted rpart Object for returning a detailed listing of a fitted rpart object of fit.
summary(rpart(z~anaemia + creatinine_phosphokinase + diabetes + ejection_fraction + high_blood_pressure 
+ platelets + serum_creatinine + serum_sodium + gender + smoking + age, data = heart_failure_clinical, 
method = "exp"))


# Plot the complexity parameter table to visualize cross-validation results.
plotcp(fit)


# Display the CP table:
# Rel error = Relative error (uncertainty of measurement relative to the size).
# xerror = Cross-validated error estimate (10-fold).
# xstd = Standard error of the xerror.
printcp(fit)

# Fit to where which is a number of observations in the root node
fit$where


# Z Survival Variable & Independant Variables fitted
# Where is a number of observations in the root node 
# Median (Median Survival Time) is calculated as the smallest survival time for which the survivor function is less than or equal to 0.5. 
# 0.95LCL = Lower Confidence Limit 95% 
# 0.95UCL = Upper Confidence Limit 95% 
km <- survfit(z~fit$where, data = heart_failure_clinical)
km


# --- Model 2: ctree (Conditional Inference Trees) ---
# 'ctree' provides a framework for non-parametric, tree-based regression and classification models.
# https://cran.r-project.org/web/packages/partykit/vignettes/ctree.pdf

install.packages("party")

# Use package 
library(party)
library(survival)

# Fit a rpart model by using dependent variable (See at page 21 of Document Rpart Code) https://cran.r-project.org/web/packages/rpart/rpart.pdf
# z = survival response variables
# z = Surv(time,DEATH_EVENT) where Surv is a survival object constructed using the survival package
# Method = "exp" means y is a survival object.
z <-Surv(heart_failure_clinical$time,heart_failure_clinical$DEATH_EVENT)
z
class(z)


# Fit & plot conditional inference tree of ctree function 
fitcstree <- ctree(z~anaemia + creatinine_phosphokinase + diabetes + ejection_fraction + high_blood_pressure + platelets + serum_creatinine + serum_sodium + gender + smoking + age, data = heart_failure_clinical)
fitcstree

plot(fitcstree)


# Fit where for searching the patients or populations being what terminal node on ctree model.
where(fitcstree)
table(where(fitcstree))


# Compute statistics for the conditional distribution of the response, including Kaplan-Meier curves for censored data.
stree <-treeresponse(fitcstree)
stree


# Package ipred (Inproved Predictors)
# Compute statistics for the conditional distribution of the response, including Kaplan-Meier curves for censored data.
# https://cran.r-project.org/web/packages/ipred/ipred.pdf

install.packages("ipred")

library(ipred)
library(survival)
library(rpart)
library(party)

# Use bagging to combine multiple survival trees and improve predictive accuracy.
# Brier score is used as a measure of the accuracy of probabilistic forecasts.
# nbagg: number of bootstrap replications.
# coob: logical, indicating whether to compute out-of-bag error estimates.

# Brier score in Model rpart (Survival Trees)
gbag <- bagging(Surv(time,DEATH_EVENT) ~., data=heart_failure_clinical, model = rpart, nbagg = 299, coob=TRUE)
print(gbag)        

# Brier score in Model ctree (Survival Trees)
gbag <- bagging(Surv(time,DEATH_EVENT) ~., data=heart_failure_clinical, model = ctree, nbagg = 299, coob=TRUE)
print(gbag)        

# Brier score in Model linear
gbag <- bagging(Surv(time,DEATH_EVENT) ~., data=heart_failure_clinical, model = lm, nbagg = 299, coob=TRUE)
print(gbag)        


# Estimators of Prediction Error, https://cran.r-project.org/web/packages/ipred/ipred.pdf

library(survival)
library(ipred)
library(survival)
library(rpart)
library(party)


## Model = rpart
# cv (cross-validation 10-fold) = estimator of the misclassification error 
errorest(DEATH_EVENT ~ ., data=heart_failure_clinical, model=rpart,
         estimator = "cv")


# boot(bootstrap) = estimator of the misclassification error
errorest(DEATH_EVENT ~ ., data=heart_failure_clinical, model=rpart,
         estimator = "boot")

## Model = ctree
# cv (cross-validation 10-fold) = estimator of the misclassification error 
errorest(DEATH_EVENT ~ ., data=heart_failure_clinical, model=ctree,
         estimator = "cv")

# boot(bootstrap) = estimator of the misclassification error
errorest(DEATH_EVENT ~ ., data=heart_failure_clinical, model=ctree,
         estimator = "boot")


## Model = linear
# cv (cross-validation 10-fold) = estimator of the misclassification error 
errorest(DEATH_EVENT ~ ., data=heart_failure_clinical, model=lm,
         estimator = "cv", predict = mypredict.lm)

# boot(bootstrap) = estimator of the misclassification error
errorest(DEATH_EVENT ~ ., data=heart_failure_clinical, model=lm,
         estimator = "boot", predict = mypredict.lm)


# Paper : Bagging Survival Trees, https://drive.google.com/file/d/1ED6Fv3niCY1-p8VRHcZ3z92hrnTsQZlH/view?usp=sharing
# https://cran.r-project.org/web/packages/ipred/ipred.pdf on Page 35 of function "rsurv"(Simulate Survival Data)

# Model A :  logarithms of the haxards = 0

library(survival)
heart_failure_clinical <- rsurv(299, model= "A", gamma=2)
coxph(Surv(time,cens) ~ ., data=heart_failure_clinical)


heart_failure_clinical <- rsurv(299, model= "A", gamma=5)
coxph(Surv(time,cens) ~ ., data=heart_failure_clinical)


heart_failure_clinical <- rsurv(299, model= "A", gamma=7)
coxph(Surv(time,cens) ~ ., data=heart_failure_clinical)

# Model B : logarithms of the haxards =  3I(X1 ??? 0.5 ??? X2 > 0.5)
# Two risk categories are established in terms of a tree with two leaves produced by predictors X1 and X2.

heart_failure_clinical <- rsurv(299, model= "B", gamma=2)
coxph(Surv(time,cens) ~ ., data=heart_failure_clinical)


heart_failure_clinical <- rsurv(299, model= "B", gamma=5)
coxph(Surv(time,cens) ~ ., data=heart_failure_clinical)


heart_failure_clinical <- rsurv(299, model= "B", gamma=7)
coxph(Surv(time,cens) ~ ., data=heart_failure_clinical)

# Model C : logarithms of the haxards =  3X1+ X2
# A linear combination of X1 and X2 is required for model C.

heart_failure_clinical <- rsurv(299, model= "C", gamma=2)
coxph(Surv(time,cens) ~ ., data=heart_failure_clinical)


heart_failure_clinical <- rsurv(299, model= "C", gamma=5)
coxph(Surv(time,cens) ~ ., data=heart_failure_clinical)


heart_failure_clinical <- rsurv(299, model= "C", gamma=7)
coxph(Surv(time,cens) ~ ., data=heart_failure_clinical)
