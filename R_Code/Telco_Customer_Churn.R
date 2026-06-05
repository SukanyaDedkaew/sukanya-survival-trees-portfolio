# Import a dataset at https://www.kaggle.com/datasets/blastchar/telco-customer-churn  
Telco_Customer_Churn <-read.csv(file.choose(),header = TRUE)

# Show whole dataset 
Telco_Customer_Churn

# Show data types
str(Telco_Customer_Churn)

# Descriptive Statistic
summary(Telco_Customer_Churn)

# Delete the row "TotalCharges" as NA 
Telco_Customer_Churn <- Telco_Customer_Churn[!is.na(Telco_Customer_Churn$TotalCharges), ]

# Filter to remove the row "Tenure" <= 0 
Telco_Customer_Churn <- Telco_Customer_Churn[Telco_Customer_Churn$Tenure > 0, ]

# Convert TotalCharges as numeric (Some will be blank. "")
Telco_Customer_Churn$TotalCharges <- as.numeric(as.character(Telco_Customer_Churn$TotalCharges))

# Convert Churn as 1/0
Telco_Customer_Churn$Churn <- as.numeric(Telco_Customer_Churn$Churn == "Yes")

# Delete CustomerID for preventing error to bagging.
Telco_Customer_Churn$CustomerID <- NULL

# Survival Trees  (Single Survival Trees)
# loading package rpart uses for recursive partitioning for classification.
# loading package survival uses for the core survival analysis routines.

install.packages("rpart")
install.packages("survival")

library(rpart)
library(survival)

# Create Surv object
z <- Surv(Telco_Customer_Churn$Tenure, Telco_Customer_Churn$Churn)

# Create survival tree by rpart by using method = "exp"
fit <- rpart(z ~ Gender + SeniorCitizen + Partner + Dependents + PhoneService + MultipleLines + 
    InternetService + OnlineSecurity + OnlineBackup + DeviceProtection + TechSupport + 
    StreamingTV + StreamingMovies + Contract + PaperlessBilling + PaymentMethod + 
    MonthlyCharges + TotalCharges, data = Telco_Customer_Churn,method = "exp")

# Show the survival trees.
plot(fit, uniform = TRUE, margin = 0.1)
text(fit, use.n = TRUE, cex = 0.7)

# Show the result of model

print(fit)

# Summarize a fitted rpart Object for returning a detailed listing of a fitted rpart object of fit.
summary(rpart(z~Gender + SeniorCitizen + Partner + Dependents + PhoneService + MultipleLines + 
                InternetService + OnlineSecurity + OnlineBackup + DeviceProtection + TechSupport + 
                StreamingTV + StreamingMovies + Contract + PaperlessBilling + PaymentMethod + 
                MonthlyCharges + TotalCharges, data = Telco_Customer_Churn, method = "exp"))


# Plot a Complexity Parameter Table for an Rpart fit for giving a visual representation of the cross-validation results in an rpart object

plotcp(fit)


# Display cp table for giving a visual representation of the cross-validation results in an rpart object
# Relative error = A measure of the uncertainty of measurement compared to the size of the measurement.
# xerror = error estimated from a 10-fold cross validation
# xstd = The standard error of the xerror 
printcp(fit)

# Fit to where which is a number of observations in the root node
fit$where


# Z Survival Variable & Independant Variables fitted
# Where is a number of observations in the root node 
# Median (Median Survival Time) is calculated as the smallest survival time for which the survivor function is less than or equal to 0.5. 
# 0.95LCL = Lower Confidence Limit 95% 
# 0.95UCL = Upper Confidence Limit 95% 
km <- survfit(z~fit$where, data = Telco_Customer_Churn)
km
__________________________________________________________________
# Package ipred (Inproved Predictors)
# Improved predictive models by indirect classification and bagging for classification, regression and survival problems as well as resampling based estimators of prediction error.
# https://cran.r-project.org/web/packages/ipred/ipred.pdf
# https://cran.r-project.org/web/packages/ipred/vignettes/ipred-examples.pdf
# https://www.rdocumentation.org/packages/ipred/versions/0.9-13/topics/rsurv


# Install package ipred
# Use library "ipred, survival, rpart & party"

install.packages("ipred")

library(ipred)
library(survival)


# Brier score for censored data estimated by 10 educations 10-fold cross-validation: 0.2 (Hothorn et al,2002)
# Brier score as a measure for the accuracy of probabilistic forecast. https://www.youtube.com/watch?v=RU1XoKcwdsE
# coob is a logical indicating whether an out-of-bag estimate of the error rate (misclassification error, root mean squared error or Brier score) should be computed.
# nbagg is an integer giving 299 of bootstrap replications.
# Function bagging is Bagging Classification, Regression and Survival Trees

# Brier score in Model rpart (Survival Trees)

set.seed(123) 
gbag <- bagging(Surv(Tenure,Churn) ~., data=Telco_Customer_Churn, 
        model = rpart, nbagg = 100, coob=TRUE)
print(gbag)        
      
gbag <- bagging(Surv(Tenure,Churn) ~., data=Telco_Customer_Churn, 
        model = rpart, nbagg = 200, coob=TRUE)
print(gbag)        

gbag <- bagging(Surv(Tenure,Churn) ~., data=Telco_Customer_Churn, 
        model = rpart, nbagg = 300, coob=TRUE)
print(gbag)        

gbag <- bagging(Surv(Tenure,Churn) ~., data=Telco_Customer_Churn, 
        model = rpart, nbagg = 400, coob=TRUE)
print(gbag)        

gbag <- bagging(Surv(Tenure,Churn) ~., data=Telco_Customer_Churn, 
        model = rpart, nbagg = 500, coob=TRUE)
print(gbag)        

_____________________________________________________________________

# Estimators of Prediction Error, https://cran.r-project.org/web/packages/ipred/ipred.pdf

library(survival)
library(ipred)
library(survival)
library(rpart)


## Model = rpart
# cv (cross-validation 10-fold) = estimator of the misclassification error 
errorest(Churn  ~ ., data=Telco_Customer_Churn, model=rpart,
         estimator = "cv")


# boot(bootstrap) = estimator of the misclassification error
errorest(Churn  ~ ., data=Telco_Customer_Churn, model=rpart,
         estimator = "boot")

