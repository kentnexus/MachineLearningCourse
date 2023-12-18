setwd("C:/Users/k_pas/OneDrive - Douglas College/School Stuff/Summer '23/CSIS 3360/Project/p2_tyrone_pastoril/data")

install.packages("tidyr","VGAM","MASS")

library(tidyr)
library(reshape2)
library(dplyr)
library(caret)
library(wrapr)
library(class)
library(e1071)
library(rpart)
library(randomForest)
library(VGAM)
library(MASS)

data <- read.csv("wdi_data.csv")

summary(data)

#data exploration
#removing rows with NA

exp_data <- na.omit(data)

#checking the correlation between variables
corr_matrix <- round(cor(exp_data[,5:20]),2)
print(corr_matrix)

melted_cormat <- melt(corr_matrix)

ggplot(data = melted_cormat, aes(x = Var1, y = Var2, fill=value)) +
  geom_tile() +
  theme(axis.text.x = element_text(angle = 60, hjust = 1)) +
  scale_fill_gradient(low="black",high="cyan") +
  coord_fixed() +
  ggtitle("Correlation Heatmap")

#GNI per capita has a positive relationship with the six dimensions
#classifying the income group based on GNI per CAPITA
#source: https://blogs.worldbank.org/opendata/new-world-bank-country-classifications-income-level-2022-2023


cl_data <- data %>% drop_na(CTRLCORR,GOVEFF,PSAVT,REGQUA,RULAW,VOIACC,GNICAPITA)

cl_data$INCOMEGRP <- ifelse(cl_data$GNICAPITA<1085,"LOW",
                              ifelse(cl_data$GNICAPITA<4255,"LOWER-MIDDLE",
                                     ifelse(cl_data$GNICAPITA<13205,"UPPER-MIDDLE","HIGH")))
cl_data$INCOMEGRP <- as.factor(cl_data$INCOMEGRP)
summary(cl_data)

#saving pre-processed data to csv
write.csv(cl_data,("cl_wdi_data.csv"))

#final data to be processed
pr_data <- cl_data %>% dplyr::select(c("TIME","CNTRYNM","CTRLCORR","GOVEFF","PSAVT","REGQUA","RULAW","VOIACC","INCOMEGRP"))

#generating a generic formula for the models
y = "INCOMEGRP"
x = c("CTRLCORR","GOVEFF","PSAVT","REGQUA","RULAW","VOIACC")

library(wrapr)
fmla <- mk_formula(y,x)

#generating the data distribution plot
featurePlot(pr_data[,x],
            pr_data$INCOMEGRP,
            plot="density",
            pch = "|",
            auto.key = list(columns = 4))

#train-test split data with 80% for the train set
set.seed(123)

training_obs <- pr_data$INCOMEGRP %>% createDataPartition(p = 0.8, list = FALSE)
train <- pr_data[training_obs,]
test <- pr_data[-training_obs,]

#model evaluation
method <- c()
test_accuracy <- c()
kappa <- c()
cod <- c()

#defining training control as cross-validation and value of K equal to 10
# train_control <- trainControl(method = "cv", number = 5)

#Linear Classification
linear <- vglm(fmla, data=train, family=multinomial)

test_prob <- predict(linear, newdata = test, type="response")

test$pred <- apply(test_prob, 1, which.max)

test$pred[which(test$pred=="1")] <- "HIGH"
test$pred[which(test$pred=="2")] <- "UPPER-MIDDLE"
test$pred[which(test$pred=="3")] <- "LOWER-MIDDLE"
test$pred[which(test$pred=="4")] <- "LOW"

test$pred <- as.factor(test$pred)

cfm_test <- confusionMatrix(data = as.factor(test$pred),
                            reference = as.factor(test$INCOMEGRP),
                            mode='everything')

method <- append(method,"Linear Classification")
test_accuracy <- append(test_accuracy,round(cfm_test$overall["Accuracy"],4))
kappa <- append(kappa,round(cfm_test$overall["Kappa"],4))
cod <- append(cod,round(cfm_test$overall["Kappa"]^2,4))

#Linear Discriminant Analysis
lda <- lda(fmla, data=train)

test$pred <- predict(lda, newdata = test)$class

cfm_test <- confusionMatrix(data = as.factor(test$pred),
                            reference = as.factor(test$INCOMEGRP),
                            mode='everything')

method <- append(method,"Linear Discriminant Analysis")
test_accuracy <- append(test_accuracy,round(cfm_test$overall["Accuracy"],4))
kappa <- append(kappa,round(cfm_test$overall["Kappa"],4))
cod <- append(cod,round(cfm_test$overall["Kappa"]^2,4))

#Naive Bayesian
nb <- naiveBayes(fmla, data=train)

test$pred <- predict(nb, newdata = test)

cfm_test <- confusionMatrix(data = as.factor(test$pred),
                            reference = as.factor(test$INCOMEGRP),
                            mode='everything')

method <- append(method,"Naive Bayes")
test_accuracy <- append(test_accuracy,round(cfm_test$overall["Accuracy"],4))
kappa <- append(kappa,round(cfm_test$overall["Kappa"],4))
cod <- append(cod,round(cfm_test$overall["Kappa"]^2,4))

#Decision Tree
tree_model <- rpart(fmla, method="class", data=train)

test$pred <- predict(tree_model, newdata = test, type = "class")

cfm_test <- confusionMatrix(data = as.factor(test$pred),
                            reference = as.factor(test$INCOMEGRP),
                            mode='everything')

method <- append(method,"Decision Tree")
test_accuracy <- append(test_accuracy,round(cfm_test$overall["Accuracy"],4))
kappa <- append(kappa,round(cfm_test$overall["Kappa"],4))
cod <- append(cod,round(cfm_test$overall["Kappa"]^2,4))

#Decision Tree with complexity .001
tree_model <- rpart(fmla, method="class", data=train, control = rpart.control(cp=0.001))

test$pred <- predict(tree_model, newdata = test, type = "class")

cfm_test <- confusionMatrix(data = as.factor(test$pred),
                            reference = as.factor(test$INCOMEGRP),
                            mode='everything')

method <- append(method,"Decision Tree with 0.001 Complexity")
test_accuracy <- append(test_accuracy,round(cfm_test$overall["Accuracy"],4))
kappa <- append(kappa,round(cfm_test$overall["Kappa"],4))
cod <- append(cod,round(cfm_test$overall["Kappa"]^2,4))

#Random Forest
rf_model <- randomForest(as.factor(INCOMEGRP)~ GOVEFF + PSAVT + REGQUA + RULAW + VOIACC + CTRLCORR, data=train)

test$pred <- predict(rf_model, newdata = test, type = "response")

cfm_test <- confusionMatrix(data = as.factor(test$pred),
                            reference = as.factor(test$INCOMEGRP),
                            mode='everything')

method <- append(method,"Random Forest")
test_accuracy <- append(test_accuracy,round(cfm_test$overall["Accuracy"],4))
kappa <- append(kappa,round(cfm_test$overall["Kappa"],4))
cod <- append(cod,round(cfm_test$overall["Kappa"]^2,4))

#feature-scaling for KNN and SVM
pmatrix <- scale(pr_data[,x])
pcenter <- attr(pmatrix, "scaled:center")
pscale <- attr(pmatrix, "scaled:scale")
rm_scales <- function(scaled_matrix) {
  attr(scaled_matrix, "scaled:center") <- NULL
  attr(scaled_matrix, "scaled:scale") <- NULL
  scaled_matrix}
pmatrix <- rm_scales(pmatrix)

pmatrix <- data.frame(pmatrix)
pmatrix$INCOMEGRP <- pr_data$INCOMEGRP

#standard normalization train-test
train_pmatrix <- pmatrix[-training_obs,]
test_pmatrix <- pmatrix[-training_obs,]

#knn model k = 3 using pmatrix
k <- 3
knn_model_test <- knn(train=train_pmatrix[,x], test=test_pmatrix[,x], cl=train_pmatrix[,y], k=k)

cfm_test <- confusionMatrix(data = as.factor(knn_model_test),
                            reference = as.factor(test_pmatrix$INCOMEGRP),
                            mode='everything')

method <- append(method, "KNN(k=3) with Standard Normalization Scaling")
test_accuracy <- append(test_accuracy,round((cfm_test$overall["Accuracy"]),4))
kappa <- append(kappa,round(cfm_test$overall["Kappa"],4))
cod <- append(cod,round(cfm_test$overall["Kappa"]^2,4))

#Linear SVM with scaled data
svm_model_radial_c50 <- svm(fmla, data=train_pmatrix, kernel="linear", cost=50, scaling=FALSE)

test_pmatrix$pred <- predict(svm_model_radial_c50, newdata = test_pmatrix)

cfm_test <- confusionMatrix(data = as.factor(test_pmatrix$pred),
                            reference = as.factor(test_pmatrix$INCOMEGRP),
                            mode='everything')

method <- append(method, "Linear SVM with Standard Normalization Scaling, c=50")
test_accuracy <- append(test_accuracy,round((cfm_test$overall["Accuracy"]),4))
kappa <- append(kappa,round(cfm_test$overall["Kappa"],4))
cod <- append(cod,round(cfm_test$overall["Kappa"]^2,4))

#Nonlinear SVM with scaled data
svm_model_radial_c50 <- svm(fmla, data=train_pmatrix, kernel="radial", cost=50, scaling=FALSE)

test_pmatrix$pred <- predict(svm_model_radial_c50, newdata = test_pmatrix)

cfm_test <- confusionMatrix(data = as.factor(test_pmatrix$pred),
                            reference = as.factor(test_pmatrix$INCOMEGRP),
                            mode='everything')

method <- append(method, "SVM with RBF with Standard Normalization Scaling, c=50")
test_accuracy <- append(test_accuracy,round((cfm_test$overall["Accuracy"]),4))
kappa <- append(kappa,round(cfm_test$overall["Kappa"],4))
cod <- append(cod,round(cfm_test$overall["Kappa"]^2,4))


#####results
results <- data.frame(method,test_accuracy,kappa,cod)
results

bestNo <- which.max(results$test_accuracy)

writeLines(paste("\tBest method:",results$method[bestNo],"\n",
                 "\tBest accuracy:",results$test_accuracy[bestNo]*100,"%","\n",
                 "\tBest kappa:",results$kappa[bestNo],"\n",
                 "\tBest Coefficient of Determination:",results$cod[bestNo]))
