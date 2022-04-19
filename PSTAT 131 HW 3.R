library(ggplot2)
library(tidyverse)
library(tidymodels)
library(corrplot)
library(ggthemes)
tidymodels_prefer()
library(ISLR)
library(yardstick)
library(corrr)
library(discrim)
library(poissonreg)
library(klaR)
titanic <- read.csv("C:\\titanic.csv")
view(titanic)
set.seed(4857)
# change survived and pclass to factos
## reorder survived factor so that "Yes" is the first level 
levels(titanic$survived)
titanic$survived <- factor(titanic$survived, levels = c("Yes", "No"))
levels(titanic$pclass)
titanic$pclass <- factor(titanic$pclass)

## QUESTION 1
# split data, stratifying on survived 
# choose proportions & confirm have right # of observations
# note any potential issues, such as missing data
titanic_split <- initial_split(titanic, prop = 0.7,
                               strata = survived)
titanic_train <- training(titanic_split)
titanic_test <- testing(titanic_split)
# 623 observations
view(titanic_train)
#268 observations
view(titanic_test)

## QUESTION 2
# use training set, get distribution of outcome variable survived 
titanic_train %>% ggplot(aes(x = survived)) + geom_bar()

## QUESTION 3
# create correlation matrix of all cont variables
# describe patterns / correlations
titanic_cont <- select_if(titanic_train, is.numeric)
cor_titanic <- titanic_cont %>% correlate()
rplot(cor_titanic)

## QUESTION 4
# create a recipe for the training data
titanic_recipe <- recipe(survived ~ pclass + sex +
                           age + sib_sp + parch + fare, 
                         data = titanic_train) %>%
  step_impute_linear(age) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_interact(terms =~starts_with("sex"):fare) %>%
  step_interact(terms=~age:fare) 
summary(titanic_recipe)

## QUESTION 5
# create a logistic regression model for classification 
# using "glm" engine
# create a workflow, add model and recipre 
# use fit to aplly workflow to training data 
log_reg <- logistic_reg() %>%
  set_engine("glm") %>%
  set_mode("classification")

log_wkflow <- workflow() %>%
  add_model(log_reg) %>%
  add_recipe(titanic_recipe)

log_fit <- fit(log_wkflow, titanic_train)

## QUESTION 6
# LDA model for classification suing "MASS" engine
lda_mod <- discrim_linear() %>%
  set_mode("classification") %>%
  set_engine("MASS")

lda_wkflow <- workflow() %>%
  add_model(lda_mod) %>%
  add_recipe(titanic_recipe)

lda_fit <- fit(lda_wkflow, titanic_train)
lda_fit

## QUESTION 7
# QDA model for classification using "MASS" engine
qda_mod <- discrim_quad() %>%
  set_mode("classification") %>%
  set_engine("MASS")

qda_wkflow <- workflow() %>%
  add_model(qda_mod) %>%
  add_recipe(titanic_recipe)

qda_fit <- fit(qda_wkflow, titanic_train)
qda_fit

## QUESTION 8
# Naive Bayes model for classification using
# "klaR" engine & set usekernel = FALSE
nb_mod <- naive_Bayes() %>%
  set_mode("classification") %>%
  set_engine("klaR") %>%
  set_args(usekernel = FALSE)

nb_wkflow <- workflow() %>%
  add_model(nb_mod) %>%
  add_recipe(titanic_recipe)

nb_fit <- fit(nb_wkflow, titanic_train)

## QUESTION 9
# log fit: .812
log_fit_results <- predict(log_fit, new_data = titanic_train %>%
                             select(-survived))
log_fit_results <- bind_cols(log_fit_results,
                             titanic_train %>%
                               select(survived))
log_fit_acc <- augment(log_fit, new_data = titanic_train) %>%
  accuracy(truth = survived, estimate = .pred_class)
log_fit_acc
# lda fit: .798
lda_fit_res <- predict(lda_fit, new_data = titanic_train %>%
                         select(-survived))
lda_fit_res <- bind_cols(lda_fit_res,
                         titanic_train %>%
                           select(survived))

lda_fit_acc <- augment(lda_fit, new_data = titanic_train) %>%
  accuracy(truth = survived, estimate = .pred_class)
lda_fit_acc    
# qda fit: .791
qda_fit_res <- predict(qda_fit, new_data = titanic_train %>%
                         select(-survived))
qda_fit_res <- bind_cols(qda_fit_res,
                         titanic_train %>%
                           select(survived))

qda_fit_acc <- augment(qda_fit, new_data = titanic_train) %>%
  accuracy(truth = survived, estimate = .pred_class)
qda_fit_acc  
# nb fit: .783
nb_fit_res <- predict(nb_fit, new_data = titanic_train %>%
                         select(-survived))
nb_fit_res <- bind_cols(nb_fit_res,
                         titanic_train %>%
                           select(survived))
nb_fit_res %>% head()
nb_fit_acc <- augment(nb_fit, new_data = titanic_train) %>%
  accuracy(truth = survived, estimate = .pred_class)
nb_fit_acc  

## QUESTION 10
# fit log model to the testing data
# give accurace on testing data
# create confusion matric with testing data
# plot an ROC curve and calc area under it (AUC)

predict(log_fit, new_data = titanic_test, type = "prob")
log_fit_res_test <- predict(log_fit, new_data = titanic_test %>%
                             select(-survived))
log_fit_res_test <- bind_cols(log_fit_res_test,
                             titanic_test %>%
                               select(survived))
log_fit_acc_test <- augment(log_fit, new_data = titanic_test) %>%
  accuracy(truth = survived, estimate = .pred_class)
log_fit_acc_test
augment(log_fit, new_data = titanic_test) %>%
  conf_mat(truth = survived, estimate = .pred_class) %>%
  autoplot(type = "heatmap")

augment(log_fit, new_data = titanic_test) %>%
  roc_curve(survived, .pred_Yes) %>%
  autoplot()

augment(log_fit, new_data = titanic_test) %>%
  roc_auc(survived, .pred_Yes)