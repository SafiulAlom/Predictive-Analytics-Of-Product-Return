
#*********************************************************************************#
#---------------------Predictive analytics of product return----------------#
#*********************************************************************************#


#remove all grafics and objects
rm(list = ls())
dev.off()


#------------------------Load packages and Dataset--------------------------------#

#function for installing required packages
install.load.packages = function(pack, packagePath){
  all.packages = pack[!(pack %in% installed.packages(lib.loc = packagePath)[, 1])]
  if (length(all.packages)) 
    install.packages(all.packages, lib = packagePath, dependencies = TRUE)
  sapply(pack, require, lib.loc = packagePath, character.only = TRUE)
}


libraryPath = "C:\\Users\\Himel\\OneDrive\\Studium\\R\\Packages"
packages =  c('caret', 'rpart', 'ggplot2', 'mlr', 
              'xgboost', 'parallelMap', 'randomForest', 'dplyr', 'klaR', 
              'lubridate','knitr')
#install and load packages
install.load.packages(packages, libraryPath)

#set working-directory
setwd('C:\\Users\\Himel\\OneDrive\\Studium\\M.Sc. Statistics\\3_Business Analytics&Data Science\\Assignment\\Data')

#load test and training set
Prod.Returns.train = read.csv('BADS_WS1819_known.csv',
                              sep = ',', na.strings = "", stringsAsFactors = FALSE)
Prod.Returns.test = read.csv('BADS_WS1819_unknown.csv',
                             sep = ',', na.strings = "", stringsAsFactors = FALSE)

#combine test and training set
Prod.Returns.train = Prod.Returns.train %>% dplyr::mutate(dataset = "train")
Prod.Returns.test = Prod.Returns.test %>% dplyr::mutate(dataset = "test")
full.Prod.Returns = dplyr::bind_rows(Prod.Returns.train, Prod.Returns.test)



#-----------------------------feature engineering------------------------------#

#check data
str(full.Prod.Returns)


#reducing factor levels of item_size
sort(table(full.Prod.Returns$item_size))
name.item_size = full.Prod.Returns%>% dplyr::select(item_size)%>%table()%>%sort()%>%names()
name.item_size = name.item_size[(length(name.item_size)-15): length(name.item_size)]
full.Prod.Returns$item_size = ifelse(!(full.Prod.Returns$item_size %in% name.item_size),
                                     'other', full.Prod.Returns$item_size) 

#reducing factor levels of item_color
sort(table(full.Prod.Returns$item_color))
name.item_color = full.Prod.Returns%>% dplyr::select(item_color)%>%table()%>%sort()%>%names()
name.item_color = name.item_color[(length(name.item_color)-14): length(name.item_color)]
full.Prod.Returns$item_color = ifelse(!(full.Prod.Returns$item_color %in% name.item_color),
                                      'other', full.Prod.Returns$item_color) 

#reducing factor levels of user_title
full.Prod.Returns$user_title = ifelse(!(full.Prod.Returns$user_title %in% c('Mr', 'Mrs')), 'other', full.Prod.Returns$user_title)

#reducing factor levels of user_state
sort(table(full.Prod.Returns$user_state))
name.user_state = full.Prod.Returns%>% dplyr::select(user_state)%>%table()%>%sort()%>%names()
name.user_state = name.user_state[(length(name.user_state)-4): length(name.user_state)]
full.Prod.Returns$user_state = ifelse(!(full.Prod.Returns$user_state %in% name.user_state),
                                      'other', full.Prod.Returns$user_state) 


#column with NA's
full.Prod.Returns %>% lapply(is.na) %>%sapply(any)
#delivery_date and user_dob contain NA's

#filling all Rows of NA's with dummy value
#missing user date of birth are replaced by mode 
full.Prod.Returns$user_dob[is.na(full.Prod.Returns$user_dob)] = names(table(full.Prod.Returns$user_dob))[table(full.Prod.Returns$user_dob) == max(table(full.Prod.Returns$user_dob))]

#missing delivery_date are replaced by a dummy date
full.Prod.Returns$delivery_date[is.na(full.Prod.Returns$delivery_date)] = "2099-31-12"

#transform character order_item_id to integer order_item_id
ID = substr(full.Prod.Returns$order_item_id, 3,length(full.Prod.Returns))
full.Prod.Returns$order_item_id = as.integer(ID)
str(full.Prod.Returns)

#character variable are transformed to factor
factor_vars = c("item_size", "item_color", "user_title", "user_state")
full.Prod.Returns[factor_vars] <- lapply(full.Prod.Returns[factor_vars], function(x) as.factor(x))

#character date variable are transformed to data type "Date"
date_vars = c("order_date", "delivery_date", "user_dob", "user_reg_date")
full.Prod.Returns[date_vars] <- lapply(full.Prod.Returns[date_vars], function(x) as.Date(x))

#create dummy date
date_vars = unlist(lapply(full.Prod.Returns,  lubridate::is.Date))
date_vars.dummy = paste(names(date_vars[date_vars]), ".dummy", sep = "")
full.Prod.Returns[date_vars.dummy] = lapply(full.Prod.Returns[date_vars], 
                                            function(x) as.integer(as.Date("2019-01-13") - x))
full.Prod.Returns[names(date_vars[date_vars])] = NULL

#return as factor
full.Prod.Returns$return = factor(full.Prod.Returns$return, levels = c(0,1), labels = c('NO', 'YES'))

#item_color and user_state aren't relevant features
full.Prod.Returns$item_color = NULL
full.Prod.Returns$user_state = NULL
#str(full.Prod.Returns)

#dummy feature
full.Prod.Returns.dummy = mlr::createDummyFeatures(full.Prod.Returns, target = 'return')
#chack data
str(full.Prod.Returns.dummy)


#devide data into original known- and unknown set
known = full.Prod.Returns.dummy %>% dplyr::filter(dataset == "train") %>%
  dplyr::select(-dataset)
unknown <- full.Prod.Returns.dummy %>% dplyr::filter(dataset == "test") %>%
  dplyr::select(-c(dataset, return))

#order_item_id is also not a relevant feature
known$order_item_id =NULL 
unknown$order_item_id = NULL


#------------------product return model with boosting--------------------#

#make task
task = mlr::makeClassifTask(data = known, target = 'return', positive  = 'YES')

#make learner
xgb.makeLearner <- mlr::makeLearner("classif.xgboost", predict.type = "prob",
                                    par.vals = list("verbose" = 0,
                                                    "early_stopping_rounds"=20))

# Set tuning parameters
xgb.makeParamSet <- makeParamSet(
  makeNumericParam("eta", lower = 0.01, upper = 0.5), 
  makeIntegerParam("nrounds", lower=80, upper=500), 
  makeIntegerParam("max_depth", lower=2, upper=8),
  makeDiscreteParam("gamma", values = 0),
  makeDiscreteParam("colsample_bytree", values = 1),
  makeDiscreteParam("min_child_weight", values = 1),
  makeDiscreteParam("subsample", values = 1),
  makeNumericParam("lambda", lower = -1, upper = 0, trafo = function(x) 10^x)
)

# random parameter-combination
xgb.makeTuneControlRandom <- mlr::makeTuneControlRandom(maxit=100, tune.threshold = FALSE)

# 5-fold cross-validation
xgb.makeResampleDesc <- makeResampleDesc(method = "RepCV", rep = 3, folds=2, stratify = TRUE)

# set.seed(123) # Set seed for the local random number generator
# # Tune parameters
# xgb.tuneParams <- mlr::tuneParams(xgb.makeLearner, task = task, resampling = xgb.makeResampleDesc,
#                                   par.set = xgb.makeParamSet, control = xgb.makeTuneControlRandom, measures = mlr::auc)
#save the optimal hyperparameters to run the model several times
# saveRDS(xgb.tuneParams, file = "C:\\Users\\Himel\\OneDrive\\Studium\\M.Sc. Statistics\\3_Business Analytics&Data Science\\Assignment\\Objects\\xgb.tuneParams.rds")
# tune = readRDS(file="C:\\Users\\Himel\\OneDrive\\Studium\\M.Sc. Statistics\\3_Business Analytics&Data Science\\Assignment\\Objects\\xgb.tuneParams.rds")

xgb.tuneParams = readRDS(file="C:\\Users\\Himel\\OneDrive\\Studium\\M.Sc. Statistics\\3_Business Analytics&Data Science\\Assignment\\Objects\\xgb.tuneParams.rds")
# Update the learner to the optimal hyperparameters
xgb.makeLearner <- setHyperPars(xgb.makeLearner, par.vals = c(xgb.tuneParams$x, "verbose" = 0))
xgb.makeLearner
model.xgb = mlr::train(xgb.makeLearner, task = task)
yhat <- predict(model.xgb, newdata = unknown)


#create a prediction file
prediction = Prod.Returns.test %>% dplyr::select("order_item_id" = order_item_id) %>% 
  dplyr::mutate("return" = yhat$data$prob.YES)

# write.csv(prediction, file =
#             "C:\\Users\\Himel\\OneDrive\\Studium\\M.Sc. Statistics\\3_Business Analytics&Data Science\\Assignment\\result3\\prediction4.csv", row.names = FALSE)




