
#*********************************************************************************#
#---------------------Predictive analytics of product return----------------#
#*********************************************************************************#

#remove all grafics and objects
rm(list = ls())

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
              'lubridate','knitr', 'GA')
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

#Summary
summary(full.Prod.Returns)

#column with NA's
full.Prod.Returns %>% lapply(is.na) %>%sapply(any)
#delivery_date and user_dob contain NA's

#filling all Rows of NA's with dummy value
#missing user date of birth are replaced by mode 
full.Prod.Returns$user_dob[is.na(full.Prod.Returns$user_dob)] = names(table(full.Prod.Returns$user_dob))[table(full.Prod.Returns$user_dob) == max(table(full.Prod.Returns$user_dob))]

#missing delivery_date are replaced by a dummy date
full.Prod.Returns$delivery_date[is.na(full.Prod.Returns$delivery_date)] = as.character(Sys.Date())

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


#----------------------Logit + GA------------------------------------#
#item_size.XXL, user_title.Mr 1-0 dummy => one column of every dummy coding are removed
known.clean = known %>% dplyr::select(-c(item_size.XXL, user_title.Mr ))
known.clean.x = known.clean%>%dplyr::select(-return)

known.clean.x = melt(cor(known.clean.x))
known.clean.x %>% dplyr::filter((value > 0.7 | value < - 0.7) & X1 != X2 )
#=> item_id and order_date.dummy are highly correlated, one of them has to be removed
known.clean = known.clean %>% dplyr::select(-c(item_id))

#create test and training set
set.seed(12345)
idx.train <- caret::createDataPartition(y = known.clean$return, p = 0.6, list = FALSE)
tr <- known.clean[idx.train, ]
ts <-  known.clean[-idx.train, ] 

standard.model <- caret::preProcess(tr, method = c("center", "scale"))
tr <- predict(standard.model, newdata = tr)
ts <- predict(standard.model, newdata = ts)

#standard logit model
logit <- glm(return~., data = tr, family = binomial(link = "logit"))

#model matrix 
train.x <- model.matrix(return~., tr)
test.x <- model.matrix(return~., ts)
train.y.numeric = (as.numeric(tr$return)-1)
test.y.numeric = (as.numeric(ts$return)-1)

predict.Prob <- function(x, beta){
  ## Calculate logit prediction
  yhat <- x %*% beta
  # Logit transformation
  prob <- exp(yhat) / (1 + exp(yhat))
  return(prob)
}

# function of decision profit
decision.profit = function(y, x, beta, itemvalue){
  
  yhat = predict.Prob(x, beta)
  Cost.TP = 0
  Cost.TN = 0
  Cost.FP = -0.5*itemvalue
  cost.FN = -0.5*5*(3 + 0.1*itemvalue)
  N = length(y)
  cost.function = (1/N)*sum(y*(yhat*Cost.TP + (1-yhat)*cost.FN) + 
                              (1-y)*(yhat*Cost.FP + (1 - yhat)*Cost.TN))
  return(cost.function)
  
}

#Fitness function
Fitness.Func <- function(beta, x, y, metric, itemvalue){
  #prob <- predict.Prob(x, beta)
  fitnessScore <- do.call('metric', list(y = y, x, beta = beta, itemvalue = itemvalue ))
  return(fitnessScore)
}


#GA Algorithm
lower.bound = min(as.numeric(logit$coefficients)) - 10
upper.bound = max(as.numeric(logit$coefficients)) + 10
ga.logit =  ga(type = "real-valued", 
               fitness = Fitness.Func, 
               x = train.x, y = train.y.numeric , metric = decision.profit, 
               itemvalue = tr$item_price,
               lower = rep(lower.bound, ncol(train.x)), upper = rep(upper.bound, ncol(train.x)),
               popSize = 70, pcrossover = 0.8, pmutation = 0.01, elitism = 0.01,
               maxiter = 500, run = 100,
               parallel = FALSE,
               monitor = FALSE 
)


#identified coefficients by using GA
coef.ga.logit = ga.logit@solution[1,]
names(coef.ga.logit) = colnames(train.x)
coef.logit = logit$coefficients

coef.mat = data.frame(Coefficient = names(coef.ga.logit),
           Estimete_GA = as.numeric(coef.ga.logit),
           estimete_standard = as.numeric(logit$coefficients))
kable(head(coef.mat, n = 6), caption = "Table of Estimates of first 6 coefficients")


#decision cost of training set
train.standard.profit = decision.profit(y = train.y.numeric , x = train.x,
                                  beta = logit$coefficients, itemvalue = tr$item_price)
train.ga.profit = decision.profit(y = train.y.numeric , x = train.x,
                                  beta = ga.logit@solution[1,], itemvalue = tr$item_price)
#decision cost of test set
test.standard.profit = decision.profit(y = test.y.numeric , x = test.x,
                                       beta = logit$coefficients, itemvalue = ts$item_price)
test.ga.profit= decision.profit(y = test.y.numeric , x = test.x,
                                beta = ga.logit@solution[1,], itemvalue = ts$item_price)
#profit table
profit = data.frame(train.ga.profit, train.standard.profit, test.ga.profit, test.standard.profit)


