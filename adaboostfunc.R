adaboostfunc = function(formula, data, M = 100){
  formula = as.formula(formula)
  classcolumn = data[, as.character(formula[[2]])]
  n = length(data[, 1])
  omega = rep(1/n, n) # step 1 of boosting adaboost algorithm on PPT, initial weights omega
  prepredict = data.frame(rep(0, n)) # used for keeping predicting values by single trees
  classlevels = nlevels(classcolumn)
  treemodels = list() # storing stats models (here are trees) for predicting on test sets
  treeprop = rep(0, M) # proportions of trees we are going to build
  for (m in 1:M) {
    omega <<- omega
    fit = rpart(formula = formula, data = data, weights = omega) # fit a tree
    flearn = predict(fit, data = data, type = "class") # predicting results by this tree
    classmis = as.numeric(classcolumn != flearn) # construct 0-1 vector which indicates misclassifications
    eps = sum(omega * classmis)/sum(omega) # step 2(b) of boosting adaboost algorithm on PPT, compute epsilon
    alpha = log((1 - eps)/eps) # step 2(b) of boosting adaboost algorithm on PPT, compute proportion of trees
    omega = omega * exp(alpha * classmis) # step 2(c) of boosting adaboost algorithm on PPT, update weights
    alter = 0.005
    if (eps >= 0.5) { omega = rep(1/n, n); alpha = 0 } # epsilon too high
    if (eps == 0) { omega = rep(1/n, n); alpha = 10 } # epsilon too low
    treemodels[[m]] = fit
    treeprop[m] = alpha # proportion for this tree
    if (m == 1) { prepredict = flearn }
    else { prepredict = data.frame(prepredict, flearn) } # construct prediction matrix by those trees
  } # the end of fitting all M trees  
  classscore = array(0, c(n, classlevels)) # classification results for every sample by every classifier
  for (i in 1:classlevels) {
    classscore[, i] = matrix(as.numeric(prepredict == levels(classcolumn)[i]), nrow = n) %*% as.vector(treeprop)
  } # loop for every class, classscore[, i] contains all scores for each class of every sample
  classprob = classscore/apply(classscore, 1, sum) # every row contains probabilities for different classes
  classpredict = rep("O", n) 
  classnumber = apply(classscore, 1, FUN = which.max)
  classpredict = as.factor(levels(classcolumn)[classnumber]) # final prediction results
  result = list(formula = formula, trees = treemodels, treeproportions = treeprop, 
             classscores = classscore, classprobability = classprob, classpredict = classpredict)
  return(result) }