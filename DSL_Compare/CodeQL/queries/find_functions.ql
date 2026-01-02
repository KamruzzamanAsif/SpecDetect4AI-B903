import python


predicate isTargetName(string name) {
  name = "KMeans" or
  name = "DBSCAN" or
  name = "AgglomerativeClustering" or
  name = "RandomForestClassifier" or
  name = "RandomForestRegressor" or
  name = "GradientBoostingClassifier" or
  name = "AdaBoostClassifier" or
  name = "LogisticRegression" or
  name = "LinearRegression" or
  name = "Lasso" or
  name = "Ridge" or
  name = "SVC" or
  name = "SVR" or
  name = "DecisionTreeClassifier" or
  name = "DecisionTreeRegressor" or
  name = "MLPClassifier" or
  name = "MLPRegressor" or
  // PyTorch Optimizers
  name = "SGD" or
  name = "Adagrad" or
  name = "Adadelta" or
  name = "Adamax" or
  name = "RMSprop" or
  name = "Net" or
  // TensorFlow Optimizers/Layers
  name = "Adam" or
  name = "Ftrl" or
  name = "Nadam" or
  name = "Dense" or
  name = "Conv2D" or
  name = "LSTM" or
  // XGBoost
  name = "XGBClassifier" or
  name = "XGBRegressor" or
  // LightGBM
  name = "LGBMClassifier" or
  name = "LGBMRegressor" or
  name = "Sequential"
}

predicate isTargetCall(Call call, string funcName) {
 
  exists(Name n | call.getFunc() = n and funcName = n.getId()) 
  or 
  exists(Attribute attr | call.getFunc() = attr and funcName = attr.getAttr())
}

from Call call, string nomFunc
where 
  isTargetName(nomFunc)                      
  and isTargetCall(call, nomFunc)            
  and not exists(call.getAPositionalArg())   
  and not exists(call.getANamedArg())        
select call, call.getLocation(), "Call to " + nomFunc + " without hyperparameters"