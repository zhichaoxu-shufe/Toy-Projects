import xgboost as xgb

# readin data
dtrain = xgb.DMatrix("data/agaricus.txt.train")
dtest = xgb.DMatrix("data/agaricus.txt.test")
# specify parameters via map
parameters = {'max_depth':2, 'eta':1, 'silent':1, 'objective': 'binary: logistic'}
num_round = 2
bst = xgb.train(param, dtrain, num_round)
# make prediction
preds = bst.predict(dtest)