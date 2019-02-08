import pandas as pd
import os
import pickle
import argparse
import numpy as np
from models import *
from metrics import *
parser = argparse.ArgumentParser(description='challenger model and metrics building')
parser.add_argument('x',metavar='xvar',type=str, help='x vars csv file')
parser.add_argument('y',metavar='yvar',type=str, help='y var csv file')
parser.add_argument('output',nargs='?', default=os.getcwd(), help='fire policy pif count assigned to the agent')
#parser.add_argument('model',nargs='?',help='benchmark model pickle file')
parser.add_argument('cv',nargs='?',type = int,default = 5, help='benchmark model pickle file')
args = parser.parse_args()


def read_xy(X=args.x,Y=args.y):
    X_train = pd.read_csv(X)
    Y_train = np.ravel(pd.read_csv(Y,header = None))
    return X_train,Y_train


'''
Cross Validation and Record the socring metrics
'''

def cross_validation(X,Y,modellist,splits = args.cv):
    print ('running {} cross validation'.format(args.cv))
    kfold = KFold(n_splits=splits, random_state=22) # k=10, split the data into 10 equal parts
    res = {}
    for i in modellist:
        model = i
        scores = cross_validate(model,X,Y, cv = kfold,scoring = scoring,return_train_score= False)
        print ('running model {}, getting score {} '.format(model,scores))
        for k in list(scores.keys()):
            if k in res.keys():
                res[k].append(scores[k])
            else:
                res[k] = [scores[k]]
    return res


'''
    Append the mean values of scoring metrics from cross valiation for each model
'''
def mean_metrics_compute(result,classifiers):
    avgDict = {}
    for k,v in result.items():
        # v is the list of grades for student k
        avgDict[k] = np.mean(v,axis = 1)
    metrics_result=pd.DataFrame(avgDict,index=classifiers)       
    print (metrics_result)
    metrics_result.to_csv(args.output + '/metrics_comparison.csv',index = None)
    print ('metrics_result saved to {}'.format(args.output + '/metrics_comparison.csv'))


def gridsearch_fit(X,Y,model):
    model.fit(X,Y)
    print(model.best_score_)
    print(model.best_estimator_)
    return model.best_estimator_


def create_dir(dir = args.output):
    if not os.path.isdir(dir):
        os.makedirs(dir)


def main(args):

    print (args)
    create_dir()
    X_train,Y_train = read_xy()
    
    ensemble_lin_rbf.fit(X_train,Y_train)
    adaboost = gridsearch_fit(X_train,Y_train,gd)
    rf = gridsearch_fit(X_train,Y_train,rfgd)


    challenger_models_names=['Linear Svm','Logistic Regression','Ada Boosting','Random Forest','Ensemble']
    challenger_models=[svm.SVC(kernel='linear'),LogisticRegression(),adaboost,rf,ensemble_lin_rbf]
    classifer_dict =dict(zip(challenger_models_names,challenger_models))
    classifiers = list(classifer_dict.keys())
    models = list(classifer_dict.values())


    res = cross_validation(X_train,Y_train,models,args.cv)
    mean_metrics_compute(res,classifiers)

    print ('running metrics boxplot...')
    metrics_boxplot(res,args.output)

    print ('running confusion matrix...')
    confusion_matrix_plot(args.output,X_train,Y_train,5)

if __name__ == '__main__':
    main(args)


