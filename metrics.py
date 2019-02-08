from sklearn.metrics.scorer import make_scorer
from sklearn.metrics import roc_curve,roc_auc_score,f1_score,confusion_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_predict
from models import svm,rfgd,gd,ensemble_lin_rbf,LogisticRegression
import seaborn as sns
import pandas as pd
import warnings

warnings.filterwarnings("ignore")
plt.style.use('fivethirtyeight')


scoring = {'acc': 'accuracy',
           'prec': 'precision_macro',
           'rec': 'recall_macro',
           'f1':make_scorer(f1_score),
           'auc': make_scorer(roc_auc_score)
          }



def metrics_boxplot(res,outputdir):

	f,ax=plt.subplots(3,3,figsize=(12,10))
	box=pd.DataFrame(res['fit_time'],index=['lsvm','logistic','aboost','rforest','ensemble'])
	box.T.boxplot(ax=ax[0,0])
	ax[0,0].set_title('Fitting Time')
	box=pd.DataFrame(res['score_time'],index=['lsvm','logistic','aboost','rforest','ensemble'])
	box.T.boxplot(ax=ax[0,1])
	ax[0,1].set_title('Score Time')
	box=pd.DataFrame(res['test_acc'],index=['lsvm','logistic','aboost','rforest','ensemble'])
	box.T.boxplot(ax=ax[0,2])
	ax[0,2].set_title('Accuracy')
	box=pd.DataFrame(res['test_auc'],index=['lsvm','logistic','aboost','rforest','ensemble'])
	box.T.boxplot(ax=ax[1,0])
	ax[1,0].set_title('AUC_ROC')
	box=pd.DataFrame(res['test_f1'],index=['lsvm','logistic','aboost','rforest','ensemble'])
	box.T.boxplot(ax=ax[1,1])
	ax[1,1].set_title('F1 Score')
	box=pd.DataFrame(res['test_prec'],index=['lsvm','logistic','aboost','rforest','ensemble'])
	box.T.boxplot(ax=ax[1,2])
	ax[1,2].set_title('Precision')
	box=pd.DataFrame(res['test_rec'],index=['lsvm','logistic','aboost','rforest','ensemble'])
	box.T.boxplot(ax=ax[2,0])
	ax[2,0].set_title('Recall')

	f.savefig(outputdir+'/boxplot.png')
	print ('box plot is saved to {}'.format(outputdir + '/boxplot.png'))
	plt.close(f)


def confusion_matrix_plot(outputdir,X_train,Y_train,cv=5):
	f,ax=plt.subplots(2,3,figsize=(12,10))
	y_pred = cross_val_predict(svm.SVC(kernel='linear'),X_train,Y_train,cv=cv)
	sns.heatmap(confusion_matrix(Y_train,y_pred),ax=ax[0,0],annot=True,fmt='2.0f')
	ax[0,0].set_title('Matrix for Linear-SVM')
	y_pred = cross_val_predict(rfgd.best_estimator_,X_train,Y_train,cv=cv)
	sns.heatmap(confusion_matrix(Y_train,y_pred),ax=ax[0,1],annot=True,fmt='2.0f')
	ax[0,1].set_title('Matrix for Random-Forests')
	y_pred = cross_val_predict(LogisticRegression(),X_train,Y_train,cv=cv)
	sns.heatmap(confusion_matrix(Y_train,y_pred),ax=ax[0,2],annot=True,fmt='2.0f')
	ax[0,2].set_title('Matrix for Logistic Regression')
	y_pred = cross_val_predict(gd.best_estimator_,X_train,Y_train,cv=cv)
	sns.heatmap(confusion_matrix(Y_train,y_pred),ax=ax[1,0],annot=True,fmt='2.0f')
	ax[1,0].set_title('Matrix for Ada Boosting')
	y_pred = cross_val_predict(ensemble_lin_rbf,X_train,Y_train,cv=cv)
	sns.heatmap(confusion_matrix(Y_train,y_pred),ax=ax[1,1],annot=True,fmt='2.0f')
	ax[1,1].set_title('Matrix for Ensemble-classfier')
	plt.subplots_adjust(hspace=0.2,wspace=0.2)
	f.savefig(outputdir+'/confusion_plot.png')   # save the figure to file
	plt.close(f)
	print ('confusion matrix plot is saved to {}'.format(outputdir + '/confusion_plot.png'))
