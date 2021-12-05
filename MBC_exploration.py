import sys
import os
import re
import math
import numpy as np
import matplotlib.pyplot as plt

from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import learning_curve
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
from UB_BB import extract_data
from sklearn.datasets import load_files

class Best_Configuration:

	def __init__(self,training_path,evaluation_path,output):
		self.training = load_files(training_path)
		self.testing = load_files(evaluation_path)
		with open(output,"w") as output_file:
			naive_bayes_pipeline = Pipeline([('vect', CountVectorizer(decode_error='ignore',stop_words='english')),('feature_selection', SelectFromModel(LinearSVC())),('clf', MultinomialNB())])
			naive_bayes_pipeline.fit(self.training.data, self.training.target)
			nb_predicted = naive_bayes_pipeline.predict(self.testing.data)
			nb_precision = metrics.precision_score(self.testing.target, nb_predicted, average='macro')
			nb_recall = metrics.recall_score(self.testing.target, nb_predicted, average='macro')
			nb_f1 = metrics.f1_score(self.testing.target, nb_predicted, average='macro')
			output_file.write(f"NB,MBC,{nb_precision},{nb_recall},{nb_f1}\n")
			print("Naive bayes")
			print("F1:",nb_f1)
			print("accuracy:", metrics.accuracy_score(self.testing.target, nb_predicted))
			print("precision:",nb_precision)
			print("recall:",nb_recall)

			svm_pipeline = Pipeline([('vect', CountVectorizer(decode_error='ignore',stop_words='english')),('tfidf', TfidfTransformer()),('clf', SGDClassifier(loss='hinge',penalty='l2',alpha=1e-3,random_state=42,max_iter=5,tol=None))])
			svm_pipeline.fit(self.training.data, self.training.target)
			svm_predicted = svm_pipeline.predict(self.testing.data)
			svm_precision = metrics.precision_score(self.testing.target, svm_predicted, average='macro')
			svm_recall = metrics.recall_score(self.testing.target, svm_predicted, average='macro')
			svm_f1 = metrics.f1_score(self.testing.target, svm_predicted, average='macro')
			output_file.write(f"SVM,MBC,{svm_precision},{svm_recall},{svm_f1}\n")
			print("SVM")
			print("F1:",svm_f1)
			print("accuracy:", metrics.accuracy_score(self.testing.target, svm_predicted))
			print("precision:",svm_precision)
			print("recall:",svm_recall)

			logistic_pipeline = Pipeline([('vect', CountVectorizer(decode_error='ignore',stop_words='english')),('tfidf', TfidfTransformer()),('logistic', LogisticRegression())])
			logistic_pipeline.fit(self.training.data, self.training.target)
			logistic_predicted = logistic_pipeline.predict(self.testing.data)
			logistic_precision = metrics.precision_score(self.testing.target, logistic_predicted, average='macro')
			logistic_recall = metrics.recall_score(self.testing.target, logistic_predicted, average='macro')
			logistic_f1 = metrics.f1_score(self.testing.target, logistic_predicted, average='macro')
			output_file.write(f"LG,MBC,{logistic_precision},{logistic_recall},{logistic_f1}\n")
			print("Logistic")
			print("F1:",logistic_f1)
			print("accuracy:", metrics.accuracy_score(self.testing.target, logistic_predicted))
			print("precision:",logistic_precision)
			print("recall:",logistic_recall)

			random_forest_pipeline = Pipeline([('vect', CountVectorizer(decode_error='ignore',stop_words='english')),('tfidf', TfidfTransformer()),('Randomforest', RandomForestClassifier())])
			random_forest_pipeline.fit(self.training.data, self.training.target)
			rf_predicted = random_forest_pipeline.predict(self.testing.data)
			rf_precision = metrics.precision_score(self.testing.target, rf_predicted, average='macro')
			rf_recall = metrics.recall_score(self.testing.target, rf_predicted, average='macro')
			rf_f1 = metrics.f1_score(self.testing.target, rf_predicted, average='macro')
			output_file.write(f"RF,MBC,{rf_precision},{rf_recall},{rf_f1}")
			print("Random Forest")
			print("F1:",rf_f1)
			print("accuracy:", metrics.accuracy_score(self.testing.target, rf_predicted))
			print("precision:",rf_precision)
			print("recall:",rf_recall)
		return

if __name__ == '__main__':
	if len(sys.argv) != 4:
		# print("MBC")
		# MBCtest = Best_Configuration("Selected 20NewsGroup/Training","Selected 20NewsGroup/Evaluation","temp.txt")
		sys.exit("python MBC_exploration.py <trainset> <evalset> <output>")
	else:
		MBC = Best_Configuration(sys.argv[1],sys.argv[2],sys.argv[3])
