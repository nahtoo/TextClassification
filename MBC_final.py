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

class Best_Configuration_Final:

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
			output_file.write(f"NB,MBC,{nb_precision},{nb_recall},{nb_f1}")
			print("Naive bayes")
			print("F1:",nb_f1)
			print("accuracy:", metrics.accuracy_score(self.testing.target, nb_predicted))
			print("precision:",nb_precision)
			print("recall:",nb_recall)

if __name__ == '__main__':
	if len(sys.argv) != 4:
		# print("MBC_Final")
		# MBCtest = Best_Configuration_Final("Selected 20NewsGroup/Training","Selected 20NewsGroup/Evaluation","temp.txt")
		sys.exit("python MBC_final.py <trainset> <evalset> <output>")
	else:
		MBC = Best_Configuration_Final(sys.argv[1],sys.argv[2],sys.argv[3])