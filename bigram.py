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

class Bigram_Baseline:

	def __init__(self,training,training_classes,testing,testing_classes,output):
		self.training = training
		self.training_classes = training_classes
		self.testing = testing
		self.testing_classes = testing_classes
		with open(output,"a") as output_file:
			naive_bayes_pipeline = Pipeline([('vect', CountVectorizer(ngram_range=(2,2))),('clf', MultinomialNB())])
			naive_bayes_pipeline.fit(training, training_classes)
			nb_predicted = naive_bayes_pipeline.predict(testing)
			nb_f1 = metrics.f1_score(testing_classes, nb_predicted, average='macro')
			nb_precision = metrics.precision_score(testing_classes, nb_predicted, average='macro')
			nb_recall = metrics.recall_score(testing_classes, nb_predicted, average='macro')
			output_file.write(f"NB,BB,{nb_precision},{nb_recall},{nb_f1}\n")
			print("Naive bayes")
			print("F1:",nb_f1)
			print("accuracy:", metrics.accuracy_score(testing_classes, nb_predicted))
			print("precision:",nb_precision)
			print("recall:",nb_recall)

			svm_pipeline = Pipeline([('vect', CountVectorizer(ngram_range=(2,2))),('clf', SGDClassifier())])
			svm_pipeline.fit(training, training_classes)
			svm_predicted = svm_pipeline.predict(testing)
			svm_precision = metrics.precision_score(testing_classes, svm_predicted, average='macro')
			svm_recall = metrics.recall_score(testing_classes, svm_predicted, average='macro')
			svm_f1 = metrics.f1_score(testing_classes, svm_predicted, average='macro')
			output_file.write(f"SVM,BB,{svm_precision},{svm_recall},{svm_f1}\n")
			print("SVM")
			print("F1:",svm_f1)
			print("accuracy:", metrics.accuracy_score(testing_classes, svm_predicted))
			print("precision:",svm_precision)
			print("recall:",svm_recall)

			logistic_pipeline = Pipeline([('vect', CountVectorizer(ngram_range=(2,2))),('logistic', LogisticRegression())])
			logistic_pipeline.fit(training, training_classes)
			logistic_predicted = logistic_pipeline.predict(testing)
			logistic_precision = metrics.precision_score(testing_classes, logistic_predicted, average='macro')
			logistic_recall = metrics.recall_score(testing_classes, logistic_predicted, average='macro')
			logistic_f1 = metrics.f1_score(testing_classes, logistic_predicted, average='macro')
			output_file.write(f"LR,BB,{logistic_precision},{logistic_recall},{logistic_f1}\n")
			print("Logistic")
			print("F1:",logistic_f1)
			print("accuracy:", metrics.accuracy_score(testing_classes, logistic_predicted))
			print("precision:",logistic_precision)
			print("recall:",logistic_recall) 

			random_forest_pipeline = Pipeline([('vect', CountVectorizer(ngram_range=(2,2))),('Randomforest', RandomForestClassifier(max_depth=25))])
			random_forest_pipeline.fit(training, training_classes)
			rf_predicted = random_forest_pipeline.predict(testing)
			rf_precision = metrics.precision_score(testing_classes, rf_predicted, average='macro')
			rf_recall = metrics.recall_score(testing_classes, rf_predicted, average='macro')
			rf_f1 = metrics.f1_score(testing_classes, rf_predicted, average='macro')
			output_file.write(f"RF,BB,{rf_precision},{rf_recall},{rf_f1}")
			print("Random Forest")
			print("F1:",rf_f1)
			print("accuracy:", metrics.accuracy_score(testing_classes, rf_predicted))
			print("precision:",rf_precision)
			print("recall:",rf_recall)
		return

	def display_LC(self):
		_ , axes = plt.subplots(1, 1, figsize=(10, 5))
		axes.set_title("Learning Curve")
		axes.grid()

		training_vector = Pipeline([('count', CountVectorizer(ngram_range=(2,2)))]).fit_transform(self.training+self.testing)
		combined_classes = self.training_classes + self.testing_classes

		selected_training_sizes = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
		
		NB_train_sizes, NB_train_scores, NB_test_scores = learning_curve(estimator=MultinomialNB(),X=training_vector,y=combined_classes,train_sizes=selected_training_sizes,shuffle=True)
		NB_train_scores_mean = np.mean(NB_train_scores, axis=1)
		NB_train_scores_std = np.std(NB_train_scores, axis=1)
		NB_test_scores_mean = np.mean(NB_test_scores, axis=1)
		NB_test_scores_std = np.std(NB_test_scores, axis=1)

		axes.fill_between(NB_train_sizes, NB_train_scores_mean - NB_train_scores_std,
							NB_train_scores_mean + NB_train_scores_std, alpha=0.1,
							color="g")
		axes.fill_between(NB_train_sizes, NB_test_scores_mean - NB_test_scores_std,
							NB_test_scores_mean + NB_test_scores_std, alpha=0.1,
							color="g")
		axes.plot(NB_train_sizes, NB_train_scores_mean, 'o-', color="g",
					label="NB Training score")
		axes.plot(NB_train_sizes, NB_test_scores_mean, 'v-', color="g",
					label="NB Cross-validation score")

		print(np.unique(self.training_classes))

		SVM_train_sizes, SVM_train_scores, SVM_test_scores = learning_curve(estimator=SGDClassifier(),X=training_vector,y=combined_classes,train_sizes=selected_training_sizes,shuffle=True)
		SVM_train_scores_mean = np.mean(SVM_train_scores, axis=1)
		SVM_train_scores_std = np.std(SVM_train_scores, axis=1)
		SVM_test_scores_mean = np.mean(SVM_test_scores, axis=1)
		SVM_test_scores_std = np.std(SVM_test_scores, axis=1)

		axes.fill_between(SVM_train_sizes, SVM_train_scores_mean - SVM_train_scores_std,
							SVM_train_scores_mean + SVM_train_scores_std, alpha=0.1,
							color="r")
		axes.fill_between(SVM_train_sizes, SVM_test_scores_mean - SVM_test_scores_std,
							SVM_test_scores_mean + SVM_test_scores_std, alpha=0.1,
							color="r")
		axes.plot(SVM_train_sizes, SVM_train_scores_mean, 'o-', color="r",
					label="SVM Training score")
		axes.plot(SVM_train_sizes, SVM_test_scores_mean, 'v-', color="r",
					label="SVM Cross-validation score")

		LG_train_sizes, LG_train_scores, LG_test_scores = learning_curve(estimator=LogisticRegression(),X=training_vector,y=self.training_classes,train_sizes=selected_training_sizes,shuffle=True)
		LG_train_scores_mean = np.mean(LG_train_scores, axis=1)
		LG_train_scores_std = np.std(LG_train_scores, axis=1)
		LG_test_scores_mean = np.mean(LG_test_scores, axis=1)
		LG_test_scores_std = np.std(LG_test_scores, axis=1)

		axes.fill_between(LG_train_sizes, LG_train_scores_mean - LG_train_scores_std,
							LG_train_scores_mean + LG_train_scores_std, alpha=0.1,
							color="r")
		axes.fill_between(LG_train_sizes, LG_test_scores_mean - LG_test_scores_std,
							LG_test_scores_mean + LG_test_scores_std, alpha=0.1,
							color="r")
		axes.plot(LG_train_sizes, LG_train_scores_mean, 'o-', color="b",
					label="LG Training score")
		axes.plot(LG_train_sizes, LG_test_scores_mean, 'v-', color="b",
					label="LG Cross-validation score")

		RF_train_sizes, RF_train_scores, RF_test_scores = learning_curve(estimator=RandomForestClassifier(),X=training_vector,y=self.training_classes,train_sizes=selected_training_sizes,shuffle=True)
		RF_train_scores_mean = np.mean(RF_train_scores, axis=1)
		RF_train_scores_std = np.std(RF_train_scores, axis=1)
		RF_test_scores_mean = np.mean(RF_test_scores, axis=1)
		RF_test_scores_std = np.std(RF_test_scores, axis=1)

		axes.fill_between(RF_train_sizes, RF_train_scores_mean - RF_train_scores_std,
							RF_train_scores_mean + RF_train_scores_std, alpha=0.1,
							color="r")
		axes.fill_between(RF_train_sizes, RF_test_scores_mean - RF_test_scores_std,
							RF_test_scores_mean + RF_test_scores_std, alpha=0.1,
							color="r")
		axes.plot(RF_train_sizes, RF_train_scores_mean, 'o-', color="m",
					label="RF Training score")
		axes.plot(RF_train_sizes, RF_test_scores_mean, 'v-', color="m",
					label="RF Cross-validation score")

		axes.legend(loc="best")
		plt.show()
		return 