Analysis project to investigate the utility of different learning algorithms for a text classification task, all implemented using the scikit library.

## <span>UB_BB.py</span>

`python UB_BB.py <trainset> <evalset> <output> <displayLC>`

where `<trainset>` is  the  parent  folder to training  data  and `<evalset>` is the parent folder to evaluation data. `<displayLC>` is  an  option  to  display  the  learning  curve . ‘1’  to  show  the  plot,  ‘0’  to  NOT  show  the  plot. `<output>` is  the path  to  a comma-separated  values  file,  which  contains  8  lines corresponding to 8 runs. 8 runs includes 4 learning algorithims (Naive Bayes, Logisitic Regression, SVM, Random Forest) for 2 configurations (Unigram tokenizer and Bigram tokenizer).

## <span>MBC_exploration.py</span>

Improved version of <span>UB_BB.py</span> with optimized hyper paremters for same 4 learning algorithims. 

`python MBCexploration.py <trainset> <evalset> <output>`





