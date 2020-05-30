# Classification of Seals in Images
This project contains 2 sub-programs- one which predicts 
whether a given image is of a seal or not 
(binary classifier) and the other which predicts 
the type of seal found in a given image 
(White-coat, Moulted-pup, Juvenile or Dead-pup).
The program executes a detailed analysis and cleaning of 
the given training data and experiments on it to decide on
the best classifier for the given images.

# Choosing a Model
The program experiments with various subsets of given 
features, and makes use of different visualizations 
to finally decide on 920-best features for binary model and 
800-best features for multi-class model.

The program also experiments with various classifiers- 
Logistic Regression, Support Vector Machines, Decision Trees,
 Random Forest as well 
as optimization algorithms such as 
Stochastic Gradient Descent, and compares each of their 
performances with one another before coming to a decision 
about the best classifier for the given data.

The various performance measures considered include- 
regular accuracy, balanced accuracy, F1 score, precision, 
recall and confusion matrices.

# Performance of the Final Model
Chosen binary classifier: 
Support Vector Machine with Linear Kernel (LinearSVC)
 with L2 regularization, 
giving a training accuracy of 97.4% and 
testing accuracy of 98.1%.

Chosen multi-class classifier:
Logistic Regression model with L2 regularization, giving
 a training accuracy of 94.78% and
 testing accuracy of 97.41%.
 
# Commented Code in the Program
A lot of the code has been commented out due to requirements from the coursework supervisor. 
They involve figures, different subsets of data and different training algorithms with respect to these subsets.
