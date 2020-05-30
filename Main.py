import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
from scipy import stats
from scipy.stats import chi2
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, \
    balanced_accuracy_score, classification_report, make_scorer
from sklearn.model_selection import cross_validate, cross_val_predict, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelBinarizer, OneHotEncoder, QuantileTransformer
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier



def print_heading(heading):
    print("\n---------------------------------------------------")
    print(heading)
    print("---------------------------------------------------")


print("Choose type of classifier:")
print("1. Binary\n2. Multiclass")
choice = input("\nEnter your choice (1 or 2): ")
while choice != '1' and choice != '2':
    print("Incorrect Choice. Please try again.")
    choice = input("\nEnter your choice (1 or 2): ")

# ----------------------------------------------------- Loading --------------------------------------------------------
print_heading("Loading")
directory = "binary/" if choice == '1' else "multi/"

# giving column names (feature numbers here) to train features (x)
feature_numbers = []
for count in range(0, 964):
    feature_numbers.append(str(count))

print("Loading X_train...")
train_x_org = pd.read_csv(directory + "X_train.csv", names=feature_numbers)
print("Loading Y_train...")
train_y_org = pd.read_csv(directory + "Y_train.csv", names=['class'])

classes = pd.unique(train_y_org['class'])
# print(classes)

# ---------------------------------------------------- Cleaning --------------------------------------------------------
print_heading("Cleaning")


def check_ordered(y):
    # separating indices of classes
    indices = {}
    for c in classes:
        indices[c] = y[y['class'] == c].index.values

    print("\nChecking if images of same class are grouped and ordered: ")
    # checking if data is ordered
    for c in classes:
        arr = indices.get(c)
        print(np.array_equal(y[arr[0]:arr[len(arr) - 1] + 1].index.values, arr))


check_ordered(train_y_org)

# merging and shuffling data since they are ordered
print("\nShuffling...")
merged_xy = train_x_org.copy()
merged_xy['class'] = train_y_org.copy()
merged_xy = merged_xy.sample(frac=1, random_state=1).reset_index(drop=True)

check_ordered(merged_xy['class'].to_frame())

# Looking for null data
print("\nChecking for null data...")
print(merged_xy.isnull().any().any())

# Looking for empty cells
print("\nChecking for empty cells...")
print(merged_xy.isna().any().any())

# separating back into x and y
train_y = merged_xy['class'].copy().to_frame()
train_x = merged_xy.drop('class', axis=1)

# checking data types
print("Getting an idea of entire data (data types)...\n")
'''
train_x.info()  # dtypes: float64(964)
train_y.info()  # dtypes: object(1)
'''
# train_y.astype(int).info()  # raises an error (invalid literal for int() with base 10: 'background');
# above error means that is is a string-- that's why it could not be changed to int

print("Count in each class: ")
if choice == '1':
    count = {'background': 0, 'seal': 0}
else:
    count = {'background': 0, 'whitecoat': 0, 'juvenile': 0, 'dead pup': 0, 'moulted pup': 0}
for row in range(0, len(train_y)):
    c = train_y.loc[row, 'class']
    val = count.get(c) + 1
    count.update({c: val})
print(count)

# Enumerating -- converting y values into class representations
if choice == '1':
    train_y.loc[train_y['class'] == 'background'] = 0
    train_y.loc[train_y['class'] == 'seal'] = 1
    train_y = train_y.astype(int)

# checking data types now
# train_x.info()  # dtypes: float64(964)
# train_y.info()  # dtypes: int32(1)

# -------------------------------------------- Analyzing & Visualizing--------------------------------------------------

'''fontsize = "30"
params = {'figure.autolayout': True,
          'legend.fontsize': fontsize,
          'figure.figsize': (12, 8),
          'axes.labelsize': fontsize,
          'axes.titlesize': fontsize,
          'xtick.labelsize': fontsize,
          'ytick.labelsize': fontsize}
plt.rcParams.update(params)'''

# analysing distribution (all data)
'''sns.set(palette='vlag', color_codes=True)
sns.distplot(train_y['class'])  # ensuring correct values

# getting an idea of each attribute
vis = train_x['410']  # checked for 0-899
vis.hist(bins=50, figsize=(20, 15))
plt.show()

vis = train_x['410']  # checked for 900-915
vis.hist(bins=50, figsize=(20, 15))
plt.show()

vis = train_x['410']  # checked for 920-963
vis.hist(bins=50, figsize=(20, 15))
plt.show()

train_x.loc[:, :'899'].describe()  # checking range

# normal distribution of 900-915
vis = train_x.loc[:, '900':'915']
vis.hist(bins=50, figsize=(20, 15))
plt.show()

train_x.loc[:, '900:'915'].describe()  # checking range

# distribution of last red component pixels
sns.set(palette='prism_r', color_codes=True)
vis = train_x.loc[:, '916':'931']
vis.hist(bins=50, figsize=(20, 15))
plt.show()

train_x.loc[:, '916':'931'].describe()  # checking range

# distribution of last green component pixels
sns.set(palette='Greens_r', color_codes=True)
vis = train_x.loc[:, '932':'947']
vis.hist(bins=50, figsize=(20, 15))
plt.show()

train_x.loc[:, '932':'947'].describe()  # checking range

# distribution of last blue component pixels
sns.set(palette='winter', color_codes=True)
vis = train_x.loc[:, '948':'963']
vis.hist(bins=50, figsize=(20, 15))
plt.show()

train_x.loc[:, '948':'963'].describe()  # checking range'''

# Correlations

# checking feature-feature correlations
'''correlations = train_x.corr()
corr_matrix = {}
for x in range(0, 964):
    feature_name = str(x)
    corr = correlations[feature_name].sort_values(ascending=False)
    most_correlated = corr[1:2]
    most_correlated = most_correlated.append(corr[963:964])
    corr_matrix[feature_name] = most_correlated.to_frame()

# getting feature pairs with correlation > 0.9
high_corr = {}
for x in range(0, 964):
    feature_name = str(x)
    corr = corr_matrix.get(feature_name)
    for m in corr.loc[:, feature_name].index:
        if corr.loc[m, feature_name] > 0.98:
            # if corr.loc[m, feature_name] > 0.95:  # many
            high_corr[feature_name] = corr.loc[m].to_frame()
            print(high_corr[feature_name])
        if corr.loc[m, feature_name] < -0.98:
            # if corr.loc[m, feature_name] < -0.5:  # max is -0.5974
            high_corr[feature_name] = corr.loc[m].to_frame()
            print(high_corr[feature_name])

'''
# heat map for highest-correlated features
'''attribs = ['931', '947', '963']
heat_map = train_x[attribs].corr()
sns.heatmap(heat_map, cmap="Reds")'''

# checking feature-output correlations
'''merged_xy = train_x.copy()
merged_xy['class'] = train_y.copy()
target_corr = merged_xy.corr()'''
'''print("\ntarget_corr['class']:")
print(target_corr["class"].sort_values(ascending=False))'''
'''
Best 2 features:
950      0.234601
630     -0.243707
'''

# check correlation of ['931', '947', '963'] to target
# print(target_corr['class'][attribs])
'''print("\ntarget_corr:")
print(target_corr)
print("\nmerged_xy:")
print(merged_xy)'''

# -------------------------------------------------- Data Preparation --------------------------------------------------
print_heading("Data Preparation")

print("Removing highly correlated features...")

# Removing highly correlated from train_x
train_x = train_x.drop(['947', '931'], axis=1)

print("Selecting the best feature set...")

# K Best
feature_names = train_x.columns

'''kbest_selector = SelectKBest(f_classif, k=920)
kbest_x = kbest_selector.fit_transform(train_x, train_y.values.ravel())
print("New shape of train_x:", kbest_x.shape)'''

'''setAC = train_x.drop(train_x.loc[:, '900':'915'].columns, axis=1)
kbest_selector = SelectKBest(f_classif, k=920)
kbest_x = kbest_selector.fit_transform(setAC, train_y.values.ravel())
print("New shape of train_x:", kbest_x.shape)'''

'''kbest_msg = "920 Best features: "
# kbest_msg = "500 Best features: "
# kbest_msg = "900 Best features: "
# kbest_msg = "700 Best features: "
# kbest_msg = "800 Best features: "'''

# Scaling

print("Scaling...")

# All
'''scaler_all = StandardScaler().fit(train_x)
scaled_all_x = scaler_all.transform(train_x)
'''
# removing correlated
'''scaler_all = StandardScaler().fit(train_x_corremoved)
scaled_all_x = scaler_all.transform(train_x_corremoved)'''

# set A
'''
subset = train_x.loc[:, '0':'899']
scaler_A = StandardScaler().fit(subset)
scaled_A_x = scaler_A.transform(subset)'''

# set B
'''
subset = train_x.loc[:, '900':'915']
scaler_B = StandardScaler().fit(subset)
scaled_B_x = scaler_B.transform(subset)'''

# set C
'''
subset = train_x.loc[:, '916':]
scaler_C = StandardScaler().fit(subset)
scaled_C_x = scaler_C.transform(subset)'''

# set AB
'''
subset = train_x.loc[:, :'915']
scaler_AB = StandardScaler().fit(subset)
scaled_AB_x = scaler_AB.transform(subset)'''

# set AC
'''setB = train_x.loc[:, '900':'915']
subset = train_x.drop(setB.columns, axis=1)
scaler_AC = StandardScaler().fit(subset)
scaled_AC_x = scaler_AC.transform(subset)'''

# set BC
'''
subset = train_x.loc[:, '900':]
scaler_BC = StandardScaler().fit(subset)
scaled_BC_x = scaler_BC.transform(subset)'''

# K Best
'''subset = kbest_x
scaler_kbest = StandardScaler().fit(subset)
scaled_kbest_x = scaler_kbest.transform(subset)'''

# ------------------------------------------------ Choosing a Subset ---------------------------------------------------

train_y = train_y.values.ravel()


# clf = LogisticRegression(solver="sag", max_iter=1000)
# clf = LogisticRegression(max_iter=2000)  # lbfgs

# Cross Validation
def validate(m, x, y, k):
    print("\n")
    '''scores = cross_validate(model, x, y, cv=k, scoring=("accuracy", "balanced_accuracy", "precision", "recall", "f1")
                            return_train_score=True)'''
    scores = cross_validate(m, x, y, cv=k, scoring="balanced_accuracy", return_train_score=True)
    print("Training balanced accuracy: ", scores["train_score"].mean())
    '''print("Training F1 score: ", scores["train_f1"].mean())
    print("Training precision: ", scores["train_precision"].mean())
    print("Training recall: ", scores["train_recall"].mean())'''
    print("---")
    print("Validation balanced accuracy: ", scores["test_score"].mean())
    '''print("Validation F1 score: ", scores["test_f1"].mean())
    print("Validation precision: ", scores["test_precision"].mean())
    print("Validation recall: ", scores["test_recall"].mean())'''


# validate("All:", clf, scaled_all_x, train_y, 10)
# validate("All with Correlated Removed:", clf, scaled_all_x, train_y, 10)
# validate("Set A only:", clf, scaled_A_x, train_y, 10)
# validate("Set B only:", clf, scaled_B_x, train_y, 10)
# validate("Set C only:", clf, scaled_C_x, train_y, 10)
# validate("Set AB:", clf, scaled_AB_x, train_y, 10)
# validate("Set AC only:", clf, scaled_AC_x, train_y, 10)
# validate("Set BC only:", clf, scaled_BC_x, train_y, 10)
# validate(kbest_msg, clf, scaled_kbest_x, train_y, 10)

# Cross validated Predictions
# cv_predictions = cross_val_predict(clf, scaled_kbest_x, train_y, cv=10)
'''cv_predictions = cross_val_predict(clf, scaled_all_x, train_y, cv=10)

# confusion_matrix()
print("Confusion Matrix:\n", confusion_matrix(train_y, cv_predictions))
# print(set(train_y) - set(cv_predictions))

# precision_score()
print("Precision: ", precision_score(train_y, cv_predictions, average=None))

# recall_score()
print("Recall: ", recall_score(train_y, cv_predictions, average=None))'''

# ------------------------------------------------- Choosing a Model ---------------------------------------------------

print_heading("Choosing a Model")

if choice == '1':
    k_val = 920
    print(k_val, "features")
    print("Chosen model: Linear SVC")
    # model = LogisticRegression(max_iter=2000, verbose=1, n_jobs=-1, tol=1)
    # to track progress, please uncomment the following (verbose)
    # model = LinearSVC(dual=False, verbose=1)
    model = LinearSVC(dual=False)
    # model = RandomForestClassifier(verbose=1, n_jobs=-1)
else:
    k_val = 800
    print(k_val, "features")
    print("Chosen model: Logistic Regression")
    # for faster performance, please uncomment the following line (n_jobs for speed and verbose for tracking)
    # model = LogisticRegression(max_iter=2000, multi_class='ovr', verbose=1, n_jobs=-1, C=0.1, tol=0.05, solver='sag')
    model = LogisticRegression(max_iter=2000, multi_class='ovr', C=0.1, tol=0.05, solver='sag')
    # model = SGDClassifier(learning_rate='optimal', alpha=0.1, verbose=1)
    # model = SGDClassifier(learning_rate='optimal', alpha=0.1, loss='log')
    # model = RandomForestClassifier(verbose=1, n_jobs=-1)

# Getting feature names of best
kbest_selector = SelectKBest(f_classif, k=k_val)
kbest_x = kbest_selector.fit_transform(train_x, train_y)
mask = kbest_selector.get_support()  # list of booleans
new_features = []  # The list of k best features

for bool, feature in zip(mask, feature_names):
    if bool:
        new_features.append(feature)
'''print("Best features: ", new_features)

useless = set(feature_names) - set(new_features)
a = []
b = []
c = []
for i in useless:
    if '0' <= i < '900':
        a.append(i)
    elif '900' <= i < '916':
        b.append(i)
    else:
        c.append(i)

print("Count from set A", len(a))  # 33 features selected
print("Count from set B", len(b))  # 7 features selected
print("Count from set C", len(c))  # 2 features selected'''
# exit()

scaler = StandardScaler().fit(kbest_x)
scaled = scaler.transform(kbest_x)
print("Validating...")

'''if choice == '1':
    validate(model, scaled, train_y, 5)

else:
    validate(model, scaled, train_y, 5)
    # predictions = cross_val_predict(model, scaled, train_y, cv=5)
    # print("Confusion Matrix:\n", confusion_matrix(train_y, predictions))
    # print("Balanced Accuracy: ", balanced_accuracy_score(train_y, predictions))
    # print("F1 score: ", f1_score(train_y, predictions, average=None))
    # print("Precision: ", precision_score(train_y, predictions, average=None))
    # print("Recall: ", recall_score(train_y, predictions, average=None))
    # print(classification_report(train_y, predictions))

exit()'''

# -------------------------------------------- Hyperparameter Tuning ---------------------------------------------------

print_heading("Hyperparameter Tuning")

'''classifiers = [LogisticRegression(max_iter=2000, verbose=1, n_jobs=-1, tol=1), SGDClassifier(), SVC(kernel="linear"), 
DecisionTreeClassifier(), RandomForestClassifier()]'''
'''classifiers = [("Decision Tree", DecisionTreeClassifier()), ("Random Forest", RandomForestClassifier()),
("Linear SVC", LinearSVC(dual=False))]'''

'''
classifiers = [("Logistic Regression", LogisticRegression(max_iter=2000, verbose=1))]
# classifiers = [("Stochastic Gradient Descent", SGDClassifier(n_jobs=-1, learning_rate='optimal', alpha=0.1))]
# classifiers = [("Random Forest", RandomForestClassifier(verbose=1, n_jobs=-1))]

for (name, clf) in classifiers:
    param_grid = [
        # {'kbest__k': [800, 900, 920]}
        # {'model__loss': ['log', 'hinge']}
        # {'model__C':[1, 0.5, 0.1]}
        # {'model__n_estimators': [100, 500], 'model__max_leaf_nodes':[None, 16]}
    ]
    pipe_steps = Pipeline([('scaler', StandardScaler()),
                           ('kbest', SelectKBest()),
                           ('model', clf)])
    msg = "\n----- Classfier: " + name + "-----"
    print(msg)

    pipe_steps.fit(train_x, train_y)

    scores = cross_validate(pipe_steps, train_x, train_y, cv=5, scoring="balanced_accuracy",
                            return_train_score=True)
    print("Training balanced_accuracy: ", scores["train_score"].mean())
    print("Validation balanced_accuracy: ", scores["test_score"].mean())

    grid_search = GridSearchCV(pipe_steps, param_grid, verbose=1, cv=5, n_jobs=-1,
                               scoring="balanced_accuracy",
                               return_train_score=True)
    grid_search.fit(train_x, train_y)
    print("Training balanced_accuracy", grid_search.cv_results_['mean_train_score'])
    print("Validation balanced_accuracy", grid_search.cv_results_['mean_test_score'])
    print(grid_search.classes_)
    print(grid_search.best_params_)
'''
# --------------------------------------------- Training the best model ------------------------------------------------
print_heading("Training")
print("Fitting chosen model...")
model.fit(scaled, train_y)
predictions = model.predict(scaled)
accuracy = accuracy_score(train_y, predictions)
bal_acc = balanced_accuracy_score(train_y, predictions)
print("Training Accuracy: ", accuracy)
print("Training Balanced Accuracy: ", bal_acc)

# ------------------------------------------------ Final Predictions ---------------------------------------------------
print_heading("Predicting on test set")
print("Loading X_test...")
test_x = pd.read_csv(directory + "X_test.csv", names=feature_numbers)

print("Removing highly correlated features...")
test_x = test_x.drop(['947', '931'], axis=1)

print("Selecting the best feature set...")
test_x = test_x[new_features]
# print(test_x.columns)

print("Scaling...")
scaled_test_x = scaler.transform(test_x)

print("Predicting...")
test_y = model.predict(scaled_test_x)

# Changing enumerations back to class names
result_pd = pd.DataFrame(data=test_y[:], columns=["class"])
if choice == '1':
    print("Decoding...")
    result_pd.loc[result_pd['class'] == 0] = 'background'
    result_pd.loc[result_pd['class'] == 1] = 'seal'

# result_pd.to_csv('binary/Y_test.csv', index=False, header=False)
# result_pd.to_csv('multi/Y_test.csv', index=False, header=False)
print(result_pd)
