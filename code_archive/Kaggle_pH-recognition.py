import numpy as np
import  matplotlib.pyplot as plt
# %matplotlib notebook
# %matplotlib inline
import pandas as pd
import seaborn as sb
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import r2_score, balanced_accuracy_score, roc_curve, roc_auc_score, confusion_matrix, classification_report


df = pd.read_csv('../input/ph-recognition/ph-data.csv') 
df.head()

colors = np.array([df.red, df.green, df.blue]).T

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
x = df.blue
y = df.green
z = df.red
ax.scatter(x, y, z, c=colors/255.0, s=30)
ax.set_title("Color distribution")
ax.set_xlabel("Blue")
ax.set_ylabel("Green")
ax.set_zlabel("Red")
plt.show()

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
labels = range(0, 15)
ph_colors = ['red','orangered', 'darkorange', 'orange', 'yellow', 'greenyellow', 'limegreen', 'green', 'aquamarine', 'aqua', 'blue', 'slateblue', 'darkslateblue', 'darkviolet', 'purple']
for l in labels:
    x = df[df['label'] == l].blue
    y = df[df['label'] == l].green
    z = df[df['label'] == l].red
    ax.scatter(x, y, z, label=l, c=ph_colors[l], s=30)
ax.set_title("Acidity distribution")
ax.set_xlabel("Blue")
ax.set_ylabel("Green")
ax.set_zlabel("Red")
ax.legend(loc='best')
plt.show()

df.isnull().sum()

def determine_acidity_3_group(df):
    if df['label'] == 7:
        val = 'neutral'
    elif df['label'] > 7:
        val = 'alkali'
    elif df['label'] < 7:
        val = 'acid'
    return val


def determine_acidity_5_group(df):
    if df['label'] == 7:
        val = 'neutral'
    elif df['label'] > 11:
        val = 'strong_alkali'
    elif df['label'] < 3:
        val = 'strong_acid'
    elif 11 >= df['label'] > 7:
        val = 'alkali'
    elif 3 <= df['label'] < 7:
        val = 'acid'    
    return val


df['acidity_3_group'] = df.apply(determine_acidity_3_group, axis=1)
df['acidity_5_group'] = df.apply(determine_acidity_5_group, axis=1)

df.head()

sb.countplot(x='label', data=df)

sb.countplot(x='acidity_3_group', data=df)

sb.countplot(x='acidity_5_group', data=df)



fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
labels = ['neutral', 'alkali', 'acid']
ph_colors = ['green','blue', 'red']
for l in range(0, 3):
    x = df[df['acidity_3_group'] == labels[l]].blue
    y = df[df['acidity_3_group'] == labels[l]].green
    z = df[df['acidity_3_group'] == labels[l]].red
    ax.scatter(x, y, z, label=labels[l], c=ph_colors[l], s=30)
ax.set_title("Acidity distribution")
ax.set_xlabel("Blue")
ax.set_ylabel("Green")
ax.set_zlabel("Red")
ax.legend(loc='best')
plt.show()

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
labels = ['neutral', 'strong_alkali', 'strong_acid', 'alkali', 'acid']
ph_colors = ['green','blue', 'red', 'aquamarine', 'yellow']
for l in range(0, 5):
    x = df[df['acidity_5_group'] == labels[l]].blue
    y = df[df['acidity_5_group'] == labels[l]].green
    z = df[df['acidity_5_group'] == labels[l]].red
    ax.scatter(x, y, z, label=labels[l], c=ph_colors[l], s=30)
ax.set_title("Acidity distribution")
ax.set_xlabel("Blue")
ax.set_ylabel("Green")
ax.set_zlabel("Red")
ax.legend(loc='best')
plt.show()



df_result_1 = df['label']
df_result_2 = df['acidity_3_group']
df_result_3 = df['acidity_5_group']
df_inputs = df.drop(['label', 'acidity_3_group', 'acidity_5_group'], axis=1)


X_train, X_test, y_train_1, y_test_1 = train_test_split(df_inputs, df_result_1, test_size=0.25, random_state=1)



y_train_2 = df['acidity_3_group'].iloc[y_train_1.index]
y_train_3 = df['acidity_5_group'].iloc[y_train_1.index]
y_test_2 = df['acidity_3_group'].iloc[y_test_1.index]
y_test_3 = df['acidity_5_group'].iloc[y_test_1.index]



def best_model(X_train, y_train):
    
    param_distribution_log_reg = {'penalty':['l2', 'none'], 'fit_intercept':[True,False], 'solver':['newton-cg', 'lbfgs', 'sag', 'saga']}
    param_distribution_KNN = {'algorithm':['auto', 'ball_tree', 'kd_tree', 'brute']}
    param_distribution_rand_forests = {'n_estimators': range(5, 20),  'max_depth': range(3, 30)}
    param_distribution_grad_boosting = {'n_estimators': range(5, 20),  'max_depth': range(3, 30)}
        
    if y_train.dtypes == 'int64':
        
        model_log_regression = GridSearchCV(LogisticRegression(), param_distribution_log_reg, scoring='balanced_accuracy', n_jobs=-1)
        model_KNN = GridSearchCV(KNeighborsClassifier(), param_distribution_KNN, scoring='balanced_accuracy', n_jobs=-1)
        model_rand_forests = RandomizedSearchCV(RandomForestClassifier(), param_distribution_rand_forests, n_iter=60, scoring='balanced_accuracy', n_jobs=-1, random_state=1)
        model_grad_boosting = RandomizedSearchCV(GradientBoostingClassifier(), param_distribution_grad_boosting, n_iter=60, scoring='balanced_accuracy', n_jobs=-1, random_state=1)
        
        model_log_regression.fit(X_train, y_train)
        model_KNN.fit(X_train, y_train)
        model_rand_forests.fit(X_train, y_train)
        model_grad_boosting.fit(X_train, y_train)
        
        result = pd.DataFrame({'Score': [model_log_regression.best_score_, model_KNN.best_score_, model_rand_forests.best_score_, model_grad_boosting.best_score_],
                               'Parameters': [model_log_regression.best_params_, model_KNN.best_params_, model_rand_forests.best_params_, model_grad_boosting.best_params_]}, 
                              index=['LogisticRegression', 'KNeighborsClassifier', 'RandomForestClassifier', 'GradientBoostingClassifier'])
        
    elif y_train.dtypes == 'object':
        
        model_log_regression = GridSearchCV(LogisticRegression(), param_distribution_log_reg, scoring='balanced_accuracy', n_jobs=-1)
        model_KNN = GridSearchCV(KNeighborsClassifier(), param_distribution_KNN, scoring='balanced_accuracy', n_jobs=-1)
        model_rand_forests = GridSearchCV(RandomForestClassifier(), param_distribution_rand_forests, scoring='balanced_accuracy', n_jobs=-1)
        model_grad_boosting = GridSearchCV(GradientBoostingClassifier(), param_distribution_grad_boosting, scoring='balanced_accuracy', n_jobs=-1)
        
        model_log_regression.fit(X_train, y_train)
        model_KNN.fit(X_train, y_train)
        model_rand_forests.fit(X_train, y_train)
        model_grad_boosting.fit(X_train, y_train)
        
        result = pd.DataFrame({'Score': [model_log_regression.best_score_, model_KNN.best_score_, model_rand_forests.best_score_, model_grad_boosting.best_score_],
                               'Parameters': [model_log_regression.best_params_, model_KNN.best_params_, model_rand_forests.best_params_, model_grad_boosting.best_params_]}, 
                              index=['LogisticRegression', 'KNeighborsClassifier', 'RandomForestClassifier', 'GradientBoostingClassifier'])
    
    return result

best_model(X_train, y_train_1)

best_model(X_train, y_train_2)

best_model(X_train, y_train_3)

best_model1 = RandomForestClassifier(n_estimators=16, max_depth=22)
best_model2 = RandomForestClassifier(n_estimators=9, max_depth=22)
best_model3 = RandomForestClassifier(n_estimators=13, max_depth=8)

best_model1.fit(X_train, y_train_1)
best_model2.fit(X_train, y_train_2)
best_model3.fit(X_train, y_train_3)

y_pred_1 = best_model1.predict(X_test)
y_pred_2 = best_model2.predict(X_test)
y_pred_3 = best_model3.predict(X_test)

def print_matrix(y_test, y_pred):
    cnf_matrix1 = confusion_matrix(y_test, y_pred)
    sb.heatmap(cnf_matrix1, annot=True, cmap='Blues', fmt='g')
    plt.ylabel('Real ph_value')
    plt.xlabel('Predicted ph_value')
    plt.show()
    return

print_matrix(y_test_1, y_pred_1)
print_matrix(y_test_2, y_pred_2)
print_matrix(y_test_3, y_pred_3)

print(classification_report(y_test_1, y_pred_1))
print(classification_report(y_test_2, y_pred_2, target_names=df.acidity_3_group.unique()))
print(classification_report(y_test_3, y_pred_3, target_names=df.acidity_5_group.unique()))

def roc_auc_score_summary(model, X_test, y_test):
    y_preb_probs = model.predict_proba(X_test)
    roc_auc_score_for_model = round(roc_auc_score(y_test, y_preb_probs, average="weighted", multi_class="ovr"), 3)
    count_of_classes = len(y_test.unique())
    print('ROC_AUC_score of model with', count_of_classes, 'classes =', roc_auc_score_for_model)
    return 

roc_auc_score_summary(best_model1, X_test, y_test_1)
roc_auc_score_summary(best_model2, X_test, y_test_2)
roc_auc_score_summary(best_model3, X_test, y_test_3)