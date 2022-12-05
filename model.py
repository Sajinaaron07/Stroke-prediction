import pandas as pd
import numpy as np

from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import f1_score

from sklearn.linear_model import LogisticRegression


import pickle
import warnings
warnings.filterwarnings('ignore')

# settings
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.max_columns', None)


#cross-validation

# Below function implements above steps.
def run_kfold(model, X_train, y_train, N_SPLITS = 10):
    f1_list = []
    oofs = np.zeros(len(X_train))
    folds = StratifiedKFold(n_splits=N_SPLITS)
    for i, (trn_idx, val_idx) in enumerate(folds.split(X_train, y_train)):
        
        print(f'\n------------- Fold {i + 1} -------------')
        X_trn, y_trn = X_train.iloc[trn_idx], y_train.iloc[trn_idx]
        X_val, y_val = X_train.iloc[val_idx], y_train.iloc[val_idx]
        
        model.fit(X_trn, y_trn)
        # Instead of directly predicting the classes we will obtain the probability of positive class.
        preds_val = model.predict_proba(X_val)[:,1]
        
        fold_f1 = f1_score(y_val, preds_val.round())
        f1_list.append(fold_f1)
        
        print(f'\nf1 score for validation set is {fold_f1}') 
        
        oofs[val_idx] = preds_val
        
    mean_f1 = sum(f1_list)/N_SPLITS
    print("\nMean validation f1 score :", mean_f1)
    
    oofs_score = f1_score(y_train, oofs.round())
    print(f'\nF1 score for oofs is {oofs_score}')
    return oofs


# Load data into memory
data = pd.read_csv('data/healthcare-dataset-stroke-data.csv')

print("No of columns in the data : ", len(data.columns))
print("No of rows in the data : ", len(data))

# features
features = ['gender', 'age', 'hypertension', 'heart_disease', 'ever_married','work_type','Residence_type',
            'avg_glucose_level', 'bmi','smoking_status']

#target
target = 'stroke'

numerical_features = ['age', 'avg_glucose_level', 'bmi']

categorical_features = ['gender', 'hypertension', 'heart_disease', 'ever_married', 'work_type', 
                        'Residence_type', 'smoking_status']


# Converting features into required datatypes
data[numerical_features] = data[numerical_features].astype(np.float64)

data[categorical_features] = data[categorical_features].astype('category')

# Replace Other label in gender with Female
data.gender.replace({'Other':"Female"}, inplace=True)

# Remove id column
data.drop('id', axis=1, inplace=True)


train, test = train_test_split(data, random_state=1, test_size=0.25, stratify=data.stroke)

print("No. of data points in training set : ", len(train))
print("No. of data points in testing set : ", len(test))


imputer = KNNImputer(n_neighbors = 5)

train[numerical_features] = imputer.fit_transform(train[numerical_features])
test[numerical_features] = imputer.transform(test[numerical_features])


def age_group(x):
    if x<13: return "Child"
    elif 13<x<20: return "Teenager"
    elif 20<x<=60: return "Adult"
    else: return "Elder"
    
train["age_group"] = train.age.apply(age_group)
test['age_group'] = test.age.apply(age_group)

def bmi_group(x):
    if x<18.5 : return "UnderWeight"
    elif 18.5<x<25: return "Healthy"
    elif 25<x<30: return "OverWeight"
    else: return "Obese"

train["bmi_group"] = train.bmi.apply(age_group)
test['bmi_group'] = test.bmi.apply(age_group)


# add new features
categorical_features.extend(["bmi_group", "age_group"])

encoder = OneHotEncoder(drop='first', sparse=False)
encoder.fit(train[categorical_features])

cols = encoder.get_feature_names(categorical_features)

train.loc[:, cols] = encoder.transform(train[categorical_features])
test.loc[:, cols] = encoder.transform(test[categorical_features])

# Drop categorical features
train.drop(categorical_features, axis=1, inplace=True)
test.drop(categorical_features, axis=1, inplace=True)


scaler = StandardScaler()
scaler.fit(train[numerical_features])

train.loc[:, numerical_features] = scaler.transform(train[numerical_features])
test.loc[:, numerical_features] = scaler.transform(test[numerical_features])


features = ['age', 'avg_glucose_level', 'bmi', 'gender_Male','hypertension_1', 'heart_disease_1', 'ever_married_Yes',
            'work_type_Never_worked', 'work_type_Private','work_type_Self-employed', 'work_type_children', 
            'Residence_type_Urban','smoking_status_formerly smoked', 'smoking_status_never smoked','smoking_status_smokes',
            'bmi_group_Child', 'bmi_group_Elder', 'bmi_group_Teenager', 'age_group_Child', 'age_group_Elder', 'age_group_Teenager']

X_train = train[features]
y_train = train[target]

X_test = test[features]
y_test = test[target]



# Base model
clf = LogisticRegression(random_state=1, class_weight='balanced')
clf.fit(X_train, y_train)
train_preds = clf.predict(X_train)
test_preds = clf.predict(X_test)
print("Train f1 Score :", f1_score(y_train, train_preds))
print("Test f1 Score :", f1_score(y_test, test_preds))



# Hyperparameter tuning
params = {
    'penalty': ['l1', 'l2','elasticnet'],
    'C':[0.0001, 0.001, 0.1, 1, 10, 100,1000],
    'fit_intercept':[True, False],
    'solver' : ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
}

clf = RandomizedSearchCV(LogisticRegression(random_state=1, class_weight='balanced'),
                         params,
                         scoring='f1',
                         verbose=1,
                         random_state=1,
                         cv=5,
                         n_iter=20)

search = clf.fit(X_train, y_train)



# Cross validation
clf = LogisticRegression(random_state = 1, class_weight='balanced', **search.best_params_)
oofs = run_kfold(clf, X_train, y_train, N_SPLITS=5)



clf = LogisticRegression(random_state = 1,class_weight='balanced', **search.best_params_)
clf.fit(X_train, y_train)


imp = pd.DataFrame([features, clf.coef_[0]]).T.sort_values(1, ascending=False).reset_index(drop=True)
imp.columns=['feature', 'coeff']


with open("onehotencoder.pkl", 'wb') as f:
    pickle.dump(encoder, f)

with open("scaler.pkl", 'wb') as f:
    pickle.dump(scaler, f)

with open("model.pkl", 'wb') as f:
    pickle.dump(clf, f)