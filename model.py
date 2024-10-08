import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.tree import DecisionTreeClassifier

# Data path
TRAIN_DATA_PATH = 'data/initial/census-income.csv'
# TEST_DATA_PATH = 'data/initial/census-income.test.csv'
CAT_FEATURES = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
              'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
              'class']
DISCRETE_VAR = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
CONTINUOUS_VAR = ['fnlwgt']
NOMINAL_VAR = ['race', 'sex', 'workclass', 'marital-status', 'occupation', 'relationship',
             'native-country']
ORDINAL_VAR = ['education', 'education-num', 'age', 'fnlwgt', 'hours-per-week', 'capital-loss', 'capital-gain']
TARGET_VAR = ['class']

def get_data(datapath):
  """Function to get data"""
  data = pd.read_csv(datapath, header=None)
  data = data.replace(' ?', np.nan)
  data = data.set_axis(CAT_FEATURES, axis=1, copy=False)
  return data

def get_data_noremove(datapath):
  """Function to get data"""
  data = pd.read_csv(datapath)
  return data

def fill_missing_data():
  """Function to fill missing data"""
  df = pd.read_csv('data/encoded/nominal.csv')
  data_copy = df.copy(deep=True)

  missing_cols = [col for col in df.columns if '_nan' in col]
  workclass_cols = [col for col in df.columns if 'workclass' in col]
  occupation_cols = [col for col in df.columns if 'occupation' in col]
  native_country_cols = [col for col in df.columns if 'native-country' in col]

  for col in missing_cols:
      data_copy.loc[data_copy[col] == 1, workclass_cols] = np.nan
      data_copy.loc[data_copy[col] == 1, occupation_cols] = np.nan
      data_copy.loc[data_copy[col] == 1, native_country_cols] = np.nan

  data_copy = data_copy.drop(missing_cols, axis=1)
  workclass_cols.remove('workclass_nan')
  occupation_cols.remove('occupation_nan')
  native_country_cols.remove('native-country_nan')

  data_copy.to_csv('data/encoded/nominal_missing.csv', index=False)
  imp = KNNImputer(n_neighbors=201, weights="distance", metric="nan_euclidean")
  filled_data = imp.fit_transform(data_copy)

  filled_df = pd.DataFrame(filled_data, columns=data_copy.columns)
  filled_df.to_csv('data/encoded/nominal_filled_raw.csv', index=False)

  columnames = [workclass_cols, occupation_cols, native_country_cols]

  for index, row in filled_df.iterrows():
      for col_group in columnames:
          if any(row[col] != 0 and row[col] != 1 for col in col_group):
              max_index = row[col_group].idxmax()
              filled_df.loc[index, max_index] = 1
              cols_set_to_zero = [x for x in col_group if x != max_index]
              filled_df.loc[index, cols_set_to_zero] = 0

  filled_df.to_csv('data/encoded/nominal_filled.csv', index=False)
  return filled_df

def encode_data(dataframe, nominal_vars, ordinal_vars):
  """Function to handle categorical data"""
  onehotencoder = OneHotEncoder()
  target = dataframe.iloc[:, -1]
  df_nominal = dataframe[nominal_vars]
  df_nominal = onehotencoder.fit_transform(df_nominal).toarray()
  df_nominal = pd.DataFrame(df_nominal)
  column_names = []
  for i, category in enumerate(onehotencoder.categories_):
      column_names.extend([f"{nominal_vars[i]}_{value}" for value in category])
  df_nominal.columns = column_names

  labelencoder = LabelEncoder()
  df_ordinal = dataframe[ordinal_vars]
  df_ordinal = df_ordinal.apply(labelencoder.fit_transform)

  df_ordinal = pd.DataFrame(df_ordinal)

  df_nominal.to_csv('data/encoded/nominal.csv', index=False)
  df_ordinal.to_csv('data/encoded/ordinal.csv', index=False)

  dataframe = pd.concat([df_nominal, df_ordinal], axis=1)
  dataframe = pd.concat([dataframe, target], axis=1)
  dataframe.to_csv('data/encoded/census-income-encoded.csv', index=False)

  return dataframe, target

def calculate_metrics(y_test, y_pred):
  acc = accuracy_score(y_test, y_pred)
  prec = precision_score(y_test, y_pred, labels=[' <=50K', ' >50K'], pos_label=' >50K')
  rec = recall_score(y_test, y_pred, labels=[' <=50K', ' >50K'], pos_label=' >50K')
  f1 = f1_score(y_test, y_pred, labels=[' <=50K', ' >50K'], pos_label=' >50K')
  return acc, prec, rec, f1

def naive_bayes_classifier(dataframe):
  X = dataframe.iloc[1:, :-1]
  y = dataframe.iloc[1:, -1]
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.67, random_state=42)
  clf = GaussianNB()
  clf.fit(X_train, y_train)
  y_pred = clf.predict(X_test)
  return calculate_metrics(y_test, y_pred)

def ann_classifier(dataframe):
  X = dataframe.iloc[1:, :-1]
  y = dataframe.iloc[1:, -1]
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.67, random_state=42)
  
  results = []
  for hidden_layers in [(100,), (100, 100), (100, 100, 100)]:
      clf = MLPClassifier(hidden_layer_sizes=hidden_layers, max_iter=1000, activation='logistic')
      clf.fit(X_train, y_train)
      y_pred = clf.predict(X_test)
      results.append(calculate_metrics(y_test, y_pred))
  
  return results

def logistic_regression_classifier(dataframe):
  X = dataframe.iloc[1:, :-1]
  y = dataframe.iloc[1:, -1]
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.67, random_state=42)
  clf = LogisticRegression(max_iter=1000)
  clf.fit(X_train, y_train)
  y_pred = clf.predict(X_test)
  return calculate_metrics(y_test, y_pred)

def random_forest_classifier(dataframe):
  X = dataframe.iloc[1:, :-1]
  y = dataframe.iloc[1:, -1]
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.67, random_state=42)
  clf = RandomForestClassifier()
  clf.fit(X_train, y_train)
  y_pred = clf.predict(X_test)
  return calculate_metrics(y_test, y_pred)

def decision_tree_classifier(dataframe):
  X = dataframe.iloc[1:, :-1]
  y = dataframe.iloc[1:, -1]
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.67, random_state=42)
  clf = DecisionTreeClassifier()
  clf.fit(X_train, y_train)
  y_pred = clf.predict(X_test)
  return calculate_metrics(y_test, y_pred)

def run_code():
  """Function to run the code"""
  dataframe = get_data(TRAIN_DATA_PATH)
  dataframe_encoded, target = encode_data(dataframe, NOMINAL_VAR, ORDINAL_VAR)
  nominal_filled = fill_missing_data()
  ordinal_data = get_data_noremove('data/encoded/ordinal.csv')
  combined_data = pd.concat([nominal_filled, ordinal_data], axis=1)
  combined_data = pd.concat([combined_data, target], axis=1)
  combined_data.to_csv('data/processed/combined.csv', index=False)

  results = {}
  results['Naive Bayes'] = naive_bayes_classifier(combined_data)
  ann_results = ann_classifier(combined_data)
  results['ANN (1 hidden layer)'] = ann_results[0]
  results['ANN (2 hidden layers)'] = ann_results[1]
  results['ANN (3 hidden layers)'] = ann_results[2]
  results['Logistic Regression'] = logistic_regression_classifier(combined_data)
  results['Random Forest'] = random_forest_classifier(combined_data)
  results['Decision Tree'] = decision_tree_classifier(combined_data)

  # Train and save the best model (Random Forest)
  X = combined_data.iloc[1:, :-1]  # All rows except the first one, all columns except the last one
  y = combined_data.iloc[1:, -1]  # All rows except the first one, the last column
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
  
  best_model = RandomForestClassifier()
  best_model.fit(X_train, y_train)
  
  # Save the model
  joblib.dump(best_model, 'random_forest_model.joblib')

  return results

if __name__ == "__main__":
  metrics = run_code()
  for model, (acc, prec, rec, f1) in metrics.items():
      print(f"{model}:")
      print(f"  Accuracy: {acc:.4f}")
      print(f"  Precision: {prec:.4f}")
      print(f"  Recall: {rec:.4f}")
      print(f"  F1 Score: {f1:.4f}")
      print()
  
  print("Best model (Random Forest) has been trained and saved as 'random_forest_model.joblib'")