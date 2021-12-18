import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, Normalizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
import prettytable
import warnings
warnings.filterwarnings("ignore")
import os

# Encode Features
class encoder():
  '''
  Encode the set of features given encoding type
  inp: encoder(FeatureType(str), features(list)
  FEatureType : 'LabelEncoder' or 'OneHotEncoder' or 'Normalizer'
  '''
  
  def __init__(self, FeatureType, features):
    self.type_ = FeatureType
    self.features = features
  
  def encoding(self, train_data, cv_data, test_data ,test=True):
    '''encode train_data, cv_data, test_data if test=True ,
        return encoded data given encoding type'''

    if self.type_=='LabelEncoder':
      enc = LabelEncoder()
      for i in self.features:
        enc.fit(train_data[i])
        train_data[i] = enc.transform(train_data[i])
        cv_data[i] = enc.transform(cv_data[i])
        if test:
          test_data[i] = enc.transform(test_data[i])
    
    elif self.type_== 'OneHotEncoder':
      enc = OneHotEncoder()
      for i in self.features:
        enc.fit(train_data[i])
        train_data[i] = enc.transform(train_data[i])
        cv_data[i] = enc.transform(cv_data[i])
        if test:
          test_data[i] = enc.transform(test_data[i])
    
    elif self.type_ =='Normalizer':
      enc = Normalizer()
      for i in self.features:
        enc.fit(train_data[i])
        train_data[i] = enc.transform(train_data[i])
        cv_data[i] = enc.transform(cv_data[i])
        if test:
          test_data[i] = enc.transform(test_data[i])
    if test:
      return enc, train_data, cv_data, test_data
    return enc, train_data, cv_data



class Model:
    def __init__(self, train_file, test_file, model):
       self.train = pd.read_csv(train_file)
       self.test = pd.read_csv(test_file)
       self.user_defined_model = model
    
    def preprocess(self):
        ## adding new features engineered
        #Ranking categories acc to color most chosen
        ranks_dict = [{i:n%10 + 1} for n,i in enumerate(self.train.groupby('category')['color'].value_counts().keys())]

        ranking_dict = {}
        for i in range(len(ranks_dict)):
            ranking_dict[['_'.join(i) for i in ranks_dict[i].keys()][0]] = list(ranks_dict[i].values())[0]
        
        # adding column with format'category_color' ex: 'Blouse_Blue' 
        self.train['color_category_rank'] = self.train['category']+ '_' + self.train['color']
        self.test['color_category_rank'] = self.test['category'] + '_' + self.test['color']
        #adding a rank column
        self.train['color_category_rank'] = self.train['color_category_rank'].apply(lambda x:ranking_dict.get(x))
        self.test['color_category_rank'] = self.test['color_category_rank'].apply(lambda x:ranking_dict.get(x))

        print("Feature engineering Done ....!\n")
        
    def split(self, test_size):
        self.x_train = self.train.drop(['item_no','success_indicator'], axis=1)
        self.y_train = self.train['success_indicator']
        self.x_test = self.test.drop(['item_no'], axis=1)
        
        #train_cv_split
        self.x_train, self.x_cv, self.y_train, self.y_cv = train_test_split(self.x_train, self.y_train, test_size= test_size, shuffle= self.y_train, random_state=42)
        print("train_cv_test split done.....!\n")
        
    def encode(self):
        #encoding all the data 
        cat_cols = self.x_train.columns[self.x_train.dtypes =='object']
        num_cols = self.x_train.columns[self.x_train.dtypes =='float']
        
        feature_enc, self.x_train, self.x_cv, self.x_test = encoder('LabelEncoder', cat_cols).encoding(self.x_train, self.x_cv, self.x_test, test=True)
        
        lbl_enc ,self.y_train, self.y_cv = encoder('LabelEncoder', ['success_indicator']).encoding(
                        pd.DataFrame(self.y_train, columns=['success_indicator']),
                        pd.DataFrame(self.y_cv, columns =['success_indicator']), None, test= False)
        print("Encoding done....!\n")
        return lbl_enc, self.x_train, self.x_cv, self.x_test, self.y_train, self.y_cv
    
    def fit(self):
        self.model = self.user_defined_model.fit(self.x_train, self.y_train)

    def predict(self, input_value):
        result = self.user_defined_model.predict(np.array([input_value]))
        
        return result

if __name__ == '__main__':
    table = prettytable.PrettyTable()
    table.field_names = ["Index","Model Name"]
    table.add_row([1,"Gaussian Naive Bayes"])
    table.add_row([2,"Logistic Regression"])
    table.add_row([3,"Decision Tree"])
    print(table)

    model_  = input("Enter the model index you want to train on :")
    model_dict = {1: GaussianNB(), 2: LogisticRegression(), 3: DecisionTreeClassifier(max_depth = 10, min_samples_split=3)}
    
    model_instance = Model(train_file = "historic.csv", test_file = "prediction_input.csv", model= model_dict.get(int(model_)))
    model_instance.preprocess()
    model_instance.split(0.2)
    lbl_enc, x_train, x_cv, x_test, y_train, y_cv = model_instance.encode()
    model_instance.fit()
    train_pred = model_instance.model.predict(x_train)
    cv_pred = model_instance.model.predict(x_cv)
    test_pred = model_instance.model.predict(x_test)
    print("Train F1_score: ", f1_score(y_train, train_pred))
    print("CV F1_score: ", f1_score(y_cv, cv_pred))

    if os.path.exists("prediction_input.csv"):
      test = pd.read_csv("prediction_input.csv")
      test_pred = lbl_enc.inverse_transform(test_pred)
      test['success_indicator'] = test_pred
      test.to_csv("predictions.csv")
      print("Predictions saved in 'predictions.csv' file")