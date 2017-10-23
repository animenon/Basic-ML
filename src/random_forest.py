import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
# from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
# To avoid pandas warnings
pd.options.mode.chained_assignment = None

# To write model to a file
from sklearn.externals import joblib

data=pd.read_csv("C:\\Users\\MENOA007\\Documents\\my_repo\\Basic-ML\\resources\\titanic\\titanic_train.csv")

# to print first 5 rows
print(data.head())
"""
   Unnamed: 0  row.names pclass  survived                         name   age      embarked             home.dest room ticket   boat     sex
0         998        999    3rd         1         McCarthy, Miss Katie  29.0           NaN                   NaN  NaN    NaN    NaN  female
1         179        180    1st         0     Millet, Mr Francis Davis  65.0   Southampton  East Bridgewater, MA  NaN    NaN  (249)    male
2         556        557    2nd         0     Sjostedt, Mr Ernst Adolf  59.0   Southampton    Sault St Marie, ON  NaN    NaN    NaN    male
3         174        175    1st         0  McCaffry, Mr Thomas Francis  46.0     Cherbourg         Vancouver, BC  NaN    NaN  (292)    male
4        1232       1233    3rd         0             Strilic, Mr Ivan  29.0           NaN                   NaN  NaN    NaN    NaN    male
"""

data_inputs = data[["pclass", "age", "sex"]]
"""
>>> data_inputs.head()
    pclass   age     sex
0      3rd  29.0  female
1      1st  65.0    male
2      2nd  59.0    male
3      1st  46.0    male
4      3rd  29.0    male
""" 

expected_output = data[["survived"]]
expected_output.head()
"""
>>> expected_output.head()
   survived
0         1
1         0
2         0
3         0
4         0
"""

# replace 3rd, 1st and 2nd with 3, 1 and 2
data_inputs['pclass']=data_inputs['pclass'].astype(str).str[0]
"""
>>> data_inputs.head()
  pclass   age     sex
0      3  29.0  female
1      1  65.0    male
2      2  59.0    male
3      1  46.0    male
4      3  29.0    male
"""

# replace female with 0 and male with 1
data_inputs["sex"] = np.where(data_inputs["sex"] == "female", 0, 1)
"""
>>> data_inputs.head()
  pclass   age  sex
0      3  29.0    0
1      1  65.0    1
2      2  59.0    1
3      1  46.0    1
4      3  29.0    1
"""

# Split data into Test/Train sets to prevent overfitting
inputs_train, inputs_test, expected_output_train, expected_output_test =  train_test_split (data_inputs, expected_output, test_size = 0.33, random_state = 42)
"""
>>> inputs_train.head()
    pclass   age  sex
618      3  19.0    1
169      3  29.0    1
830      1  54.0    1
140      3  29.0    1
173      2  28.0    1

>>> inputs_train.head()
    pclass   age  sex
618      3  19.0    1
169      3  29.0    1
830      1  54.0    1
140      3  29.0    1
173      2  28.0    1

>>> expected_output_train.head()
     survived
618         0
169         0
830         1
140         0
173         0

>>> expected_output_test.head()
     survived
72          1
120         1
296         0
314         1
710         1
"""

# Machine Learning using RandomForestClassifier

rf = RandomForestClassifier (n_estimators=100)

# provide the random forest instance with data to learn
rf.fit(inputs_train, expected_output_train)
"""
Model created-
RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=None, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=1,
            oob_score=False, random_state=None, verbose=0,
            warm_start=False)
"""

# Accuracy
print(rf.score(inputs_test, expected_output_test))
""" 0.79276315789473684 """


# Save Model in pickle
joblib.dump(rf, "titanic_RF_model", compress=9)

