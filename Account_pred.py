import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score

data = pd.read_csv('Financial_inclusion_dataset.csv')
print(data)

# THERE IS NO MISSING AND CORRUPTED VALUES IN THE DATASET

# BASED ON THE FACT THAT THE MOST FEATURES OF OUR DATASET IS OBJECT VARIABLE WE CAN NOT HAVE OUTLIERS IN THIS DATASET

# CATEGORICAL FEATURES ENCODING
needed_data = data.drop('uniqueid', axis=1)
print(needed_data)

# convert bank_account variable into numerical
# One hot encoding
needed_data_hot = pd.get_dummies(needed_data, columns=['country', 'year', 'location_type', 'cellphone_access',
                                                       'gender_of_respondent'], prefix=['country', 'year',
                                                                                        'location_type',
                                                                                        'cellphone_access',
                                                                                        'gender_of_respondent'])
print(needed_data_hot)

# the output feature mapping
needed_data_hot["bank_account"] = needed_data_hot["bank_account"].map({"Yes": 1, "No": 0})

label_encoder = LabelEncoder()

needed_data_hot['relationship_with_head'] = label_encoder.fit_transform(needed_data_hot['relationship_with_head'])
needed_data_hot['marital_status'] = label_encoder.fit_transform(needed_data_hot['marital_status'])
needed_data_hot['education_level'] = label_encoder.fit_transform(needed_data_hot['education_level'])
needed_data_hot['job_type'] = label_encoder.fit_transform(needed_data_hot['job_type'])
print(needed_data_hot)


def_data = needed_data_hot

# FEATURES EXTRACTIONS
y = def_data['bank_account']
X = def_data.drop('bank_account', axis = 1)

# DATA SPLITTING
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size = 0.2, random_state= 75)


# BUILD OUR LOGIC MODEL
logreg = LogisticRegression()

# FIT OUR TRAINING DATA
i = logreg.fit(X_train, y_train)

# TEST OUR MODEL PERFORMANCE
y_test_pred = logreg.predict(X_test)
print(y_test_pred)

# MODEL EVALUATION
logreg_test_mean = mean_squared_error(y_test, y_test_pred)
logreg_test_r2 = r2_score(y_test, y_test_pred)
print('logreg_test_mean :', logreg_test_mean)
print('logreg_test_r2 :', logreg_test_r2)

# APPLICATION CONSTRUCTION

st.title('HOLDING ACCOUNT PREDICTION')

# Entrées des données qui serviront pour la prédiction
st.write('EVERY CASE SHOULD BE COMPLETED')
st.write('CHOOSE THE COUNTRY IN WITCH YOU WANT TO MAKE YOUR PREDICTION BY MATCHING THE APPROPRIATE COUNTRY WTIH 1')
country_Kenya = st.number_input('KENYA')
country_Rwanda = st.number_input('RWANDA')
country_Tanzania = st.number_input('TANZANIA')
country_Uganda = st.number_input('UGANDA')
st.write('CHOOSE THE YEAR IN WITCH YOU WANT TO MAKE YOUR PREDICTION BY MATCHING THE APPROPRIATE YEAR WITH 1')
year_2016 = st.number_input('2016')
year_2017 = st.number_input('2017')
year_2018 = st.number_input('2018')
st.write('CHOOSE THE LOCATION TYPE WITH 1 FOR THE APPROPRIATE TYPE')
location_type_Rural = st.number_input('RURAL')
location_type_Urban = st.number_input('URBAN')
st.write('CHOOSE IF THE RESPONDENT HAS A CELL PHONE BY MATCHING THE APPROPRIATE CASE WITH 1')
cellphone_access_No = st.number_input('NO')
cellphone_access_Yes = st.number_input('YES')
st.write('CHOOSE THE HOUSEHOLD SIZE BY INPUTTING A NUMBER')
household_size = st.number_input('HOUSEHOLD_SIZE')
age_of_respondent = st.number_input('AGE_OF_RESPONDENT')
st.write('CHOOSE THE APPROPRIATE GENDER BY INPUTTING 1')
gender_of_respondent_Female = st.number_input('FEMALE')
gender_of_respondent_Male = st.number_input('MALE')
st.write('CHOOSE THE APPROPRIATE RELATIONSHIP: Spouse:5 , Head of Household:1 , Other relative:3 , Child:0 , Parent:4 , Other non-relatives:2')
relationship_with_head = st.number_input('RELATIONSHIP_WITH_HEAD')
st.write('CHOOSE THE MARITAL STATUS: Married/Living together:2, Widowed:4 , Single/Never Married:3 , Divorced/Seperated:0 , Dont know:1')
marital_status = st.number_input('MARITAL_STATUS')
st.write('CHOOSE THE EDUCATION LEVEL OF THE RESPONDENT: Secondary education:3 , No formal education:0 , Vocational/Specialised training:5 , Primary education:2 ,Tertiary education:4 , Other/Dont know/RTA:1')
education_level = st.number_input('EDUCATION_LEVEL')
st.write('CHOOSE THE JOB TYPE: Self employed:9 , Government Dependent:4 ,Formally employed Private:3 , Informally employed:5 ,Formally employed Government:2 , Farming and Fishing:1 ,Remittance Dependent:8 , Other Income:7 ,Dont Know/Refuse to answer:0 , No Income:6')
job_type = st.number_input('JOB_TYPE')
# Faire la prédiction
if st.button('PREDICT'):
    data = [[household_size, age_of_respondent, relationship_with_head, marital_status, education_level, job_type,
             country_Kenya, country_Rwanda, country_Tanzania, country_Uganda, year_2016, year_2017, year_2018,
             location_type_Rural, location_type_Urban, cellphone_access_No, cellphone_access_Yes,
             gender_of_respondent_Female, gender_of_respondent_Male]]
    predict = logreg.predict(data)
    if predict == 0:
        st.success(f"This person does not have an account")
    else:
        st.success(f"This person has an account")



