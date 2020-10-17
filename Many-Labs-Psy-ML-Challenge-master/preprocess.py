import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from preprocess_utils import *

data = pd.read_csv('ML3AllSites.csv')

# decription of the 
print('type of the csv file is: ', type(data))
print('total number of labels is: ', len(data.columns))
# print(type(data.columns))

list_of_feature=[]

# print all labels
for item in data.columns:
    list_of_feature.append(item)
    print(item)


# we manually select some features
# most features are open questions and need to be dealt with using NLP knowledge
# encode the columns
extracted_data = data[[
                    'attention',
                    'bestgrade3',
                    'bestgrade4',
                    'bestgrade5',
                    'big5_01',
                    'big5_02',
                    'big5_03',
                    'big5_04',
                    'big5_05',
                    'big5_06',
                    'big5_07',
                    'big5_08',
                    'big5_09',
                    'big5_10',
                    'age',
                    'elm_01',
                    'elm_02',
                    'elm_03',
                    'elm_04',
                    'elm_05',
                    'intrinsic_01',
                    'intrinsic_02',
                    'intrinsic_03',
                    'intrinsic_04',
                    'intrinsic_05',
                    'intrinsic_06',
                    'intrinsic_07',
                    'intrinsic_08',
                    'intrinsic_09',
                    'intrinsic_10',
                    'intrinsic_11',
                    'intrinsic_12',
                    'intrinsic_13',
                    'intrinsic_14',
                    'intrinsic_15',
                    'nfc_01',
                    'nfc_02',
                    'nfc_03',
                    'nfc_04',
                    'nfc_05',
                    'nfc_06',
                    'pate_01',
                    'pate_02',
                    'pate_03',
                    'pate_04',
                    'pate_05',
                    'selfesteem_01',
                    'stress_01',
                    'stress_02',
                    'stress_03',
                    'stress_04',
                    'worstgrade3',
                    'worstgrade4',
                    'worstgrade5',
]]

# # print(extracted_data.head)
# deal with each feature

# first the numerical ones
attention = extracted_data['attention']
# print(attention.unique())
extracted_data['attention'] = standard_scaler(attention)


extracted_data['bestgrade3']=standard_scaler(extracted_data['bestgrade3'])
extracted_data['bestgrade4']=standard_scaler(extracted_data['bestgrade4'])
extracted_data['bestgrade5']=standard_scaler(extracted_data['bestgrade5'])
extracted_data['big5_01'] = standard_scaler(extracted_data['big5_01'])
extracted_data['big5_02'] = standard_scaler(extracted_data['big5_02'])
extracted_data['big5_03'] = standard_scaler(extracted_data['big5_03'])
extracted_data['big5_04'] = standard_scaler(extracted_data['big5_04'])
extracted_data['big5_05'] = standard_scaler(extracted_data['big5_05'])
extracted_data['big5_06'] = standard_scaler(extracted_data['big5_06'])
extracted_data['big5_07'] = standard_scaler(extracted_data['big5_07'])
extracted_data['big5_08'] = standard_scaler(extracted_data['big5_08'])
extracted_data['big5_09'] = standard_scaler(extracted_data['big5_09'])
extracted_data['big5_10'] = standard_scaler(extracted_data['big5_10'])

age = extracted_data['age']
print(type(age))
print(type(age[1]))
print(age.unique())
for index in range(len(age)):
    if isinstance(age[index], int):
        if age[index] < 0 or age[index] > 100:
            age[index] =np.nan
        else:
            age[index] == float(age[index])
    if isinstance(age[index], str):
        if age[index] == 'Too Old (18)':
            age[index] = 18.0
        if age[index] == 'we':
            age[index] = np.nan
        elif age[index] == 'almost 18':
            age[index] = 18.0
        elif age[index] == 'almost 19':
            age[index] = 19.0
        else:
            age[index] = float(str(age[index])[:2])
    elif isinstance(age[index], float):
        if age[index] < 0 or age[index] > 100:
            age[index] = np.nan
print(age.unique())
extracted_data['age'] = standard_scaler(age)

extracted_data['elm_01'] = standard_scaler(extracted_data['elm_01'])
extracted_data['elm_02'] = standard_scaler(extracted_data['elm_02'])
extracted_data['elm_03'] = standard_scaler(extracted_data['elm_03'])
extracted_data['elm_04'] = standard_scaler(extracted_data['elm_04'])
extracted_data['elm_05'] = standard_scaler(extracted_data['elm_05'])


extracted_data['elm_01'] = standard_scaler(extracted_data['elm_01'])
extracted_data['elm_02'] = standard_scaler(extracted_data['elm_02'])
extracted_data['elm_03'] = standard_scaler(extracted_data['elm_03'])
extracted_data['elm_04'] = standard_scaler(extracted_data['elm_04'])
extracted_data['elm_05'] = standard_scaler(extracted_data['elm_05'])


extracted_data['intrinsic_01'] = standard_scaler(extracted_data['intrinsic_01'])
extracted_data['intrinsic_02'] = standard_scaler(extracted_data['intrinsic_02'])
extracted_data['intrinsic_03'] = standard_scaler(extracted_data['intrinsic_03'])
extracted_data['intrinsic_04'] = standard_scaler(extracted_data['intrinsic_04'])
extracted_data['intrinsic_05'] = standard_scaler(extracted_data['intrinsic_05'])
extracted_data['intrinsic_06'] = standard_scaler(extracted_data['intrinsic_06'])
extracted_data['intrinsic_07'] = standard_scaler(extracted_data['intrinsic_01'])
extracted_data['intrinsic_08'] = standard_scaler(extracted_data['intrinsic_01'])
extracted_data['intrinsic_09'] = standard_scaler(extracted_data['intrinsic_01'])
extracted_data['intrinsic_10'] = standard_scaler(extracted_data['intrinsic_01'])
extracted_data['intrinsic_11'] = standard_scaler(extracted_data['intrinsic_11'])
extracted_data['intrinsic_12'] = standard_scaler(extracted_data['intrinsic_12'])
extracted_data['intrinsic_13'] = standard_scaler(extracted_data['intrinsic_13'])
extracted_data['intrinsic_14'] = standard_scaler(extracted_data['intrinsic_14'])
extracted_data['intrinsic_15'] = standard_scaler(extracted_data['intrinsic_15'])


extracted_data['nfc_02'] = standard_scaler(extracted_data['nfc_02'])
extracted_data['nfc_03'] = standard_scaler(extracted_data['nfc_03'])
extracted_data['nfc_04'] = standard_scaler(extracted_data['nfc_04'])
extracted_data['nfc_05'] = standard_scaler(extracted_data['nfc_05'])
extracted_data['nfc_06'] = standard_scaler(extracted_data['nfc_06'])

extracted_data['pate_01'] = standard_scaler(extracted_data['pate_01'])
extracted_data['pate_02'] = standard_scaler(extracted_data['pate_02'])
extracted_data['pate_03'] = standard_scaler(extracted_data['pate_03'])
extracted_data['pate_04'] = standard_scaler(extracted_data['pate_04'])
extracted_data['pate_05'] = standard_scaler(extracted_data['pate_05'])

extracted_data['selfesteem_01'] = standard_scaler(extracted_data['selfesteem_01'])
extracted_data['stress_01'] = standard_scaler(extracted_data['stress_01'])
extracted_data['stress_02'] = standard_scaler(extracted_data['stress_02'])
extracted_data['stress_03'] = standard_scaler(extracted_data['stress_03'])
extracted_data['stress_04'] = standard_scaler(extracted_data['stress_04'])
extracted_data['worstgrade3'] = standard_scaler(extracted_data['worstgrade3'])
extracted_data['worstgrade4'] = standard_scaler(extracted_data['worstgrade4'])
extracted_data['worstgrade5'] = standard_scaler(extracted_data['worstgrade5'])


# deal with categorical data

bestgrade1=data['bestgrade1']
# print(bestgrade1.unique())
term_2012=[]
term_2013=[]
term_2014=[]
for index in range(len(bestgrade1)):
    item = bestgrade1[index]
    item = str(item)
    if '14' in item:
        term_2014.append(1.0)
        term_2013.append(0.0)
        term_2012.append(0.0)
    elif '13' in item:
        term_2014.append(0.0)
        term_2013.append(1.0)
        term_2012.append(0.0)
    elif '12' in item:
        term_2014.append(0.0)
        term_2013.append(0.0)
        term_2012.append(1.0)
    else:
        term_2014.append(0.0)
        term_2013.append(0.0)
        term_2012.append(0.0)
df_term={'term_2012':term_2012, 'term_2013': term_2013, 'term_2014':term_2014}
df_term = pd.DataFrame(data=df_term, index=bestgrade1.index)

# print(df_term['term_2012'].unique())
# print(df_term['term_2013'].unique())
# print(df_term['term_2014'].unique())

print(extracted_data.index == df_term.index)
extracted_data.join(df_term, how='left')
print('join manipulation a success')

gender = data['gender']
# print(gender.unique())
gender_male = []
gender_female = []
for index in range(len(gender)):
    if str(gender[index])=='1':
        gender_male.append(1.0)
        gender_female.append(0.0)
    elif str(gender[index])=='2':
        gender_male.append(0.0)
        gender_female.append(1.0)
    elif 'neutral' in str(gender[index]):
        gender_male.append(1.0)
        gender_female.append(1.0)
    else:
        gender_male.append(np.nan)
        gender_female.append(np.nan)
df_gender = {'male':gender_male, 'female': gender_female}
df_gender=pd.DataFrame(data=df_gender, index=gender.index)
extracted_data.join(df_gender, how='left')
print('join manipulation a success')

year = data['year']
print(year.unique())
# one pass
year_reformed=[]
for index in range(len(year)):
    if isinstance(year[index], float) or isinstance(year[index], int):
        if np.isnan(year[index])==True:
            year_reformed.append(np.nan)
        else:
            year_reformed.append(float(year[index]))
    else:
        item = str(year[index])
        if '1' in item or 'first' in item:
            year_reformed.append(1.0)
        elif '2' in item or 'second' in item:
            year_reformed.append(2.0)
        elif 'junior' in item or '3' in item or 'Junior' in item:
            year_reformed.append(3.0)
        elif '5' in item or 'fifth' in item or 'Fifth' in item:
            year_reformed.append(5.0)
        elif '6' in item or 'Sixth' in item or 'sixth' in item:
            year_reformed.append(6.0)
        elif '4' in item or 'Senior' in item or 'senior' in item:
            year_reformed.append(4.0)
        else:
            year_reformed.append(np.nan)
# print(set(year_reformed))
extracted_data['year']=year_reformed
print('year encoding a success')

worstgrade1=data['worstgrade1']
# print(worstgrade1.unique())
worst_term_2012=[]
worst_term_2013=[]
worst_term_2014=[]
for index in range(len(worstgrade1)):
    item = worstgrade1[index]
    item = str(item)
    if '14' in item:
        worst_term_2014.append(1.0)
        worst_term_2013.append(0.0)
        worst_term_2012.append(0.0)
    elif '13' in item:
        worst_term_2014.append(0.0)
        worst_term_2013.append(1.0)
        worst_term_2012.append(0.0)
    elif '12' in item:
        worst_term_2014.append(0.0)
        worst_term_2013.append(0.0)
        worst_term_2012.append(1.0)
    else:
        worst_term_2014.append(0.0)
        worst_term_2013.append(0.0)
        worst_term_2012.append(0.0)
df_worst_term={'worst_term_2012':worst_term_2012, 'worst_term_2013': worst_term_2013, 'worst_term_2014':worst_term_2014}
df_worst_term = pd.DataFrame(data=df_worst_term, index=worstgrade1.index)

# print(df_worst_term['worst_term_2012'].unique())
# print(df_worst_term['worst_term_2013'].unique())
# print(df_worst_term['worst_term_2014'].unique())

# print(extracted_data.index == df_worst_term.index)
extracted_data.join(df_worst_term, how='left')
print('join manipulation a success')


# save the data into csv file
extracted_data.to_csv('whatever_you_want_to_call_it.csv')