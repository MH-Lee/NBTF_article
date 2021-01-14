import pandas as pd
import warnings
import re
from datetime import datetime, timedelta
warnings.filterwarnings(action='ignore')

data1 = pd.read_excel("./data_final.xlsx")
data1['Documents'] = data1['Overview'] + "\n" + data1['Service_description']
data1['Documents'] = data1['Documents'].apply(lambda x: str(x).strip())

data1['Textlen'] = data1['Documents'].apply(lambda x: len(x))
data1_re = data1[data1['Textlen'] > 200]

regex = re.compile(r'\d\d\d\d-\d\d-\d\d')
data1_re['Established'] = data1_re['Sub_info'].apply(lambda x: regex.findall(str(x)))
data1_re['Established'] = data1_re['Established'].apply(lambda x: "".join(x))

data1_re.columns

data1_re[['Company', 'Documents', 'Fund_info', 'Textlen', 'Established']].to_excel('final_data2.xlsx', index=False)

data1 = pd.read_excel("./startup_data_new2.xlsx")
data1.columns
data1['Textlen'] = data1['document'].apply(lambda x: len(x))
data1_re = data1[data1['Textlen'] > 200]

regex = re.compile(r'\d\d\d\d-\d\d-\d\d')
data1_re['Established'] = data1_re['Sub_info'].apply(lambda x: regex.findall(str(x)))
data1_re['Established'] = data1_re['Established'].apply(lambda x: "".join(x))

data1_re.columns

data1_re.columns = ['Company', 'Documents', 'Fund_info', 'Sub_info', 'Textlen', 'Established']

data1_re[['Company', 'Documents', 'Fund_info', 'Textlen', 'Established']].to_excel('startup_data3.xlsx', index=False)

from ast import literal_eval

total_data = pd.read_excel('./total_data1.xlsx')
total_data.drop_duplicates("Company", inplace=True)
total_data['Fund_info'] = total_data['Fund_info'].apply(lambda x : literal_eval(x))

def make_investor_col(x):
    if len(x) == 0:
        return ""
    else:
        result = ""
        for idx, el in enumerate(x):
            if idx == 0:
                tmp = ", ".join(el['header'])
                result = result + tmp
            else:
                tmp = ", ".join(el['header'])
                if tmp == "":
                    continue
                else:
                    result = result + ", " + tmp
        return result

total_data['investor'] = total_data['Fund_info'].apply(make_investor_col)
total_data.to_excel('./total_data2.xlsx', index=False)
