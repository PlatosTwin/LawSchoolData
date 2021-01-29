# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import datetime as dt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics


def datetime_to_float(d):
    """Convert yyyy-mm-dd (datetime.date) to float (seconds)."""
    total_seconds = (d - epoch).total_seconds()
    return total_seconds


#  Seconds in a day
sec_day = 60*60*24

#  Use 2015 as a date benchmark
epoch = dt.date(2015, 1, 1)

#  Suppress Pandas SettingWithCopyWarning
pd.options.mode.chained_assignment = None

#  Time-delimit the data
x_min = datetime_to_float(dt.date(2019, 8, 1))
x_max = datetime_to_float(dt.date(2020, 5, 31))

#  Read-in data, downloaded from https://www.lawschooldata.org/download
filename = '/Users/lsdata.csv'
df = pd.read_csv(filename, skiprows=1, low_memory=False)

print("\nShape of original file: " + str(df.shape))

#  Drop unnecessary/uninteresting columns
drop_cols = ['scholarship', 'attendance', 'is_in_state', 'is_fee_waived', 'is_conditional_scholarship',
             'is_international', 'international_gpa', 'is_lsn_import']
df_dcol = df.drop(drop_cols, axis=1)

#  Remove rows that are missing crucial values
filter_rows = ['sent_at', 'decision_at', 'lsat', 'gpa', 'years_out', 'cycle_id']
df_filtered = df_dcol.dropna(subset=filter_rows)

#  Convert sent_at and decision_at to float
df_filtered.loc[:, 'sent_at'] = \
    df_filtered['sent_at'].map(lambda x: datetime_to_float(dt.date(int(x[0:4]), int(x[5:7]), int(x[8:10]))))
df_filtered.loc[:, 'decision_at'] = \
    df_filtered['decision_at'].map(lambda x: datetime_to_float(dt.date(int(x[0:4]), int(x[5:7]), int(x[8:10]))))

#  Remove all dates older than x_min and more recent than x_max
df_filtered = df_filtered[df_filtered['sent_at'] > x_min]
df_filtered = df_filtered[df_filtered['decision_at'] < x_max]

#  Format 'softs' column as int, removing leading "T" (NOTE: will not work if nan values are not filtered out of 'softs'
# df_filtered.loc[:, 'softs'] = df_filtered['softs'].map(lambda x: int(x[1]))

#  Filter by status: accepted, rejected, waitlisted, etc.
# df_filtered = df_filtered[df_filtered['result'] == 'Accepted']

print("Shape of trimmed and filtered file: " + str(df_filtered.shape))

#####

#  Create data sets for individual schools
harvard = df_filtered[df_filtered['school_name'] == 'Harvard University']
stanford = df_filtered[df_filtered['school_name'] == 'Stanford University']
yale = df_filtered[df_filtered['school_name'] == 'Yale University']
chicago = df_filtered[df_filtered['school_name'] == 'University of Chicago']
columbia = df_filtered[df_filtered['school_name'] == 'Columbia University']
nyu = df_filtered[df_filtered['school_name'] == 'New York University']
penn = df_filtered[df_filtered['school_name'] == 'University of Pennsylvania']
virginia = df_filtered[df_filtered['school_name'] == 'University of Virginia']
michigan = df_filtered[df_filtered['school_name'] == 'University of Michigan']
berkeley = df_filtered[df_filtered['school_name'] == 'University of California—Berkeley']

top_ten_list = ['Harvard University', 'Stanford University', 'Yale University', 'University of Chicago',
                'Columbia University', 'New York University', 'University of Pennsylvania', 'University of Virginia',
                'University of Michigan']
top_ten = df_filtered[df_filtered['school_name'].str.contains('|'.join(top_ten_list))]

school = top_ten

print("\nNumber of samples for chosen school: %i" % (school.shape[0]))

#  Length of wait, from sent_at to decision_at
duration = (school['decision_at'] - school['sent_at'])/sec_day

print("Average wait: %.2f" % duration.mean())

#####

#  Set up for RandomForest
X = school.iloc[:, [2, 10, 14]].values  # features: sent_at, lsat, gpa (softs index: 11)
y = duration.values  # target: length of wait from sent_at to decision_at

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)

#  Scale features values
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

regressor = RandomForestRegressor(n_estimators=50, random_state=13)
regressor.fit(X_train, y_train)

#  Assess model
y_pred = regressor.predict(X_test)
print("\nMean Absolute Error: %.2f" % metrics.mean_absolute_error(y_test, y_pred))
print("Mean Squared Error: %.2f" % metrics.mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error: %.2f" % np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

#  Predict given input stats: sent_at, lsat, gpa
to_predict = sc.transform(np.array([datetime_to_float(dt.date(2019, 11, 13)), 171, 3.89]).reshape(1, -1))
print("\nPrediction: %.2f" % regressor.predict(to_predict)[0])

#####

#  Set up 3D plot
fig = plt.figure()
ax = plt.axes(projection='3d')

ax.set_xlim(x_min, x_max)
ax.set_xlabel('Time')
ax.set_ylim(school['gpa'].min()*0.90, school['gpa'].max()*1.1)
ax.set_ylabel('GPA')
ax.set_zlim(school['lsat'].min()*0.99, school['lsat'].max()*1.01)
ax.set_zlabel('LSAT')

x = school['sent_at']
y = school['gpa']
z = school['lsat']
ax.scatter(x, y, z, c='yellow')

x = school['decision_at']
y = school['gpa']
z = school['lsat']
ax.scatter(x, y, z, c='blue')

#  Draw lines connecting sent_at and decision_at points, color-coded by status
for applicant in school.itertuples():
    sent = applicant[3]
    decision = applicant[9]
    gpa = applicant[15]
    lsat = applicant[11]
    result = applicant[10]

    color = 'black'
    if result == 'Accepted':
        color = 'green'
    elif result == 'Rejected':
        color = 'red'
    elif 'Wait' in result:
        color = 'orange'

    ax.plot3D([sent, decision], [gpa, gpa], [lsat, lsat], c=color)

plt.show()
