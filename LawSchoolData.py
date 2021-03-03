# -*- coding: utf-8 -*-

import numpy as np
import datetime as dt

import pandas as pd
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import Axes3D

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics


#  Suppress Pandas SettingWithCopyWarning
pd.options.mode.chained_assignment = None

#  Time-delimit the data
t_min = "10/01/2017"
t_max = "08/31/2021"

#  Read-in data, downloaded from https://www.lawschooldata.org/download
filename = '/Users/Desktop/lsdata.csv'
df = pd.read_csv(filename, skiprows=1, low_memory=False)

print("\nShape of original file: " + str(df.shape))

#  Drop unnecessary/uninteresting columns
drop_cols = ['scholarship', 'attendance', 'is_in_state', 'is_fee_waived', 'is_conditional_scholarship',
             'is_international', 'international_gpa', 'is_lsn_import']
df_dcol = df.drop(drop_cols, axis=1)

#  Remove rows that are missing crucial values
filter_rows = ['sent_at', 'decision_at', 'lsat', 'gpa', 'result']
df_filtered = df_dcol.dropna(subset=filter_rows)

#  Convert sent_at and decision_at to datetime
df_filtered.loc[:, 'sent_at'] = pd.to_datetime(df_filtered['sent_at'])
df_filtered.loc[:, 'decision_at'] = pd.to_datetime(df_filtered['decision_at'])

#  Filter such that data falls within t_min and t_max range
df_filtered = df_filtered[(df_filtered['decision_at'] > t_min) & (df_filtered['decision_at'] < t_max)]

#  Format 'softs' column as int, removing leading "T" (NOTE: will not work if nan values are not filtered out of 'softs'
# df_filtered.loc[:, 'softs'] = df_filtered['softs'].map(lambda x: int(x[1]))

#  Filter by status: accepted, rejected, waitlisted, etc.
# df_filtered = df_filtered[df_filtered['result'] == 'Accepted']

print("Shape of trimmed and filtered file: " + str(df_filtered.shape))

#####

#  Create data sets for individual schools
yale = df_filtered[df_filtered['school_name'] == 'Yale University']
harvard = df_filtered[df_filtered['school_name'] == 'Harvard University']
stanford = df_filtered[df_filtered['school_name'] == 'Stanford University']
chicago = df_filtered[df_filtered['school_name'] == 'University of Chicago']
columbia = df_filtered[df_filtered['school_name'] == 'Columbia University']
nyu = df_filtered[df_filtered['school_name'] == 'New York University']
penn = df_filtered[df_filtered['school_name'] == 'University of Pennsylvania']
virginia = df_filtered[df_filtered['school_name'] == 'University of Virginia']
michigan = df_filtered[df_filtered['school_name'] == 'University of Michigan']
berkeley = df_filtered[df_filtered['school_name'] == 'University of California—Berkeley']

top_ten_list = ['Yale University', 'Harvard University', 'Stanford University', 'University of Chicago',
                'Columbia University', 'New York University', 'University of Pennsylvania', 'University of Virginia',
                'University of Michigan']
top_ten = df_filtered[df_filtered['school_name'].str.contains('|'.join(top_ten_list))]

top_four_list = ['Yale University', 'Harvard University', 'Stanford University', 'University of Chicago']
top_four = df_filtered[df_filtered['school_name'].str.contains('|'.join(top_four_list))]

school = columbia

print("\nNumber of samples for chosen school: %i" % (school.shape[0]))

#  Length of wait, from sent_at to decision_at, in days
duration = ((school['decision_at'] - school['sent_at'])/np.timedelta64(1, 's'))/(60*60*24)

print("Average wait: %0.f" % duration.mean())

#####
#  Plot sent_at vs. decision_at, stacking cycles
#####

school_stack = school.copy()

cycle18 = school_stack[(school_stack['decision_at'] > '10/01/2017') & (school_stack['decision_at'] < '08/31/2018')]
cycle19 = school_stack[(school_stack['decision_at'] > '10/01/2018') & (school_stack['decision_at'] < '08/31/2019')]
cycle20 = school_stack[(school_stack['decision_at'] > '10/01/2019') & (school_stack['decision_at'] < '08/31/2020')]
cycle21 = school_stack[(school_stack['decision_at'] > '10/01/2020') & (school_stack['decision_at'] < '08/31/2021')]

#  Account for 2020 being a leap year
cycle20.loc[cycle20['sent_at'] == '02/29/2020', 'sent_at'] = dt.datetime(2020, 02, 28)
cycle20.loc[cycle20['decision_at'] == '02/29/2020', 'decision_at'] = dt.datetime(2020, 02, 28)

cycle19.loc[:, 'sent_at'] = cycle19['sent_at'].map(lambda x: dt.datetime(x.year-1, x.month, x.day))
cycle20.loc[:, 'sent_at'] = cycle20['sent_at'].map(lambda x: dt.datetime(x.year-2, x.month, x.day))
cycle21.loc[:, 'sent_at'] = cycle21['sent_at'].map(lambda x: dt.datetime(x.year-3, x.month, x.day))

cycle19.loc[:, 'decision_at'] = cycle19['decision_at'].map(lambda x: dt.datetime(x.year-1, x.month, x.day))
cycle20.loc[:, 'decision_at'] = cycle20['decision_at'].map(lambda x: dt.datetime(x.year-2, x.month, x.day))
cycle21.loc[:, 'decision_at'] = cycle21['decision_at'].map(lambda x: dt.datetime(x.year-3, x.month, x.day))

#  Plot sent_at vs. decision_at, stacking cycles
markers = ['1', '3', '2', '4']
cycle = ['18', '19', '20', '21']

for i, mark in enumerate(markers):

    school_temp = None
    exec('school_temp = ' + 'cycle' + cycle[i])

    accepted = school_temp[school_temp['result'] == 'Accepted']
    rejected = school_temp[school_temp['result'] == 'Rejected']
    waitlisted = school_temp[school_temp['result'].str.contains('Wait')]

    plt.scatter(accepted['sent_at'], accepted['decision_at'],
                color='green', marker=mark, label="Accepted 20" + cycle[i])
    plt.scatter(rejected['sent_at'], rejected['decision_at'],
                color='red', marker=mark, label="Rejected 20" + cycle[i])
    plt.scatter(waitlisted['sent_at'], waitlisted['decision_at'],
                color='orange', marker=mark, label="Waitlisted 20" + cycle[i])

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
plt.gca().xaxis.set_minor_locator(mdates.WeekdayLocator(interval=1))

plt.gca().yaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
plt.gca().yaxis.set_major_locator(mdates.MonthLocator())
plt.gca().yaxis.set_minor_locator(mdates.WeekdayLocator(interval=1))

plt.xlabel("Date Sent")
plt.ylabel("Date Decision Received")

plt.legend()

plt.show()

#####
#  Random Forest, to predict wait time
#####

#  Set up for RandomForest
school_date_float = school.copy()
school_date_float.loc[:, 'sent_at'] = school_date_float['sent_at'].map(lambda x: mdates.date2num(x))
X = school_date_float.iloc[:, [2, 10, 14]].values  # Features: sent_at, lsat, gpa (softs index: 11)
y = duration.values  # Target: length of wait from sent_at to decision_at

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)

#  Scale features values
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#  Fit data
regressor = RandomForestRegressor(n_estimators=50, random_state=13)
regressor.fit(X_train, y_train)

#  Assess model
y_pred = regressor.predict(X_test)
print("\nMean Absolute Error: %.2f" % metrics.mean_absolute_error(y_test, y_pred))
print("Mean Squared Error: %.2f" % metrics.mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error: %.2f" % np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

#  Predict given input stats: sent_at, lsat, gpa
to_predict = sc.transform(np.array([mdates.date2num(dt.datetime(2019, 11, 13)), 171, 3.89]).reshape(1, -1))
print("\nPrediction: %.2f" % regressor.predict(to_predict)[0])

#####
#  Date slider (does not stack cycles)
#####

#  Setup to date/slider plot
fig, ax = plt.subplots()
plt.subplots_adjust(left=0.1, bottom=0.20)

#  Set initial values
initial_time = school['decision_at'].min()
time_delim_init = school[school['decision_at'] <= initial_time]

#  Establish scatter plot
scat = ax.scatter(time_delim_init['decision_at'], time_delim_init['lsat'])

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
plt.gca().xaxis.set_minor_locator(mdates.WeekdayLocator(interval=1))
plt.gcf().autofmt_xdate()
ax.set_xlim([school['decision_at'].min() - dt.timedelta(days=7), school['decision_at'].max() + dt.timedelta(days=7)])

plt.ylim([140, 181])

#  Define slider
ax_slider = plt.axes([0.15, 0.05, 0.65, 0.03])
slider = Slider(ax_slider, 'Date', mdates.date2num(school['decision_at'].min()),
                mdates.date2num(school['decision_at'].max()),
                valinit=mdates.date2num(initial_time),
                valfmt="%i")

#  Update function, called upon slider movement
#  (splitter/rev_splitter colors show up incorrectly if slider is dragged from beginning; click in middle, instead)
def update(val):
    time_delim = school[school['decision_at'] <= mdates.num2date(val).replace(tzinfo=None)]

    splitters = np.intersect1d(np.where(time_delim['lsat'] > 175), np.where(time_delim['gpa'] < 3.78))
    rev_splitters = np.intersect1d(np.where(time_delim['lsat'] < 170), np.where(time_delim['gpa'] > 3.95))

    print("\nNumber of splitters: %i " % len(splitters))
    print("Number of reverse splitters: %i " % len(rev_splitters))

    colors_new = np.zeros(time_delim.shape[0])
    colors_new[[splitters]] = 0.9
    colors_new[[rev_splitters]] = 0.4
    scat.set_array(colors_new)

    time_delim.loc[:, 'decision_at'] = time_delim['decision_at'].map(lambda x: mdates.date2num(x))

    xx = np.vstack((time_delim['decision_at'], time_delim['lsat']))
    scat.set_offsets(xx.T)


#  Call update function on slider value change
slider.on_changed(update)

plt.show()

#####
#  3D plotting (does not display human-readable dates on x axis)
#####

#  Set up 3D plot
# fig = plt.figure()
# ax = plt.axes(projection='3d')
#
# school_date_num = school.copy()
# school_date_num.loc[:, 'sent_at'] = school_date_num['sent_at'].map(lambda x: mdates.date2num(x))
# school_date_num.loc[:, 'decision_at'] = school_date_num['decision_at'].map(lambda x: mdates.date2num(x))
#
# # ax.set_xlim(t_min, t_max)
# ax.set_xlabel('Time')
# ax.set_ylim(school['gpa'].min()*0.90, school['gpa'].max()*1.1)
# ax.set_ylabel('GPA')
# ax.set_zlim(school['lsat'].min()*0.99, school['lsat'].max()*1.01)
# ax.set_zlabel('LSAT')
#
# x = school_date_num['sent_at']
# y = school['gpa']
# z = school['lsat']
# ax.scatter(x, y, z, c='yellow')
#
# x = school_date_num['decision_at']
# y = school['gpa']
# z = school['lsat']
# ax.scatter(x, y, z, c='blue')
#
# #  Draw lines connecting sent_at and decision_at points, color-coded by status
# for applicant in school_date_num.itertuples():
#     sent = applicant[3]
#     # sent = dt.datetime(sent.year, sent.month, sent.day)
#     decision = applicant[9]
#     # decision = dt.datetime(decision.year, decision.month, decision.day)
#     gpa = applicant[15]
#     lsat = applicant[11]
#     result = applicant[10]
#
#     color = 'black'
#     if result == 'Accepted':
#         color = 'green'
#     elif result == 'Rejected':
#         color = 'red'
#     elif 'Wait' in result:
#         color = 'orange'
#
#     ax.plot3D([sent, decision], [gpa, gpa], [lsat, lsat], c=color)
#
# plt.show()
