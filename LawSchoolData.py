# -*- coding: utf-8 -*-

import numpy as np
import datetime as dt

import pandas as pd
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.dates as mdates
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as c

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

school_name = 'Stanford'  # Select the school to analyze, below

#  Suppress Pandas SettingWithCopyWarning
pd.options.mode.chained_assignment = None

#  Time-delimit the data (used to delimit 'decision_at')
t_min = "10/01/2017"
t_max = "08/31/2021"

#  Read-in medians data, downloaded from https://7sage.com/top-law-school-admissions/
filename_percentiles = '/Users/johnsmith/Desktop/lsmedians.csv'
dff = pd.read_csv(filename_percentiles, low_memory=False)
dff = dff[:10]  # Limit to top ten schools

#  Read-in admissions data, downloaded from https://www.lawschooldata.org/download
filename_admitdata = '/Users/johnsmith/Desktop/lsdata.csv'
df = pd.read_csv(filename_admitdata, skiprows=1, low_memory=False)

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

#  Get school admissions medians
if school_name != 'NYU':
    school_percentiles = dff[dff['School'].str.contains(school_name[1:])]
else:
    school_percentiles = dff[dff['School'] == 'New York University']

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

school = None
exec('school = ' + school_name.lower())

print("\nNumber of samples for chosen school: %i" % (school.shape[0]))

#  Length of wait, from sent_at to decision_at, in days
duration = ((school['decision_at'] - school['sent_at'])/np.timedelta64(1, 's'))/(60*60*24)

print("Average wait: %0.f" % duration.mean())


def split_revsplit(input):
    splits = np.intersect1d(np.where(input['lsat'] > school_percentiles['L75'].values[0]),
                               np.where(input['gpa'] < school_percentiles['G25'].values[0]))
    rev_splits = np.intersect1d(np.where(input['lsat'] < school_percentiles['L25'].values[0]),
                                   np.where(input['gpa'] > school_percentiles['G75'].values[0]))

    return splits, rev_splits

#####
#  Plot sent_at vs. decision_at, stacking cycles
#####

school_stack = school.copy()

#  Break data down by cycles
cycle18 = school_stack[(school_stack['decision_at'] > '10/01/2017') & (school_stack['decision_at'] < '08/31/2018')]
cycle19 = school_stack[(school_stack['decision_at'] > '10/01/2018') & (school_stack['decision_at'] < '08/31/2019')]
cycle20 = school_stack[(school_stack['decision_at'] > '10/01/2019') & (school_stack['decision_at'] < '08/31/2020')]
cycle21 = school_stack[(school_stack['decision_at'] > '10/01/2020') & (school_stack['decision_at'] < '08/31/2021')]

#  Account for 2020 being a leap year
cycle20.loc[cycle20['sent_at'] == '02/29/2020', 'sent_at'] = dt.datetime(2020, 02, 28)
cycle20.loc[cycle20['decision_at'] == '02/29/2020', 'decision_at'] = dt.datetime(2020, 02, 28)

#  Standardize cycle years, sent_at
cycle19.loc[:, 'sent_at'] = cycle19['sent_at'].map(lambda x: dt.datetime(x.year-1, x.month, x.day))
cycle20.loc[:, 'sent_at'] = cycle20['sent_at'].map(lambda x: dt.datetime(x.year-2, x.month, x.day))
cycle21.loc[:, 'sent_at'] = cycle21['sent_at'].map(lambda x: dt.datetime(x.year-3, x.month, x.day))

#  Standardize cycle years, sent_at
cycle19.loc[:, 'decision_at'] = cycle19['decision_at'].map(lambda x: dt.datetime(x.year-1, x.month, x.day))
cycle20.loc[:, 'decision_at'] = cycle20['decision_at'].map(lambda x: dt.datetime(x.year-2, x.month, x.day))
cycle21.loc[:, 'decision_at'] = cycle21['decision_at'].map(lambda x: dt.datetime(x.year-3, x.month, x.day))

school_stack = pd.concat([cycle18, cycle19, cycle20, cycle21])

#  Plot sent_at vs. decision_at, stacking cycles
markers = ['v', '^', '<', 'o']
cycles = ['18', '19', '20', '21']

fig, ax = plt.subplots()

for i, mark in enumerate(markers):

    cycle = None
    exec('cycle = ' + 'cycle' + cycles[i])

    accepted = cycle[cycle['result'] == 'Accepted']
    rejected = cycle[cycle['result'] == 'Rejected']
    waitlisted = cycle[cycle['result'].str.contains('Wait')]

    splitters_a, rev_splitters_a = split_revsplit(accepted)
    splitters_r, rev_splitters_r = split_revsplit(rejected)
    splitters_w, rev_splitters_w = split_revsplit(waitlisted)

    edge_colors_a = np.array(['green'] * accepted.shape[0], dtype=object)
    edge_colors_a[splitters_a] = 'b'
    edge_colors_a[rev_splitters_a] = 'k'

    edge_colors_r = np.array(['red'] * rejected.shape[0], dtype=object)
    edge_colors_r[splitters_r] = 'b'
    edge_colors_r[rev_splitters_r] = 'k'

    edge_colors_w = np.array(['orange'] * waitlisted.shape[0], dtype=object)
    edge_colors_w[splitters_w] = 'b'
    edge_colors_w[rev_splitters_w] = 'k'

    ax.scatter(accepted['sent_at'], accepted['decision_at'],
                color='green', edgecolors=edge_colors_a, marker=mark, label="A 20" + cycles[i], s=17, zorder=3)
    ax.scatter(rejected['sent_at'], rejected['decision_at'],
                color='red', edgecolors=edge_colors_r, marker=mark, label="R 20" + cycles[i], s=17, zorder=3)
    ax.scatter(waitlisted['sent_at'], waitlisted['decision_at'],
                color='orange', edgecolors=edge_colors_w, marker=mark, label="W 20" + cycles[i], s=17, zorder=3)

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
plt.gca().xaxis.set_minor_locator(mdates.WeekdayLocator(interval=1))

plt.gca().yaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
plt.gca().yaxis.set_major_locator(mdates.MonthLocator())
plt.gca().yaxis.set_minor_locator(mdates.WeekdayLocator(interval=1))

plt.xlabel("Date Sent")
plt.ylabel("Date Decision Received")
plt.title("Admissions Data for 20" + str(int(cycles[0])-1) + "-20" + cycles[0] +
          " to 20" + str(int(cycles[-1])-1) + "-20" + cycles[-1] + ", " + school_name +
          " (" + str(school_stack.shape[0]) + " samples)")

custom_markers = [Line2D([0], [0], marker=markers[0], markerfacecolor='grey', markeredgecolor='grey', markersize=7, ls=''),
                  Line2D([0], [0], marker=markers[1], markerfacecolor='grey', markeredgecolor='grey', markersize=7, ls=''),
                  Line2D([0], [0], marker=markers[2], markerfacecolor='grey', markeredgecolor='grey', markersize=7, ls=''),
                  Line2D([0], [0], marker=markers[3], markerfacecolor='grey', markeredgecolor='grey', markersize=7, ls=''),
                  Line2D([0], [0], marker='o', markerfacecolor='white', markeredgecolor='b', markersize=7, ls=''),
                  Line2D([0], [0], marker='o', markerfacecolor='white', markeredgecolor='k', markersize=7, ls='')]
ax.legend(custom_markers, [str(int(cycles[0])-1) + '/' + cycles[0],
                           str(int(cycles[1])-1) + '/' + cycles[1],
                           str(int(cycles[2])-1) + '/' + cycles[2],
                           str(int(cycles[3])-1) + '/' + cycles[3],
                           'Splitters',
                           'Reverse Splitters'])

plt.axhline(y=dt.datetime.strptime(max(df['decision_at']), '%Y-%m-%d')-dt.timedelta(days=365*3),
            linewidth=1, color='gray', linestyle='--', zorder=2)

plt.grid(zorder=0)

ax.annotate('Current as of ' + max(df['decision_at']) + ' (-----)',
            xy=(1, 0), xytext=(0, 5),
            xycoords=('axes fraction', 'figure fraction'),
            textcoords='offset points',
            size=7, color='gray', ha='right', va='bottom')

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
#  Date slider with splitters/revsplitters, stacking cycles
#####

#  Setup to date/slider plot
fig, ax = plt.subplots()
plt.subplots_adjust(left=0.1, bottom=0.20)

#  Set initial values
initial_time = school_stack['decision_at'].min()
time_delim_init = school_stack[school_stack['decision_at'] <= initial_time]

splitters, rev_splitters = split_revsplit(time_delim_init)

colors_init = np.array(['k']*time_delim_init.shape[0], dtype=object)
colors_init[splitters] = 'c'
colors_init[rev_splitters] = 'm'

#  Establish scatter plot
scat = ax.scatter(time_delim_init['decision_at'], time_delim_init['lsat'], s=5, c=colors_init, zorder=3)

plt.title("Acceptances for 20" + str(int(cycles[0])-1) + "-20" + cycles[0] +
          " to 20" + str(int(cycles[-1])-1) + "-20" + cycles[-1] + ", " + school_name +
          " (" + str(school_stack.shape[0]) + " samples)")
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
plt.gca().xaxis.set_minor_locator(mdates.WeekdayLocator(interval=1))
plt.gcf().autofmt_xdate()
ax.set_xlim([school_stack['decision_at'].min() - dt.timedelta(days=7),
             school_stack['decision_at'].max() + dt.timedelta(days=7)])

plt.axhline(y=school_percentiles['L75'].values[0], linewidth=1, color='gray', linestyle='--', zorder=0)
plt.axhline(y=school_percentiles['L25'].values[0], linewidth=1, color='gray', linestyle='--', zorder=0)

plt.ylim([140, 181])

#  Define slider
ax_slider = plt.axes([0.15, 0.05, 0.65, 0.03])
slider = Slider(ax_slider, 'Date', mdates.date2num(school_stack['decision_at'].min()),
                mdates.date2num(school_stack['decision_at'].max()),
                valinit=mdates.date2num(initial_time),
                valfmt="%i")


#  Update function, called upon slider movement
def update(val):
    time_delim = school_stack[school_stack['decision_at'] <= mdates.num2date(val).replace(tzinfo=None)]

    splitters, rev_splitters = split_revsplit(time_delim)

    colors_new = np.array(['k']*time_delim.shape[0], dtype=object)
    colors_new[splitters] = 'c'
    colors_new[rev_splitters] = 'm'
    scat.set_facecolors(colors_new)

    time_delim.loc[:, 'decision_at'] = time_delim['decision_at'].map(lambda x: mdates.date2num(x))

    xx = np.vstack((time_delim['decision_at'], time_delim['lsat']))
    scat.set_offsets(xx.T)

#  Call update function on slider value change
slider.on_changed(update)

ax.annotate('Current as of ' + max(df['decision_at']),
            xy=(1, 0), xytext=(0, 0),
            xycoords=('axes fraction', 'figure fraction'),
            textcoords='offset points',
            size=7, color='gray', ha='right', va='bottom')

plt.show()

#####
#  Splitter probabilities
#####

# splitters = np.intersect1d(np.where(time_delim['lsat'] > 175), np.where(time_delim['gpa'] < 3.78))
# rev_splitters = np.intersect1d(np.where(time_delim['lsat'] < 170), np.where(time_delim['gpa'] > 3.95))
#
# school_date_float.loc[:, 'decision_at'] = school_date_float['decision_at'].map(lambda x: mdates.date2num(x))

## For applications in a given two week period, see if being a splitter/revsplitter makes a difference to acceptance,
## based on others timelines from that same period.

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
