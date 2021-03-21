# -*- coding: utf-8 -*-

import numpy as np
import math
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

##  Assess if this cycle decisions were delayed as compared with last cycle:
##      By A/W/R, splitters/rev splitters.
##      Histogram of distributions by above groupings.
##  Number of applicants completing their LSD profiles, by group/school and by time, plot with % of total on y-axis

#  Suppress Pandas SettingWithCopyWarning
pd.options.mode.chained_assignment = None

#  Time-delimit the data (used to delimit 'decision_at' of main dataframe)
t_min = '09/01/2017'
t_max = '08/31/2021'

#  Read-in medians data, downloaded from https://7sage.com/top-law-school-admissions/
filename_percentiles = '/Users/Shared/lsmedians.csv'
dff = pd.read_csv(filename_percentiles, low_memory=False)
dff = dff[:20]  # Limit to top twenty schools

#  Read-in admissions data, downloaded from https://www.lawschooldata.org/download
filename_admitdata = '/Users/Shared/lsdata.csv'
df = pd.read_csv(filename_admitdata, skiprows=1, low_memory=False)

print('\nShape of original file: ' + str(df.shape))

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
df_filtered = df_filtered[(df_filtered['decision_at'] >= t_min) & (df_filtered['decision_at'] <= t_max)]

#  Format 'softs' column as int, removing leading 'T' (NOTE: will not work if nan values are not filtered out of 'softs'
# df_filtered.loc[:, 'softs'] = df_filtered['softs'].map(lambda x: int(x[1]))

#  Filter by status: accepted, rejected, waitlisted, etc.
# df_filtered = df_filtered[df_filtered['result'] == 'Accepted']

print('Shape of trimmed and filtered file: ' + str(df_filtered.shape))

#####

#  Get admissions medians for selected school
if ('top' not in school_name) & (school_name != 'NYU'):
    school_percentiles = dff[dff['School'].str.contains(school_name[1:])]
elif school_name == 'NYU':
    school_percentiles = dff[dff['School'] == 'New York University']
elif 'top' in school_name:
    if 'eleven' in school_name:
        cutoff = 11
    else:
        cutoff = 5

    temp_dff = dff[:cutoff]
    L75 = temp_dff.groupby('School')['L75'].mean().mean()
    L25 = temp_dff.groupby('School')['L25'].mean().mean()
    G75 = temp_dff.groupby('School')['G75'].mean().mean()
    G25 = temp_dff.groupby('School')['G25'].mean().mean()

    temp = {'L75': [L75], 'L25': [L25], 'G75': [G75], 'G25': [G25]}
    school_percentiles = pd.DataFrame(temp)

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
northwestern = df_filtered[df_filtered['school_name'] == 'Northwestern University']

top_eleven_list = ['Yale University', 'Harvard University', 'Stanford University', 'University of Chicago',
                   'Columbia University', 'New York University', 'University of Pennsylvania', 'University of Virginia',
                   'University of Michigan', 'University of California—Berkeley', 'Northwestern University']
top_eleven = df_filtered[df_filtered['school_name'].str.contains('|'.join(top_eleven_list))]

top_five_list = ['Yale University', 'Harvard University', 'Stanford University', 'University of Chicago', 'Columbia University']
top_five = df_filtered[df_filtered['school_name'].str.contains('|'.join(top_five_list))]

school = None
exec('school = ' + school_name.lower())

if 'top' in school_name:
    school_name = "Top " + school_name[4].upper() + school_name[5:]

print('\nNumber of samples for chosen school: %i' % (school.shape[0]))

#  Length of wait, from sent_at to decision_at, in days
duration = ((school['decision_at'] - school['sent_at'])/np.timedelta64(1, 's'))/(60*60*24)

print('Average wait across all data: %0.f' % duration.mean())


def split_revsplit(df_input):
    splits = np.intersect1d(np.where(df_input['lsat'] > school_percentiles['L75'].values[0]),
                            np.where(df_input['gpa'] < school_percentiles['G25'].values[0]))
    rev_splits = np.intersect1d(np.where(df_input['lsat'] < school_percentiles['L25'].values[0]),
                                np.where(df_input['gpa'] > school_percentiles['G75'].values[0]))

    return splits, rev_splits


#####
#  Divide data into cycle-specific dataframes
#####

school_stack = school.copy()

#  Break data down by cycles
snt_tstart = '09/01'  # 0000
snt_tend = '04/15'  # 0001 (default: 04/15)
dec_tstart = '08/31'  # 0000
dec_tend = '09/01'  # 0001 (default: 09/01)

cycle18 = school_stack[(school_stack['sent_at'] >= snt_tstart + '/2017') & (school_stack['sent_at'] <= snt_tend + '/2018') &
                       (school_stack['decision_at'] <= dec_tend + '/2018') & (school_stack['decision_at'] >= dec_tstart + '/2017')]
cycle19 = school_stack[(school_stack['sent_at'] >= snt_tstart + '/2018') & (school_stack['sent_at'] <= snt_tend + '/2019') &
                       (school_stack['decision_at'] <= dec_tend + '/2019') & (school_stack['decision_at'] >= dec_tstart + '/2018')]
cycle20 = school_stack[(school_stack['sent_at'] >= snt_tstart + '/2019') & (school_stack['sent_at'] <= snt_tend + '/2020') &
                       (school_stack['decision_at'] <= dec_tend + '/2020') & (school_stack['decision_at'] >= dec_tstart + '/2019')]
cycle21 = school_stack[(school_stack['sent_at'] >= snt_tstart + '/2020') & (school_stack['sent_at'] <= snt_tend + '/2021') &
                       (school_stack['decision_at'] <= dec_tend + '/2021') & (school_stack['decision_at'] >= dec_tstart + '/2020')]

#  Account for 2020 being a leap year
cycle20.loc[cycle20['sent_at'] == '02/29/2020', 'sent_at'] = dt.datetime(2020, 02, 28)
cycle20.loc[cycle20['decision_at'] == '02/29/2020', 'decision_at'] = dt.datetime(2020, 02, 28)

#  Standardize cycle years, sent_at
cycle19.loc[:, 'sent_at'] = cycle19['sent_at'].map(lambda x: dt.datetime(x.year-1, x.month, x.day))
cycle20.loc[:, 'sent_at'] = cycle20['sent_at'].map(lambda x: dt.datetime(x.year-2, x.month, x.day))
cycle21.loc[:, 'sent_at'] = cycle21['sent_at'].map(lambda x: dt.datetime(x.year-3, x.month, x.day))

#  Standardize cycle years, decision_at
cycle19.loc[:, 'decision_at'] = cycle19['decision_at'].map(lambda x: dt.datetime(x.year-1, x.month, x.day))
cycle20.loc[:, 'decision_at'] = cycle20['decision_at'].map(lambda x: dt.datetime(x.year-2, x.month, x.day))
cycle21.loc[:, 'decision_at'] = cycle21['decision_at'].map(lambda x: dt.datetime(x.year-3, x.month, x.day))

school_stack = pd.concat([cycle18, cycle19, cycle20, cycle21])


#####
#  EXPERIMENTAL
#####
# filter_rows = ['sent_at', 'lsat', 'gpa']
# temp = df_dcol.dropna(subset=filter_rows)
#
# #  Convert sent_at and decision_at to datetime
# temp.loc[:, 'sent_at'] = pd.to_datetime(df_filtered['sent_at'])
# temp.loc[:, 'decision_at'] = pd.to_datetime(df_filtered['decision_at'])
#
# #  Filter such that data falls within t_min and t_max range
# temp = temp[(temp['decision_at'] >= t_min) & (temp['decision_at'] <= t_max)]
#
# snt_tstart = '09/01'  # 0000
# snt_tend = '04/15'  # 0001 (default: 04/15)
# dec_tstart = '08/31'  # 0000
# dec_tend = '09/01'  # 0001 (default: 09/01)
# #  Break data down by cycles
# cycle18a = temp[(temp['sent_at'] >= snt_tstart + '/2017') & (temp['sent_at'] <= snt_tend + '/2018')]
# cycle19a = temp[(temp['sent_at'] >= snt_tstart + '/2018') & (temp['sent_at'] <= snt_tend + '/2019')]
# cycle20a = temp[(temp['sent_at'] >= snt_tstart + '/2019') & (temp['sent_at'] <= snt_tend + '/2020')]
# cycle21a = temp[(temp['sent_at'] >= snt_tstart + '/2020') & (temp['sent_at'] <= snt_tend + '/2021')]
#
# print("\ncycle18: " + str(cycle18a.shape[0]))
# print("cycle19: " + str(cycle19a.shape[0]))
# print("cycle20: " + str(cycle20a.shape[0]))
# print("cycle21: " + str(cycle21a.shape[0]))


#####
#  Study and plot sent_at vs. decision_at, stacking cycles
#####

markers = ['v', '^', '<', 'o']
cycles = ['18', '19', '20', '21']

#  Plot cycle by cycle
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

    plt.scatter(accepted['sent_at'], accepted['decision_at'],
                color='green', edgecolors=edge_colors_a, marker=mark, label='A 20' + cycles[i], s=17, zorder=3)
    plt.scatter(rejected['sent_at'], rejected['decision_at'],
                color='red', edgecolors=edge_colors_r, marker=mark, label='R 20' + cycles[i], s=17, zorder=3)
    plt.scatter(waitlisted['sent_at'], waitlisted['decision_at'],
                color='orange', edgecolors=edge_colors_w, marker=mark, label='W 20' + cycles[i], s=17, zorder=3)

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
plt.gca().xaxis.set_minor_locator(mdates.WeekdayLocator(interval=1))

plt.gca().yaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
plt.gca().yaxis.set_major_locator(mdates.MonthLocator())
plt.gca().yaxis.set_minor_locator(mdates.WeekdayLocator(interval=1))

plt.xlabel('Date Sent')
plt.ylabel('Date Decision Received')
plt.title('Admissions Data for 20' + str(int(cycles[0])-1) + '-20' + cycles[0] +
          ' to 20' + str(int(cycles[-1])-1) + '-20' + cycles[-1] + ', ' + school_name +
          ' (' + str(school_stack.shape[0]) + ' samples)')

#  Format legend
custom_markers = [Line2D([0], [0], marker=markers[0], markerfacecolor='grey', markeredgecolor='grey', markersize=7, ls=''),
                  Line2D([0], [0], marker=markers[1], markerfacecolor='grey', markeredgecolor='grey', markersize=7, ls=''),
                  Line2D([0], [0], marker=markers[2], markerfacecolor='grey', markeredgecolor='grey', markersize=7, ls=''),
                  Line2D([0], [0], marker=markers[3], markerfacecolor='grey', markeredgecolor='grey', markersize=7, ls=''),
                  Line2D([0], [0], marker='o', markerfacecolor='white', markeredgecolor='b', markersize=7, ls=''),
                  Line2D([0], [0], marker='o', markerfacecolor='white', markeredgecolor='k', markersize=7, ls='')]
custom_labels = [str(int(cycles[0])-1) + '/' + cycles[0] + ' (n=' + str(cycle18.shape[0]) + ')',
                 str(int(cycles[1])-1) + '/' + cycles[1] + ' (n=' + str(cycle19.shape[0]) + ')',
                 str(int(cycles[2])-1) + '/' + cycles[2] + ' (n=' + str(cycle20.shape[0]) + ')',
                 str(int(cycles[3])-1) + '/' + cycles[3] + ' (n=' + str(cycle21.shape[0]) + ')']
plt.legend(custom_markers, custom_labels + ['Splitters', 'Reverse Splitters'])

#  Draw horizontal line at date of latest decision in file
plt.axhline(y=dt.datetime.strptime(max(df['decision_at']), '%Y-%m-%d')-dt.timedelta(days=365*3),
            linewidth=0.75, color='steelblue', linestyle='--', zorder=2)

plt.grid(zorder=0)

plt.annotate('Current as of ' + max(df['decision_at']) + ' (-----)',
             xy=(1, 0), xytext=(0, 5),
             xycoords=('axes fraction', 'figure fraction'),
             textcoords='offset points',
             size=7, color='gray', ha='right', va='bottom')

# plt.xlim(dt.datetime(2017, 8, 15), dt.datetime(2018, 4, 15))
# plt.ylim(dt.datetime(2017, 8, 15), dt.datetime(2018, 6, 15))

plt.show()

#####
#  Study duration of wait by cycle (print output + histogram)
#####

print('\n' + '{0:<8} {1:<11} {2:<11} {3:<11}'.format('Cycle', 'Avg. Wait', 'Std. Dev.', 'n='))
print('-'*36)

#  Calculate duration statistics by cycle
durations18 = durations19 = durations20 = durations21 = None
durations18a = durations19a = durations20a = durations21a = None
durations18r = durations19r = durations20r = durations21r = None
durations18w = durations19w = durations20w = durations21w = None
cycle_wait_means = []  # To store means for each cycle
for c in cycles:
    c_temp = None
    exec('c_temp = cycle' + c)
    duration_temp = ((c_temp['decision_at'] - c_temp['sent_at'])/np.timedelta64(1, 's'))/(60*60*24)
    exec('durations' + c + ' = duration_temp')

    print('{0:<8} {1:<11} {2:<11} {3:<11}'.format(str(int(c)-1) + '/' + c,
                                                  str(int(duration_temp.mean())),
                                                  str(int(duration_temp.std())),
                                                  c_temp.shape[0]))

    #  Store wait times by result, for processing below
    accepted = c_temp[c_temp['result'] == 'Accepted']
    rejected = c_temp[c_temp['result'] == 'Rejected']
    waitlisted = c_temp[c_temp['result'].str.contains('Wait')]

    exec('durations' + c + 'a' +
         " = ((accepted['decision_at'] - accepted['sent_at']) / np.timedelta64(1, 's')) / (60 * 60 * 24)")
    exec ('durations' + c + 'r' +
          " = ((rejected['decision_at'] - rejected['sent_at']) / np.timedelta64(1, 's')) / (60 * 60 * 24)")
    exec ('durations' + c + 'w' +
          " = ((waitlisted['decision_at'] - waitlisted['sent_at']) / np.timedelta64(1, 's')) / (60 * 60 * 24)")

    cycle_wait_means.append(duration_temp.mean())

#  Create print output to assess how wait times have changed by cycle and by decision
print('\n' + '{0:<8} {1:<8} {2:<11} {3:<11} {4:<11}'.format('Cycle', 'Dec.', 'Avg. Wait', 'Std. Dev.', 'n='))
print('-'*45)
for s in ['a', 'r', 'w']:
    for c in cycles:
        t1 = str(int(c)-1) + '/' + c  # Cycle
        t2 = s  # Result/decision
        t3 = ''  # Avg. Wait
        try:
            exec('t3 = str(int(durations' + c + s + '.mean()))')
        except ValueError:
            t3 = '--'
        t4 = ''  # Std. Dev.
        try:
            exec ('t4 = str(int(durations' + c + s + '.std()))')
        except ValueError:
            t4 = '--'
        t5 = ''  # n=
        exec ('t5 = str(durations' + c + s + '.shape[0])')
        print('{0:<8} {1:<8} {2:<11} {3:<11} {4:<11}'.format(t1, t2, t3, t4, t5))

#  Plot histogram
day_lim = 250
num_bins = int(math.ceil(day_lim/7)) + 1
n, bins, patches = plt.hist([durations18[durations18 < day_lim], durations19[durations19 < day_lim],
                             durations20[durations20 < day_lim], durations21[durations21 < day_lim]],
                            bins=num_bins, stacked=True, density=True, label=custom_labels)

plt.title('Number of Days from Sent to Decision, ' + school_name + ' (' + str(school_stack.shape[0]) + ' samples)')
plt.xlabel('Number of days')
plt.ylabel('Frequency')

#  Format legend
handles, labels = plt.gca().get_legend_handles_labels()
temp = ['; mean=' + str(int(i)) + ')' for i in cycle_wait_means]
labels = [a[:-1] + b for a, b in zip(labels, temp)]
plt.legend(handles[::-1], labels[::-1]) # Reverse legend to match order on histogram

#  Demarcate means by cycle
for cwm in cycle_wait_means:
    plt.axvline(x=cwm, linewidth=0.75, color='k', linestyle='--')

plt.annotate('Current as of ' + max(df['decision_at']) + '.',
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
print('\nMean Absolute Error: %.2f' % metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error: %.2f' % metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error: %.2f' % np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

#  Predict given input stats: sent_at, lsat, gpa
to_predict = sc.transform(np.array([mdates.date2num(dt.datetime(2019, 11, 13)), 171, 3.89]).reshape(1, -1))
print('\nPrediction: %.2f' % regressor.predict(to_predict)[0])

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

plt.title('Acceptances for 20' + str(int(cycles[0])-1) + '-20' + cycles[0] +
          ' to 20' + str(int(cycles[-1])-1) + '-20' + cycles[-1] + ', ' + school_name +
          ' (' + str(school_stack.shape[0]) + ' samples)')
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
plt.gca().xaxis.set_minor_locator(mdates.WeekdayLocator(interval=1))
plt.gcf().autofmt_xdate()
ax.set_xlim([school_stack['decision_at'].min() - dt.timedelta(days=7),
             school_stack['decision_at'].max() + dt.timedelta(days=7)])

#  Draw horizontal lines at L75 and L25 and vertical line at date of latest decision in file
plt.axhline(y=school_percentiles['L75'].values[0], linewidth=1, color='gray', linestyle='--', zorder=0)
plt.axhline(y=school_percentiles['L25'].values[0], linewidth=1, color='gray', linestyle='--', zorder=0)
plt.axvline(x=dt.datetime.strptime(max(df['decision_at']), '%Y-%m-%d')-dt.timedelta(days=365*3),
            linewidth=0.75, color='steelblue', linestyle='--', zorder=2)

plt.ylim([140, 181])
plt.xlim(school_stack['decision_at'].min() - dt.timedelta(weeks=3),
         school_stack['decision_at'].max() + dt.timedelta(weeks=3))

#  Define slider
ax_slider = plt.axes([0.15, 0.05, 0.65, 0.03])
slider = Slider(ax_slider, 'Date', mdates.date2num(school_stack['decision_at'].min()),
                mdates.date2num(school_stack['decision_at'].max()),
                valinit=mdates.date2num(initial_time),
                valfmt='%i')


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

ax.annotate('Current as of ' + max(df['decision_at']) + ' (-----)',
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
