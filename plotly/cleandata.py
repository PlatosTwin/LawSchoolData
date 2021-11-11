# -*- coding: utf-8 -*-
import datetime as dt
from os import getcwd
from pathlib import Path

import numpy as np
import csv
import pandas as pd
from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()

#  Suppress Pandas SettingWithCopyWarning
pd.options.mode.chained_assignment = None

#  Read-in medians data, compiled from ABA: https://www.americanbar.org/groups/legal_education/resources/statistics/
fname_percentiles = 'lsmedians.csv'
dfmeds = pd.read_csv(fname_percentiles)

#  Read-in admissions data, downloaded from https://www.lawschooldata.org/download
fname_admit = '/Users/Shared/lsdata.csv'
df = pd.read_csv(fname_admit, skiprows=1, low_memory=False)

print('\nPreparing to clean ' + fname_admit + '...')
print('\nShape of original file: ' + str(df.shape))

#  Drop unnecessary/uninteresting columns
drop_cols = ['ur_at', 'ur2_at', 'is_military', 'is_character_and_fitness_issues', 'non_trad', 'years_out', 'softs',
             'simple_status', 'scholarship', 'attendance', 'is_in_state', 'is_fee_waived', 'is_conditional_scholarship',
             'is_international', 'international_gpa', 'is_lsn_import']
df_dcol = df.drop(drop_cols, axis=1)

#  Rearrange columns to avoid errors, from future adjustments, in timeline.py
cols_to_order = ['school_name', 'sent_at', 'decision_at', 'lsat', 'gpa', 'cycle_id']
new_columns = cols_to_order + (df_dcol.columns.drop(cols_to_order).tolist())
df_dcol = df_dcol[new_columns]

#  Remove rows that are missing crucial values
filter_rows = ['lsat', 'gpa', 'result']
df_filtered = df_dcol.dropna(subset=filter_rows)

#  Convert non-null sent_at and decision_at entries to datetime
df_filtered.loc[:, 'sent_at'] = pd.to_datetime(df_filtered['sent_at'])
df_filtered.loc[:, 'decision_at'] = pd.to_datetime(df_filtered['decision_at'])

T11 = ['Yale University', 'Harvard University', 'Stanford University', 'University of Chicago',
       'Columbia University', 'New York University', 'University of Pennsylvania', 'University of Virginia',
       'University of Michigan', 'University of California—Berkeley', 'Northwestern University']

df11 = df_filtered[df_filtered['school_name'].str.contains('|'.join(T11))]

# Remove rows with result == 'Intend to Apply' and 'Intend to Apply, Withdrawn'
indexes = df11[df11['result'].str.contains('|'.join(['Intend to Apply', 'Intend to Apply, Withdrawn']))].index
df11.drop(indexes, inplace=True)

print('Shape of filtered and delimited file: ' + str(df_filtered.shape))


def label_cycle(row):
    """
    Assign admissions cycle, determined either by the sent_at and decision_at dates in
    conjunction, by only one if the other is null, or by the cycle_id in lsdata.csv if both are null.
    :param row:
    :return: cycle label, int:
    """

    #  TODO: translate to np.select: https://towardsdatascience.com/efficient-implementation-of-conditional-logic-on-pandas-dataframes-4afa61eb7fce
    if pd.isnull(row['sent_at']) & pd.isnull(row['decision_at']):
        return int(row['cycle_id']) + 3

    if pd.isnull(row['sent_at']):
        if (row['decision_at'] <= dec_tend22) & \
                (row['decision_at'] >= dec_tstart22):
            return 22
        if (row['decision_at'] <= dec_tend21) & \
                (row['decision_at'] >= dec_tstart21):
            return 21
        if (row['decision_at'] <= dec_tend20) & \
                (row['decision_at'] >= dec_tstart20):
            return 20
        if (row['decision_at'] <= dec_tend19) & \
                (row['decision_at'] >= dec_tstart19):
            return 19
        if (row['decision_at'] <= dec_tend18) & \
                (row['decision_at'] >= dec_tstart18):
            return 18

    if pd.isnull(row['decision_at']):
        if (row['sent_at'] >= snt_tstart22) & \
                (row['sent_at'] <= snt_tend22):
            return 22
        if (row['sent_at'] >= snt_tstart21) & \
                (row['sent_at'] <= snt_tend21):
            return 21
        if (row['sent_at'] >= snt_tstart20) & \
                (row['sent_at'] <= snt_tend20):
            return 20
        if (row['sent_at'] >= snt_tstart19) & \
                (row['sent_at'] <= snt_tend19):
            return 19
        if (row['sent_at'] >= snt_tstart18) & \
                (row['sent_at'] <= snt_tend18):
            return 18

    if (row['sent_at'] >= snt_tstart22) & \
            (row['sent_at'] <= snt_tend22) & \
            (row['decision_at'] <= dec_tend22) & \
            (row['decision_at'] >= dec_tstart22):
        return 22
    if (row['sent_at'] >= snt_tstart21) & \
            (row['sent_at'] <= snt_tend21) & \
            (row['decision_at'] <= dec_tend21) & \
            (row['decision_at'] >= dec_tstart21):
        return 21
    if (row['sent_at'] >= snt_tstart20) & \
            (row['sent_at'] <= snt_tend20) & \
            (row['decision_at'] <= dec_tend20) & \
            (row['decision_at'] >= dec_tstart20):
        return 20
    if (row['sent_at'] >= snt_tstart19) & \
            (row['sent_at'] <= snt_tend19) & \
            (row['decision_at'] <= dec_tend19) & \
            (row['decision_at'] >= dec_tstart19):
        return 19
    if (row['sent_at'] >= snt_tstart18) & \
            (row['sent_at'] <= snt_tend18) & \
            (row['decision_at'] <= dec_tend18) & \
            (row['decision_at'] >= dec_tstart18):
        return 18

    return 0


def simplify_result(row):
    """
    Assign simplified decision string as a function of lawschooldata.org decision string, collapsing into one all
    waitlisted decisions and all hold decisions.
    :param row:
    :return one of 'A', 'R', 'WL', 'P', 'WD', 'H', '?':
    """
    if row['result'] == 'Rejected':
        return 'R'

    if any(s in row['result'] for s in ('Wait', 'WL')):
        return 'WL'

    if row['result'] == 'Accepted':
        return 'A'

    if row['result'] == 'Pending':
        return 'P'

    if row['result'] == 'Withdrawn':
        return 'WD'

    if 'Hold' in row['result']:
        return 'H'

    return '?'


def label_color(row):
    """
    Assign marker fill color as a function of admissions decision: green for admitted, red for rejected, orange for
    waitlisted, black for held, pink for all others.
    :param row:
    :return color label, string:
    """
    if row['decision'] == 'A':
        return 'green'

    if row['decision'] == 'R':
        return 'red'

    if row['decision'] == 'WL':
        return 'orange'

    if row['decision'] == 'H':
        return 'black'

    return 'pink'


def label_marker(row):
    """
    Assign marker style as a function of cycle: circle for 2021, triangle-sw for 2020, triangle-se for 2019,
    triangle-ne for 2018.
    :param row:
    :return marker label, string:
    """
    if row['cycle'] == 22:
        return 'circle'

    if row['cycle'] == 21:
        return 'triangle-nw'

    if row['cycle'] == 20:
        return 'triangle-sw'

    if row['cycle'] == 19:
        return 'triangle-se'

    if row['cycle'] == 18:
        return 'triangle-ne'


def label_splitter(row):
    """
    Assign marker border color as a function of decision: green for admitted, red for rejected, orange for held,
    blue for splitters, black for reverse splitters.
    :param row:
    :return border color label, string:
    """
    dfmeds_temp = dfmeds[(dfmeds['cycle'] == row['cycle']) & (dfmeds['school_name'] == row['school_name'])]

    #  Splitters = blue
    if (row['lsat'] > dfmeds_temp['L75'].values[0]) & \
            (row['gpa'] < dfmeds_temp['G25'].values[0]):
        return 'blue'

    #  Reverse splitters = black
    if (row['lsat'] < dfmeds_temp['L25'].values[0]) & \
            (row['gpa'] > dfmeds_temp['G75'].values[0]):
        return 'black'

    if row['decision'] == 'A':
        return 'green'

    if row['decision'] == 'R':
        return 'red'

    if row['decision'] == 'WL':
        return 'orange'

    return 'pink'


def label_wait(row):
    """
    Determine the number of days between sent_at and decision_at.
    :param row:
    :return wait time in days as int, or float NaN:
    """
    return ((row['decision_at'] - row['sent_at']) / np.timedelta64(1, 's')) / (60 * 60 * 24)


#  Prepare to break data down by cycles
snt_tstart = '09/01'  # 0000
snt_tend = '04/15'  # 0001 (default: 04/15)
dec_tstart = '08/31'  # 0000
dec_tend = '09/01'  # 0001 (default: 09/01)

snt_tstart18 = dt.datetime.strptime(snt_tstart + '/2017', '%m/%d/%Y')
snt_tstart19 = dt.datetime.strptime(snt_tstart + '/2018', '%m/%d/%Y')
snt_tstart20 = dt.datetime.strptime(snt_tstart + '/2019', '%m/%d/%Y')
snt_tstart21 = dt.datetime.strptime(snt_tstart + '/2020', '%m/%d/%Y')
snt_tstart22 = dt.datetime.strptime(snt_tstart + '/2021', '%m/%d/%Y')

snt_tend18 = dt.datetime.strptime(snt_tend + '/2018', '%m/%d/%Y')
snt_tend19 = dt.datetime.strptime(snt_tend + '/2019', '%m/%d/%Y')
snt_tend20 = dt.datetime.strptime(snt_tend + '/2020', '%m/%d/%Y')
snt_tend21 = dt.datetime.strptime(snt_tend + '/2021', '%m/%d/%Y')
snt_tend22 = dt.datetime.strptime(snt_tend + '/2022', '%m/%d/%Y')

dec_tstart18 = dt.datetime.strptime(dec_tstart + '/2017', '%m/%d/%Y')
dec_tstart19 = dt.datetime.strptime(dec_tstart + '/2018', '%m/%d/%Y')
dec_tstart20 = dt.datetime.strptime(dec_tstart + '/2019', '%m/%d/%Y')
dec_tstart21 = dt.datetime.strptime(dec_tstart + '/2020', '%m/%d/%Y')
dec_tstart22 = dt.datetime.strptime(dec_tstart + '/2021', '%m/%d/%Y')

dec_tend18 = dt.datetime.strptime(dec_tend + '/2018', '%m/%d/%Y')
dec_tend19 = dt.datetime.strptime(dec_tend + '/2019', '%m/%d/%Y')
dec_tend20 = dt.datetime.strptime(dec_tend + '/2020', '%m/%d/%Y')
dec_tend21 = dt.datetime.strptime(dec_tend + '/2021', '%m/%d/%Y')
dec_tend22 = dt.datetime.strptime(dec_tend + '/2022', '%m/%d/%Y')

#  Set 2021/2022 stats to corresponding stats from prior year, if current year's are empty
for m in range(2):
    for school in T11:
        temp = dfmeds[dfmeds['school_name'] == school].replace(np.NaN, '')
        dfmeds[dfmeds['school_name'] == school] = \
            temp.apply(lambda x:
                       [i if i != ''
                        else temp[temp['cycle'] == x[9] - 1].iloc[:, k].values[0] for k, i in enumerate(x)],
                       axis=1,
                       result_type='expand')

#  TODO: streamline and make more efficient
#  Label cycles: 18, 19, 20, 21, 22
df11['cycle_id'] = df11.apply(lambda row: label_cycle(row), axis=1)
df11.rename(columns={'cycle_id': 'cycle'}, inplace=True)
df11 = df11[(df11['cycle'] > 17) & (df11['cycle'] < 23)]
print('\n1/6: Labelled cycles...')

#  Simplify results
df11['decision'] = df11.apply(lambda row: simplify_result(row), axis=1)
print('2/6: Simplified results...')

#  Add color indicator by result, for plotting
df11['color'] = df11.apply(lambda row: label_color(row), axis=1)
print('3/6: Added colors...')

#  Create markers based on cycle, for plotting
df11['marker'] = df11.apply(lambda row: label_marker(row), axis=1)
print('4/6: Set markers...')

#  Mark splitters/reverse splitters, for plotting and analysis
df11['splitter'] = df11.apply(lambda row: label_splitter(row), axis=1)
print('5/6: Marked splitters...')

#  Calculate wait time, for plotting and analysis
df11['wait'] = df11.apply(lambda row: label_wait(row), axis=1)
print('6/6: Calculated wait times...')

#  Account for 2020 being a leap year, to aid date normalization
df11.loc[df11['sent_at'] == '02/29/2020', 'sent_at'] = dt.datetime(2020, 2, 28)
df11.loc[df11['decision_at'] == '02/29/2020', 'decision_at'] = dt.datetime(2020, 2, 28)

fname_save = 'lsdata_clean.csv'
df11.to_csv(fname_save, index=False)

print('\nCompleted and saved reference file to: ' + fname_save + '.')

#  Save footer
with open(fname_admit, newline='') as f:
    reader = csv.reader(f)
    for row in reader:
        updated = row[0]
        break

current_of = dt.datetime.strptime(updated[updated.index(':') + 2:updated.index(':') + 12], '%Y-%m-%d')
current = 'Current as of ' + str(current_of.month) + '/' + str(current_of.day) + '/%i' % dt.datetime.now().year
update_time = 'and updated approximately weekly.'
footer = current + ' ' + update_time

cwd = Path(getcwd())
fname_footer = str(cwd.parent.absolute()) + '/docs/_includes/footer.html'
with open(fname_footer, 'w') as f:
    f.write(footer)

print('Saved footer file.')
