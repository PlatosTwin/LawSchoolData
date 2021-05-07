# -*- coding: utf-8 -*-
import math
from os import getcwd
from pathlib import Path
import datetime as dt
import numpy as np
import pandas as pd
from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()

#  Suppress Pandas SettingWithCopyWarning
pd.options.mode.chained_assignment = None

#  Time-delimit the data (used to delimit 'decision_at' of main dataframe)
t_min = dt.datetime.strptime('2017-09-12', '%Y-%m-%d')
t_max = dt.datetime.strptime('2021-08-31', '%Y-%m-%d')

#  Read-in medians data, downloaded from https://7sage.com/top-law-school-admissions/
fname_percentiles = '/Users/Shared/lsmedians.csv'
dfmeds = pd.read_csv(fname_percentiles, low_memory=False)
dfmeds = dfmeds[:20]  # Limit to top twenty schools

#  Read-in admissions data, downloaded from https://www.lawschooldata.org/download
fname_admit = '/Users/Shared/lsdata.csv'
df = pd.read_csv(fname_admit, skiprows=1, low_memory=False)

print('\nPreparing to clean ' + fname_admit + '...')
print('\nShape of original file: ' + str(df.shape))

#  Drop unnecessary/uninteresting columns
drop_cols = ['simple_status', 'scholarship', 'attendance', 'is_in_state', 'is_fee_waived', 'is_conditional_scholarship',
             'is_international', 'international_gpa', 'is_lsn_import']
df_dcol = df.drop(drop_cols, axis=1)

#  Remove rows that are missing crucial values
filter_rows = ['lsat', 'gpa', 'result']
df_filtered = df_dcol.dropna(subset=filter_rows)

df_filtered.loc[:, 'sent_at'] = pd.to_datetime(df_filtered['sent_at'])
df_filtered.loc[:, 'decision_at'] = pd.to_datetime(df_filtered['decision_at'])

#  Filter such that data falls within t_min and t_max range
# df_filtered = df_filtered[(df_filtered['decision_at'] >= t_min) & (df_filtered['decision_at'] <= t_max) &
#                           (df_filtered['sent_at'] <= t_max) & (df_filtered['sent_at'] >= t_min)]

print('Shape of trimmed and filtered file: ' + str(df_filtered.shape))

top_eleven_list = ['Yale University', 'Harvard University', 'Stanford University', 'University of Chicago',
                   'Columbia University', 'New York University', 'University of Pennsylvania', 'University of Virginia',
                   'University of Michigan', 'University of California—Berkeley', 'Northwestern University']

df11 = df_filtered[df_filtered['school_name'].str.contains('|'.join(top_eleven_list))]


def label_cycle(row):
    #  Break data down by cycles
    snt_tstart = '09/01'  # 0000
    snt_tend = '04/15'  # 0001 (default: 04/15)
    dec_tstart = '08/31'  # 0000
    dec_tend = '09/01'  # 0001 (default: 09/01)

    if pd.isnull(row['sent_at']) & pd.isnull(row['decision_at']):
        return int(row['cycle_id']) + 3

    if pd.isnull(row['sent_at']):
        if (row['decision_at'] <= dt.datetime.strptime(dec_tend + '/2018', '%m/%d/%Y')) & \
                (row['decision_at'] >= dt.datetime.strptime(dec_tstart + '/2017', '%m/%d/%Y')):
            return 18
        if (row['decision_at'] <= dt.datetime.strptime(dec_tend + '/2019', '%m/%d/%Y')) & \
                (row['decision_at'] >= dt.datetime.strptime(dec_tstart + '/2018', '%m/%d/%Y')):
            return 19
        if (row['decision_at'] <= dt.datetime.strptime(dec_tend + '/2020', '%m/%d/%Y')) & \
                (row['decision_at'] >= dt.datetime.strptime(dec_tstart + '/2019', '%m/%d/%Y')):
            return 20
        if (row['decision_at'] <= dt.datetime.strptime(dec_tend + '/2021', '%m/%d/%Y')) & \
                (row['decision_at'] >= dt.datetime.strptime(dec_tstart + '/2020', '%m/%d/%Y')):
            return 21

    if pd.isnull(row['decision_at']):
        if (row['sent_at'] >= dt.datetime.strptime(snt_tstart + '/2017', '%m/%d/%Y')) & \
                (row['sent_at'] <= dt.datetime.strptime(snt_tend + '/2018', '%m/%d/%Y')):
            return 18
        if (row['sent_at'] >= dt.datetime.strptime(snt_tstart + '/2018', '%m/%d/%Y')) & \
                (row['sent_at'] <= dt.datetime.strptime(snt_tend + '/2019', '%m/%d/%Y')):
            return 19
        if (row['sent_at'] >= dt.datetime.strptime(snt_tstart + '/2019', '%m/%d/%Y')) & \
                (row['sent_at'] <= dt.datetime.strptime(snt_tend + '/2020', '%m/%d/%Y')):
            return 20
        if (row['sent_at'] >= dt.datetime.strptime(snt_tstart + '/2020', '%m/%d/%Y')) & \
                (row['sent_at'] <= dt.datetime.strptime(snt_tend + '/2021', '%m/%d/%Y')):
            return 21

    if (row['sent_at'] >= dt.datetime.strptime(snt_tstart + '/2017', '%m/%d/%Y')) & \
            (row['sent_at'] <= dt.datetime.strptime(snt_tend + '/2018', '%m/%d/%Y')) & \
            (row['decision_at'] <= dt.datetime.strptime(dec_tend + '/2018', '%m/%d/%Y')) & \
            (row['decision_at'] >= dt.datetime.strptime(dec_tstart + '/2017', '%m/%d/%Y')):
        return 18
    if (row['sent_at'] >= dt.datetime.strptime(snt_tstart + '/2018', '%m/%d/%Y')) & \
            (row['sent_at'] <= dt.datetime.strptime(snt_tend + '/2019', '%m/%d/%Y')) & \
            (row['decision_at'] <= dt.datetime.strptime(dec_tend + '/2019', '%m/%d/%Y')) & \
            (row['decision_at'] >= dt.datetime.strptime(dec_tstart + '/2018', '%m/%d/%Y')):
        return 19
    if (row['sent_at'] >= dt.datetime.strptime(snt_tstart + '/2019', '%m/%d/%Y')) & \
            (row['sent_at'] <= dt.datetime.strptime(snt_tend + '/2020', '%m/%d/%Y')) & \
            (row['decision_at'] <= dt.datetime.strptime(dec_tend + '/2020', '%m/%d/%Y')) & \
            (row['decision_at'] >= dt.datetime.strptime(dec_tstart + '/2019', '%m/%d/%Y')):
        return 20
    if (row['sent_at'] >= dt.datetime.strptime(snt_tstart + '/2020', '%m/%d/%Y')) & \
            (row['sent_at'] <= dt.datetime.strptime(snt_tend + '/2021', '%m/%d/%Y')) & \
            (row['decision_at'] <= dt.datetime.strptime(dec_tend + '/2021', '%m/%d/%Y')) & \
            (row['decision_at'] >= dt.datetime.strptime(dec_tstart + '/2020', '%m/%d/%Y')):
        return 21

    return 0


def simplify_result(row):
    if row['result'] == 'Accepted':
        return 'A'

    if row['result'] == 'Rejected':
        return 'R'

    if any(s in row['result'] for s in ('Wait', 'WL')):
        return 'WL'

    if row['result'] == 'Pending':
        return 'P'

    if row['result'] == 'Withdrawn':
        return 'WD'

    if 'Hold' in row['result']:
        return 'H'

    return '?'


def label_color(row):
    if row['decision'] == 'A':
        return 'green'

    if row['decision'] == 'R':
        return 'red'

    if row['decision'] == 'WL':
        return 'orange'

    return 'black'


def label_marker(row):
    if row['cycle'] == 18:
        return 'triangle-ne'

    if row['cycle'] == 19:
        return 'triangle-se'

    if row['cycle'] == 20:
        return 'triangle-sw'

    if row['cycle'] == 21:
        return 'circle'


def label_splitter(row):
    #  Splitters = blue
    if (row['lsat'] > dfmeds[dfmeds['School'] == row['school_name']]['L75'].values[0]) & \
            (row['gpa'] < dfmeds[dfmeds['School'] == row['school_name']]['G25'].values[0]):
        return 'blue'

    #  Reverse splitters = black
    if (row['lsat'] < dfmeds[dfmeds['School'] == row['school_name']]['L25'].values[0]) & \
            (row['gpa'] > dfmeds[dfmeds['School'] == row['school_name']]['G75'].values[0]):
        return 'black'

    if row['decision'] == 'A':
        return 'green'

    if row['decision'] == 'WL':
        return 'orange'

    if row['decision'] == 'R':
        return 'red'


def label_wait(row):
    return ((row['decision_at'] - row['sent_at'])/np.timedelta64(1, 's'))/(60*60*24)


#  Label cycles: 18, 19, 20, 21
df11['cycle_id'] = df11.apply(lambda row: label_cycle(row), axis=1)
df11.rename(columns={'cycle_id': 'cycle'}, inplace=True)
df11 = df11[df11['cycle'] > 17]
print('\n1/6: Labelled cycles...')

#  Simplify results
df11['decision'] = df11.apply(lambda row: simplify_result(row), axis=1)
print('2/6: Simplified results...')

#  Add color indicator by result, for plotting
df11['color'] = df11.apply(lambda row: label_color(row), axis=1)
print('3/6: Added colors...')

#  Create markers based on cycle
df11['marker'] = df11.apply(lambda row: label_marker(row), axis=1)
print('4/6: Set markers...')

#  Mark splitters/reverse splitters
df11['splitter'] = df11.apply(lambda row: label_splitter(row), axis=1)
print('5/6: Marked splitters...')

#  Calculate wait time
df11['wait'] = df11.apply(lambda row: label_wait(row), axis=1)
print('6/6: Calculated wait times...')

#  Account for 2020 being a leap year
df11.loc[df11['sent_at'] == '02/29/2020', 'sent_at'] = dt.datetime(2020, 2, 28)
df11.loc[df11['decision_at'] == '02/29/2020', 'decision_at'] = dt.datetime(2020, 2, 28)

fname_save = 'lsdata_clean.csv'
df11.to_csv(fname_save, index=False)

print('\nCompleted and saved reference file to: ' + fname_save + '.')

#  Update footer with date of latest entry in lsdata.csv
current_of = max(df11[df11['cycle'] == 21]['decision_at'])
current = 'Current as of ' + str(current_of.month) + '/' + str(current_of.day) + '/2021.'
reference = 'Admissions data from LawSchoolData.org. Medians data from 7Sage.com.'
footer = current + ' ' + reference

cwd = Path(getcwd())
fname_footer = str(cwd.parent.absolute()) + '/docs/_includes/footer.html'
with open(fname_footer, 'w') as f:
    f.write(footer)

print('Saved footer file.')
