from os import getcwd
from pathlib import Path
import datetime as dt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import numpy as np
import pandas as pd
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

#  Read-in admissions data
fname_admit = 'lsdata_clean.csv'
df11 = pd.read_csv(fname_admit, low_memory=False)

#  Convert sent_at and decision_at to datetime
df11.loc[:, 'sent_at'] = pd.to_datetime(df11['sent_at'])
df11.loc[:, 'decision_at'] = pd.to_datetime(df11['decision_at'])

cycles = [18, 19, 20, 21]

#  Normalize years
for i, cycle in enumerate(cycles[1:]):
    df11.loc[df11['cycle'] == cycle, 'sent_at'] = \
        df11[df11['cycle'] == cycle]['sent_at'].map(lambda t: dt.datetime(t.year - (i + 1), t.month, t.day))

    df11.loc[df11['cycle'] == cycle, 'decision_at'] = \
        df11[df11['cycle'] == cycle]['decision_at'].map(lambda t: dt.datetime(t.year - (i + 1), t.month, t.day))

T11 = ['Yale University', 'Harvard University', 'Stanford University', 'University of Chicago',
       'Columbia University', 'New York University', 'University of Pennsylvania', 'University of Virginia',
       'University of Michigan', 'University of California—Berkeley', 'Northwestern University']

T11_short = ['Yale', 'Harvard', 'Stanford', 'UChicago', 'Columbia', 'NYU', 'UPenn', 'Virginia', 'Michigan',
             'Berkeley', 'Northwestern ']

#  Create figure
fig = make_subplots(rows=2, cols=2, start_cell="top-left")

current_of = max(df11[df11['cycle'] == 21]['decision_at'])

#  Set initial histogram traces
row = 1
col = 1
showlegend = False

for r in ['all', 'A', 'R', 'WL']:
    for c in cycles:
        if r == 'all':
            df_temp = df11[(df11['school_name'] == T11[0]) & (df11['cycle'] == c) & (df11['decision_at'] <= current_of)]
            showlegend = True
        else:
            df_temp = df11[(df11['school_name'] == T11[0]) & (df11['cycle'] == c) &
                           (df11['decision_at'] <= current_of) & (df11['decision'] == r)]
            showlegend = False

        fig.add_trace(go.Histogram(
            x=df_temp['wait'],
            name=str(c-1) + '/' + str(c) + ' (n=%0.f' % df11[(df11['school_name'] == T11[0]) & (df11['cycle'] == c) & (df11['decision_at'] <= current_of)].shape[0] + ')',
            xbins=dict(
                start=min(df_temp['wait']),
                end=max(df_temp['wait']),
                size=7
                ),
            meta='Yale, ' + str(c-1) + '/' + str(c) + ' (n=%0.f' % df_temp.shape[0] + ')<br>Avg. Wait: %0.f' % df_temp['wait'].mean(),
            hovertemplate='%{meta}<br>%{x} days<extra></extra>',
            showlegend=showlegend,
            legendgroup=str(c)
            ),
            row=row,
            col=col
        )

    if (row == 2) & (col == 2):
        col -= 1

    if (row == 1) & (col == 2):
        row += 1

    if (row == 1) & (col == 1):
        col += 1



updatemenu = []
button_schools = []

#  Button with one option for each school
for i, school in enumerate(T11):
    x = []
    name = []
    meta = []

    for r in ['all', 'A', 'R', 'WL']:
        for c in cycles:
            if r == 'all':
                df_temp = df11[(df11['school_name'] == school) &
                               (df11['decision_at'] <= current_of) & (df11['cycle'] == c)]
            else:
                df_temp = df11[(df11['school_name'] == school) & (df11['decision_at'] <= current_of) &
                                (df11['cycle'] == c) & (df11['decision'] == r)]

            x.append(df_temp['wait'])
            name.append(str(c-1) + '/' + str(c) + ' (n=%0.f' % df11[(df11['school_name'] == school) & (df11['cycle'] == c) & (df11['decision_at'] <= current_of)].shape[0] + ')')
            meta.append(T11_short[i] + ', ' + str(c-1) + '/' + str(c) + ' (n=%0.f' % df_temp.shape[0] + ')<br>Avg. Wait: %0.f' % df_temp['wait'].mean())

    button_schools.append(
        dict(
            method='update',
            label=T11_short[i],
            visible=True,
            args=[
                dict(
                x=x,
                name=name,
                meta=meta,
                hovertemplate='%{meta}<br>%{x} days<extra></extra>'
                )
            ],
        )
    )

#  Adjust updatemenus
updatemenu = []
menu = dict()
updatemenu.append(menu)

updatemenu[0]['buttons'] = button_schools
updatemenu[0]['direction'] = 'down'
updatemenu[0]['showactive'] = True
updatemenu[0]['pad'] = {'l': 10, 'r': 10, 't': 10}
updatemenu[0]['x'] = 1.02
updatemenu[0]['xanchor'] = 'left'
updatemenu[0]['y'] = 1.06
updatemenu[0]['yanchor'] = 'top'

# The two histograms are drawn on top of another
fig.update_layout(
    updatemenus=updatemenu,
    barmode='stack',
    legend_title='App. Cycle',
    colorway=['violet', 'seagreen', 'coral', 'cornflowerblue']*4,
    title={
        'text': 'Wait Times Distribution',
        'y': 0.945,
        'x': 0.46,
        'xanchor': 'center',
        'yanchor': 'top'
    },
    legend=dict(
        yanchor='bottom',
        y=0.00,
        xanchor='left',
        x=1.02,
        traceorder='reversed'
    )
)

fig.update_xaxes(title_text="Wait Time (days) - All", range=[10, 200], row=1, col=1)
fig.update_xaxes(title_text="Wait Time (days) - Accepted", range=[10, 200], row=1, col=2)
fig.update_xaxes(title_text="Wait Time (days) - Rejected", range=[10, 200], row=2, col=2)
fig.update_xaxes(title_text="Wait Time (days) - Waitlisted", range=[10, 200], row=2, col=1)

# Update yaxis properties
fig.update_yaxes(title_text="Count", row=1, col=1)
fig.update_yaxes(title_text="Count", row=1, col=2)
fig.update_yaxes(title_text="Count", row=2, col=1)
fig.update_yaxes(title_text="Count", row=2, col=2)

current_of = max(df11[df11['cycle'] == 21]['decision_at'])
fig.add_annotation(
    text='Current as of ' + str(current_of.month) + '/' + str(current_of.day) + '/2021 (-----)',
    xref='paper', yref='paper',
    x=1.135, y=-0.1,
    showarrow=False, font=dict(size=8, color='gray')
)

fig.add_annotation(
    text='Admissions data from LSData.org. Medians data from 7Sage.com. (c) 2021',
    xref='paper', yref='paper',
    x=0, y=-0.1,
    showarrow=False, font=dict(size=8, color='lightgray')
)

fig.show(config=dict(modeBarButtonsToRemove=['autoScale2d']))

# cwd = Path(getcwd())
# pio.write_html(fig, file=str(cwd.parent.absolute()) + '/docs/_includes/waithistogram.html', auto_open=False, config=dict(modeBarButtonsToRemove=['autoScale2d']))
# print('\nFinished writing to waithistogram.html.')

#
# #  Create print output to assess how wait times have changed by cycle and by decision
# print('\n' + '{0:<8} {1:<8} {2:<11} {3:<11} {4:<11}'.format('Cycle', 'Dec.', 'Avg. Wait', 'Std. Dev.', 'n='))
# print('-'*45)
# for s in ['a', 'r', 'w']:
#     for c in cycles:
#         durations_c_s = None
#         exec('durations_c_s = durations' + c + s)
#
#         t1 = str(int(c)-1) + '/' + c  # Cycle
#         t2 = s  # Result/decision
#         t3 = ''  # Avg. Wait
#         try:
#             t3 = str(int(durations_c_s.mean()))
#         except ValueError:
#             t3 = '--'
#         t4 = ''  # Std. Dev.
#         try:
#             t4 = str(int(durations_c_s.std()))
#         except ValueError:
#             t4 = '--'
#         t5 = ''  # n=
#         t5 = str(durations_c_s.shape[0])
#         print('{0:<8} {1:<8} {2:<11} {3:<11} {4:<11}'.format(t1, t2, t3, t4, t5))

#  Plot histogram (note: profiles with "Pending" as a result are included in calculations)
# day_lim = 250
# num_bins = int(math.ceil(day_lim/7)) + 1
# n, bins, patches = plt.hist([durations18[durations18 < day_lim], durations19[durations19 < day_lim],
#                              durations20[durations20 < day_lim], durations21[durations21 < day_lim]],
#                             bins=num_bins, stacked=True, density=True, label=custom_labels, zorder=3)
#
# plt.title('Number of Days from Sent to Decision, ' + school_name +
#           ', through ' + str(max(df_filtered['decision_at']))[5:-9] + ' Each Cycle (' +
#           str(len(durations18+durations19+durations20+durations21)) + ' samples)')
# plt.xlabel('Number of days')
# plt.ylabel('Frequency')
#
# #  Demarcate means by cycle
# for cwm in cycle_wait_means:
#     plt.axvline(x=cwm, linewidth=0.75, color='k', linestyle='--', zorder=5)
