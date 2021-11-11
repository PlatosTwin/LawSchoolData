import datetime as dt
from os import getcwd
from pathlib import Path

from itertools import chain
import numpy as np
import csv
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from pandas.plotting import register_matplotlib_converters
from plotly.graph_objs.layout import XAxis

register_matplotlib_converters()

#  Read-in admissions data
fname_admit = 'lsdata_clean.csv'
df11 = pd.read_csv(fname_admit, low_memory=False)

#  Drop rows with null sent_at and decision_at
df11 = df11.dropna(subset=['sent_at', 'decision_at'])

#  Convert sent_at and decision_at to datetime
df11.loc[:, 'sent_at'] = pd.to_datetime(df11['sent_at'])
df11.loc[:, 'decision_at'] = pd.to_datetime(df11['decision_at'])

cycles = [18, 19, 20, 21, 22]

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
             'Berkeley', 'Northwestern']

#  Create a figure with a second x axis, to plot percentage lines
layout = go.Layout(
    xaxis=XAxis(
        title='Date Sent',
        dtick='M1',
        tickformat='%B',
        ticklabelmode='period',
        range=[min(df11['sent_at']) - dt.timedelta(days=4), dt.datetime(2018, 3, 15)]
    ),
    xaxis2=XAxis(
        title='Percent',
        overlaying='x',
        side='top',
        dtick=10,
        ticks='outside',
        showgrid=False,
        zeroline=False
    ))

fig = go.Figure(layout=layout)

init = T11[0]
symbol = ['triangle-ne', 'triangle-se', 'triangle-sw', 'triangle-nw', 'circle']

#  Add decision (A, R, WL) scatter traces
for k, c in enumerate(cycles):
    fig.add_trace(go.Scatter(
        x=df11[(df11['school_name'] == init) & (df11['cycle'] == c)]['sent_at'],
        y=df11[(df11['school_name'] == init) & (df11['cycle'] == c)]['decision_at'],
        mode='markers',
        marker=dict(
            size=5,
            color=df11[(df11['school_name'] == init) & (df11['cycle'] == c)]['color'],
            symbol=df11[(df11['school_name'] == init) & (df11['cycle'] == c)]['marker'],
            line=dict(
                width=1,
                color=df11[(df11['school_name'] == init) & (df11['cycle'] == c)]['splitter'])),
        name=str(c - 1) + '/' + str(c) +
             ' (n=' + str(df11[(df11['school_name'] == init) & (df11['cycle'] == c) &
                               (df11['decision'].str.contains('|'.join(['A', 'R', 'WL'])))].shape[0]) + ')',
        customdata=df11[(df11['school_name'] == init) & (df11['cycle'] == c)],
        showlegend=False,
        legendgroup=c,
        hovertemplate='LSAT: %{customdata[3]:.0f}<br>GPA: %{customdata[4]}<br>'
                      'Sent: %{customdata[1]|%m/%d}<br>Decision: %{customdata[2]|%m/%d}<br>'
                      'Cycle: %{customdata[5]}<extra></extra>'
    )
    )

    fig.add_trace(go.Scatter(
        x=[pd.Series(dt.datetime(2001, 9, 1))],
        y=[pd.Series(dt.datetime(2001, 5, 1))],
        mode='markers',
        marker=dict(
            size=8,
            color='black',
            symbol=symbol[k]),
        name=str(c - 1) + '/' + str(c) +
             ' (n=' + str(df11[(df11['school_name'] == init) & (df11['cycle'] == c) &
                               (df11['decision'].str.contains('|'.join(['A', 'R', 'WL'])))].shape[0]) + ')',
        legendgroup=c
    )
    )

#  Calculate percentages from past cycles
dfpct = pd.DataFrame(columns=['school_name', 'pctn', 'pcta', 'pctr', 'pctw', 'chancea',
                              'totaln', 'totala', 'totalr', 'totalw', 'date'])

for school in T11:
    cycles_past = df11[(df11['school_name'] == school) & (df11['cycle'] != 22)]
    num_weeks = int((max(cycles_past['decision_at']) - min(cycles_past['decision_at'])).days / 7) + 1
    earliest = min(cycles_past['decision_at'])
    total_a = cycles_past[cycles_past['decision'] == 'A'].shape[0]
    total_r = cycles_past[cycles_past['decision'] == 'R'].shape[0]
    total_w = cycles_past[cycles_past['decision'] == 'WL'].shape[0]
    total_n = total_a + total_r + total_w

    for i in range(num_weeks):
        temp = cycles_past[cycles_past['decision_at'] <= earliest + i * dt.timedelta(weeks=1)]
        num_a = temp[temp['decision'] == 'A'].shape[0]
        num_r = temp[temp['decision'] == 'R'].shape[0]
        num_w = temp[temp['decision'] == 'WL'].shape[0]
        num_wd = temp[temp['decision'] == 'WD'].shape[0]  # number withdrawn

        try:
            chance_a = (total_a - num_a) / (cycles_past.shape[0] - num_a - num_r - num_w - num_wd)
        except ZeroDivisionError:
            chance_a = 0

        index = len(dfpct)
        dfpct.loc[index] = [school,
                            100 * (num_a + num_r + num_w) / total_n,
                            100 * num_a / total_a,
                            100 * num_r / total_r,
                            100 * num_w / total_w,
                            100 * chance_a,
                            total_n,
                            total_a,
                            total_r,
                            total_w,
                            earliest + i * dt.timedelta(weeks=1)]

#  Add percentage traces
#  Notified
dfpct_alpha = '0.4)'
fig.add_trace(go.Scatter(
    x=dfpct[dfpct['school_name'] == init]['pctn'],
    y=dfpct[dfpct['school_name'] == init]['date'],
    line=dict(color='RGBA(0,0,0,' + dfpct_alpha, width=1.25),
    mode='lines',
    xaxis='x2',
    name='Hist. Pct. Notified (n=%0.f' % max(dfpct[dfpct['school_name'] == init]['totaln']) + ')',
    legendgroup=1
),
)

#  Accepted
fig.add_trace(go.Scatter(
    x=dfpct[dfpct['school_name'] == init]['pcta'],
    y=dfpct[dfpct['school_name'] == init]['date'],
    line=dict(color='RGBA(0,177,64,' + dfpct_alpha, width=1.25),
    mode='lines',
    xaxis='x2',
    name='Hist. Pct. A (n=%0.f' % max(dfpct[dfpct['school_name'] == init]['totala']) + ')',
    legendgroup=2
)
)

#  Rejected
fig.add_trace(go.Scatter(
    x=dfpct[dfpct['school_name'] == init]['pctr'],
    y=dfpct[dfpct['school_name'] == init]['date'],
    line=dict(color='RGBA(255,0,0,' + dfpct_alpha, width=1.25),
    mode='lines',
    xaxis='x2',
    name='Hist. Pct. R (n=%0.f' % max(dfpct[dfpct['school_name'] == init]['totalr']) + ')',
    legendgroup=3
)
)

#  Waitlisted
fig.add_trace(go.Scatter(
    x=dfpct[dfpct['school_name'] == init]['pctw'],
    y=dfpct[dfpct['school_name'] == init]['date'],
    line=dict(color='RGBA(255,165,0,' + dfpct_alpha, width=1.25),
    mode='lines',
    xaxis='x2',
    name='Hist. Pct. WL (n=%0.f' % max(dfpct[dfpct['school_name'] == init]['totalw']) + ')',
    legendgroup=4
)
)

#  Chance of acceptance
fig.add_trace(go.Scatter(
    x=dfpct[dfpct['school_name'] == init]['chancea'],
    y=dfpct[dfpct['school_name'] == init]['date'],
    line=dict(color='RGBA(0,255,255,' + dfpct_alpha, width=1.25),
    mode='lines',
    xaxis='x2',
    name='Acceptance Likelihood',
    legendgroup=5
)
)

#  Update layout for percentage traces
fig.data[10].update(customdata=dfpct[dfpct['school_name'] == init],
                    hovertemplate='%{customdata[1]:.0f}%<br>%{customdata[10]|%m/%d}<extra></extra>')
fig.data[11].update(customdata=dfpct[dfpct['school_name'] == init],
                    hovertemplate='%{customdata[2]:.0f}%<br>%{customdata[10]|%m/%d}<extra></extra>')
fig.data[12].update(customdata=dfpct[dfpct['school_name'] == init],
                    hovertemplate='%{customdata[3]:.0f}%<br>%{customdata[10]|%m/%d}<extra></extra>')
fig.data[13].update(customdata=dfpct[dfpct['school_name'] == init],
                    hovertemplate='%{customdata[4]:.0f}%<br>%{customdata[10]|%m/%d}<extra></extra>')
fig.data[14].update(customdata=dfpct[dfpct['school_name'] == init],
                    hovertemplate='%{customdata[5]:.0f}%<br>%{customdata[10]|%m/%d}<extra></extra>')

#  Dropdown menu
updatemenu = []
buttons = []

constant_x = [pd.Series(dt.datetime(2001, 9, 1))]
constant_y = [pd.Series(dt.datetime(2001, 5, 1))]
#  Button with one option for each school
for i, school in enumerate(T11):
    x = [df11[(df11['school_name'] == school) & (df11['cycle'] == 18)]['sent_at']] + \
        constant_x + \
        [df11[(df11['school_name'] == school) & (df11['cycle'] == 19)]['sent_at']] + \
        constant_x + \
        [df11[(df11['school_name'] == school) & (df11['cycle'] == 20)]['sent_at']] + \
        constant_x + \
        [df11[(df11['school_name'] == school) & (df11['cycle'] == 21)]['sent_at']] + \
        constant_x + \
        [df11[(df11['school_name'] == school) & (df11['cycle'] == 22)]['sent_at']] + \
        constant_x + \
        [dfpct[dfpct['school_name'] == school][c] for c in ['pctn', 'pcta', 'pctr', 'pctw', 'chancea']]

    y = [df11[(df11['school_name'] == school) & (df11['cycle'] == 18)]['decision_at']] + \
        constant_y + \
        [df11[(df11['school_name'] == school) & (df11['cycle'] == 19)]['decision_at']] + \
        constant_y + \
        [df11[(df11['school_name'] == school) & (df11['cycle'] == 20)]['decision_at']] + \
        constant_y + \
        [df11[(df11['school_name'] == school) & (df11['cycle'] == 21)]['decision_at']] + \
        constant_y + \
        [df11[(df11['school_name'] == school) & (df11['cycle'] == 22)]['decision_at']] + \
        constant_y + \
        [dfpct[dfpct['school_name'] == school]['date']] * 5

    name = [str(c - 1) + '/' + str(c) +
            ' (n=' + str(df11[(df11['school_name'] == school) & (df11['cycle'] == c) &
                              (df11['decision'].str.contains('|'.join(['A', 'R', 'WL'])))].shape[0]) + ')'
            for c in cycles] + \
           [str(c - 1) + '/' + str(c) +
            ' (n=' + str(df11[(df11['school_name'] == school) & (df11['cycle'] == c) &
                              (df11['decision'].str.contains('|'.join(['A', 'R', 'WL'])))].shape[0]) + ')'
            for c in cycles] + \
           ['Hist. Pct. Notified (n=%0.f' % max(dfpct[dfpct['school_name'] == school]['totaln']) + ')',
            'Hist. Pct. A (n=%0.f' % max(dfpct[dfpct['school_name'] == school]['totala']) + ')',
            'Hist. Pct. R (n=%0.f' % max(dfpct[dfpct['school_name'] == school]['totalr']) + ')',
            'Hist. Pct. W (n=%0.f' % max(dfpct[dfpct['school_name'] == school]['totalw']) + ')',
            'Acceptance Likelihood']

    marker = [
        dict(
            size=5,
            color=df11[(df11['school_name'] == school) & (df11['cycle'] == 18)]['color'],
            symbol=df11[(df11['school_name'] == school) & (df11['cycle'] == 18)]['marker'],
            line=dict(
                width=1,
                color=df11[(df11['school_name'] == school) & (df11['cycle'] == 18)]['splitter'])),
        dict(
            size=8,
            color='black',
            symbol=symbol[0]),
        dict(
            size=5,
            color=df11[(df11['school_name'] == school) & (df11['cycle'] == 19)]['color'],
            symbol=df11[(df11['school_name'] == school) & (df11['cycle'] == 19)]['marker'],
            line=dict(
                width=1,
                color=df11[(df11['school_name'] == school) & (df11['cycle'] == 19)]['splitter'])),
        dict(
            size=8,
            color='black',
            symbol=symbol[1]),
        dict(
            size=5,
            color=df11[(df11['school_name'] == school) & (df11['cycle'] == 20)]['color'],
            symbol=df11[(df11['school_name'] == school) & (df11['cycle'] == 20)]['marker'],
            line=dict(
                width=1,
                color=df11[(df11['school_name'] == school) & (df11['cycle'] == 20)]['splitter'])),
        dict(
            size=8,
            color='black',
            symbol=symbol[2]),
        dict(
            size=5,
            color=df11[(df11['school_name'] == school) & (df11['cycle'] == 21)]['color'],
            symbol=df11[(df11['school_name'] == school) & (df11['cycle'] == 21)]['marker'],
            line=dict(
                width=1,
                color=df11[(df11['school_name'] == school) & (df11['cycle'] == 21)]['splitter'])),
        dict(
            size=8,
            color='black',
            symbol=symbol[3]),
        dict(
            size=5,
            color=df11[(df11['school_name'] == school) & (df11['cycle'] == 22)]['color'],
            symbol=df11[(df11['school_name'] == school) & (df11['cycle'] == 22)]['marker'],
            line=dict(
                width=1,
                color=df11[(df11['school_name'] == school) & (df11['cycle'] == 22)]['splitter'])),
        dict(
            size=8,
            color='black',
            symbol=symbol[4]),
    ]

    showlegend = [False, True] * 5 + [True] * 5

    legendgroup = list(np.array([[(i + 1) * m + 7, (i + 1) * m + 7] for m in cycles]).flatten()) + [1, 2, 3, 4, 5]

    buttons.append(
        dict(
            method='update',
            label=T11_short[i] + ' (n=' + str(df11[(df11['school_name'] == school) &
                                                   (df11['decision'].str.contains('|'.join(['A', 'R', 'WL'])))].shape[
                                                  0]) + ')',
            visible=True,
            args=[{
                'x': x,
                'y': y,
                'showlegend': showlegend,
                'legendgroup': legendgroup,
                'marker': marker,
                'name': name,
                'customdata': list(
                    np.array([[np.array(df11[(df11['school_name'] == school) & (df11['cycle'] == c)]) for i in [1, 2]]
                              for c in cycles], dtype=object).flatten()) +
                              [np.array(dfpct[dfpct['school_name'] == school]) for i in range(4, 9)]
            },
            ],
        )
    )

updatemenu = []
menu = dict()
updatemenu.append(menu)

updatemenu[0]['buttons'] = buttons
updatemenu[0]['direction'] = 'down'
updatemenu[0]['showactive'] = True
updatemenu[0]['pad'] = {'l': 10, 'r': 10, 't': 10}
updatemenu[0]['x'] = 1.02
updatemenu[0]['xanchor'] = 'left'
updatemenu[0]['y'] = 1.06
updatemenu[0]['yanchor'] = 'top'

#  Adjust layout
fig.update_layout(
    updatemenus=updatemenu,
    xaxis_title='Sent Date',
    yaxis_title='Decision Date',
    legend_title='App. Cycle + Percentages',
    autosize=True,
    margin=dict(l=120, r=100, autoexpand=True),
    height=700,
    title={
        'text': 'Decision Timeline',
        'y': 0.98,
        'x': 0.46,
        'xanchor': 'center',
        'yanchor': 'top'
    },
    legend=dict(
        yanchor='bottom',
        y=0.00,
        xanchor='left',
        x=1.02,
        traceorder='normal'
    )
)

#  Add current-of line
with open('/Users/Shared/lsdata.csv', newline='') as f:
    reader = csv.reader(f)
    for row in reader:
        updated = row[0]
        break

current_of = '2017' + updated[updated.index(':') + 2:updated.index(':') + 12][4:]
current_of = dt.datetime.strptime(current_of, '%Y-%m-%d')

fig.add_shape(
    type='line',
    x0=dt.datetime(2017, 5, 1),
    y0=current_of,
    x1=dt.datetime(2018, 5, 1),
    y1=current_of,
    line=dict(
        color='LightSeaGreen',
        width=1,
        dash='dashdot',
    )
)

#  Add one-month spaced y=x lines
for i in range(7):
    fig.add_shape(
        type='line',
        x0=dt.datetime(2017, 1, 1),
        y0=dt.datetime(2017, 1 + i, 1),
        x1=dt.datetime(2019, 1, 1),
        y1=dt.datetime(2019, 1 + i, 1),
        line=dict(
            color='LightGray',
            width=0.65,
            dash='dashdot',
        )
    )

fig.update_yaxes(
    dtick='M1',
    tickformat='%B',
    ticklabelmode='period',
    range=[min(df11['decision_at']) - dt.timedelta(days=7), dt.datetime(2018, 5, 15)],
)

# fig.show(config=dict(modeBarButtonsToRemove=['autoScale2d']))

cwd = Path(getcwd())
pio.write_html(fig,
               file=str(cwd.parent.absolute()) + '/docs/_includes/timeline.html',
               auto_open=False,
               config=dict(modeBarButtonsToRemove=['autoScale2d']))
print('\nFinished writing to timeline.html.')
