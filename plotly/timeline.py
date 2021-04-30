from os import getcwd
from pathlib import Path
import numpy as np
import datetime as dt
import plotly.graph_objects as go
from plotly.graph_objs.layout import XAxis
import plotly.io as pio
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

#  Add decision (A, R, WL) traces
for c in cycles:
    fig.add_trace(go.Scatter(
        x=df11[(df11['school_name'] == T11[0]) & (df11['cycle'] == c)]['sent_at'],
        y=df11[(df11['school_name'] == T11[0]) & (df11['cycle'] == c)]['decision_at'],
        mode='markers',
        marker=dict(
            size=5,
            color=df11[(df11['school_name'] == T11[0]) & (df11['cycle'] == c)]['color'],
            symbol=df11[(df11['school_name'] == T11[0]) & (df11['cycle'] == c)]['marker'],
            line=dict(
                width=1,
                color=df11[(df11['school_name'] == T11[0]) & (df11['cycle'] == c)]['splitter'])),
        name=str(c-1) + '/' + str(c) +
             ' (n=' + str(df11[(df11['school_name'] == T11[0]) & (df11['cycle'] == c) &
                               (df11['decision'].str.contains('|'.join(['A', 'R', 'WL'])))].shape[0]) + ')',
        customdata=df11[(df11['school_name'] == T11[0]) & (df11['cycle'] == c)],
        hovertemplate='LSAT: %{customdata[9]:.0f}<br>GPA: %{customdata[13]}<br>'
                      'Sent: %{customdata[1]|%m/%d}<br>Decision: %{customdata[7]|%m/%d}<extra></extra>'
        )
    )


#  Add percent lines for past cycles
pct_trace = pd.DataFrame(columns=['school_name', 'pctn', 'pcta', 'pctr', 'pctw', 'chancea',
                                  'totaln', 'totala', 'totalr', 'totalw', 'date'])

for school in T11:
    cycles_past = df11[(df11['school_name'] == school) & (df11['cycle'] != 21)]
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

        index = len(pct_trace)
        pct_trace.loc[index] = [school,
                                100*(num_a + num_r + num_w)/total_n,
                                100*num_a/total_a,
                                100*num_r/total_r,
                                100*num_w/total_w,
                                100*chance_a,
                                total_n,
                                total_a,
                                total_r,
                                total_w,
                                earliest + i*dt.timedelta(weeks=1)]

#  Notified
pct_trace_alpha = '0.4)'
fig.add_trace(go.Scatter(
    x=pct_trace[pct_trace['school_name'] == T11[0]]['pctn'],
    y=pct_trace[pct_trace['school_name'] == T11[0]]['date'],
    line=dict(color='RGBA(0,0,0,' + pct_trace_alpha, width=1.25),
    mode='lines',
    xaxis='x2',
    name='Hist. Pct. Notified (n=%0.f' % max(pct_trace[pct_trace['school_name'] == T11[0]]['totaln']) + ')'),
)

#  Accepted
fig.add_trace(go.Scatter(
    x=pct_trace[pct_trace['school_name'] == T11[0]]['pcta'],
    y=pct_trace[pct_trace['school_name'] == T11[0]]['date'],
    line=dict(color='RGBA(0,177,64,' + pct_trace_alpha, width=1.25),
    mode='lines',
    xaxis='x2',
    name='Hist. Pct. A (n=%0.f' % max(pct_trace[pct_trace['school_name'] == T11[0]]['totala']) + ')')
)

#  Rejected
fig.add_trace(go.Scatter(
    x=pct_trace[pct_trace['school_name'] == T11[0]]['pctr'],
    y=pct_trace[pct_trace['school_name'] == T11[0]]['date'],
    line=dict(color='RGBA(255,0,0,' + pct_trace_alpha, width=1.25),
    mode='lines',
    xaxis='x2',
    name='Hist. Pct. R (n=%0.f' % max(pct_trace[pct_trace['school_name'] == T11[0]]['totalr']) + ')')
)

#  Waitlisted
fig.add_trace(go.Scatter(
    x=pct_trace[pct_trace['school_name'] == T11[0]]['pctw'],
    y=pct_trace[pct_trace['school_name'] == T11[0]]['date'],
    line=dict(color='RGBA(255,165,0,' + pct_trace_alpha, width=1.25),
    mode='lines',
    xaxis='x2',
    name='Hist. Pct. WL (n=%0.f' % max(pct_trace[pct_trace['school_name'] == T11[0]]['totalw']) + ')')
)

#  Chance of acceptance
fig.add_trace(go.Scatter(
    x=pct_trace[pct_trace['school_name'] == T11[0]]['chancea'],
    y=pct_trace[pct_trace['school_name'] == T11[0]]['date'],
    line=dict(color='RGBA(0,255,255,' + pct_trace_alpha, width=1.25),
    mode='lines',
    xaxis='x2',
    name='Acceptance Likelihood')
)

fig.data[4].update(customdata=pct_trace[pct_trace['school_name'] == T11[0]],
                   hovertemplate='%{customdata[1]:.0f}%<br>%{customdata[10]|%m/%d}<extra></extra>')
fig.data[5].update(customdata=pct_trace[pct_trace['school_name'] == T11[0]],
                   hovertemplate='%{customdata[2]:.0f}%<br>%{customdata[10]|%m/%d}<extra></extra>')
fig.data[6].update(customdata=pct_trace[pct_trace['school_name'] == T11[0]],
                   hovertemplate='%{customdata[3]:.0f}%<br>%{customdata[10]|%m/%d}<extra></extra>')
fig.data[7].update(customdata=pct_trace[pct_trace['school_name'] == T11[0]],
                   hovertemplate='%{customdata[4]:.0f}%<br>%{customdata[10]|%m/%d}<extra></extra>')
fig.data[8].update(customdata=pct_trace[pct_trace['school_name'] == T11[0]],
                   hovertemplate='%{customdata[5]:.0f}%<br>%{customdata[10]|%m/%d}<extra></extra>')

#  Dropdown menu
updatemenu = []
buttons = []

#  Button with one option for each school
for i, school in enumerate(T11):
    x = [df11[(df11['school_name'] == school) & (df11['cycle'] == c)]['sent_at'] for c in cycles] + \
        [pct_trace[pct_trace['school_name'] == school][c] for c in ['pctn', 'pcta', 'pctr', 'pctw', 'chancea']]

    y = [df11[(df11['school_name'] == school) & (df11['cycle'] == c)]['decision_at'] for c in cycles] +\
        [pct_trace[pct_trace['school_name'] == school]['date']]*5

    name = [str(c-1) + '/' + str(c) +
            ' (n=' + str(df11[(df11['school_name'] == school) & (df11['cycle'] == c) &
                              (df11['decision'].str.contains('|'.join(['A', 'R', 'WL'])))].shape[0]) + ')'
            for c in cycles] + \
           ['Hist. Pct. Notified (n=%0.f' % max(pct_trace[pct_trace['school_name'] == school]['totaln']) + ')',
            'Hist. Pct. A (n=%0.f' % max(pct_trace[pct_trace['school_name'] == school]['totala']) + ')',
            'Hist. Pct. R (n=%0.f' % max(pct_trace[pct_trace['school_name'] == school]['totalr']) + ')',
            'Hist. Pct. W (n=%0.f' % max(pct_trace[pct_trace['school_name'] == school]['totalw']) + ')',
            'Acceptance Likelihood']

    buttons.append(
        dict(
            method='update',
            label=T11_short[i] + ' (n=' + str(df11[(df11['school_name'] == school) &
                                                   (df11['decision'].str.contains('|'.join(['A', 'R', 'WL'])))].shape[0]) + ')',
            visible=True,
            args=[{
                'x': x,
                'y': y,
                'marker':
                    [dict(
                        size=5,
                        color=df11[(df11['school_name'] == school) & (df11['cycle'] == c)]['color'],
                        symbol=df11[(df11['school_name'] == school) & (df11['cycle'] == c)]['marker'],
                        line=dict(
                            width=1,
                            color=df11[(df11['school_name'] == school) & (df11['cycle'] == c)]['splitter'])) for c in cycles],
                'name': name,
                'customdata': [np.array(df11[(df11['school_name'] == school) & (df11['cycle'] == c)]) for c in cycles] +
                              [np.array(pct_trace[pct_trace['school_name'] == school]) for i in range(4, 9)],
                'hovertemplate':['LSAT: %{customdata[9]:.0f}<br>GPA: %{customdata[13]}<br>'
                                 'Sent: %{customdata[1]|%m/%d}<br>Decision: %{customdata[7]|%m/%d}<extra></extra>']*4 +
                                ['%{customdata[1]:.0f}%<br>%{customdata[10]|%m/%d}<extra></extra>',
                                 '%{customdata[2]:.0f}%<br>%{customdata[10]|%m/%d}<extra></extra>',
                                 '%{customdata[3]:.0f}%<br>%{customdata[10]|%m/%d}<extra></extra>',
                                 '%{customdata[4]:.0f}%<br>%{customdata[10]|%m/%d}<extra></extra>',
                                 '%{customdata[5]:.0f}%<br>%{customdata[10]|%m/%d}<extra></extra>']
                },
            ],
        )
    )

#  Adjust updatemenus
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

fig.update_layout(
    updatemenus=updatemenu,
    xaxis_title='Sent Date',
    yaxis_title='Decision Date',
    legend_title='App. Cycle',
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

fig.add_shape(
    type='line',
    x0=dt.datetime(2017, 5, 1),
    y0=max(df11[df11['cycle'] == 21]['decision_at']),
    x1=dt.datetime(2018, 5, 1),
    y1=max(df11[df11['cycle'] == 21]['decision_at']),
    line=dict(
        color='LightSeaGreen',
        width=1,
        dash='dashdot',
    )
)

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

current_of = max(df11[df11['cycle'] == 21]['decision_at'])
fig.add_annotation(
    text='Current as of ' + str(current_of.month) + '/' + str(current_of.day) + '/2021 (-----)',
    xref='paper', yref='paper',
    x=1.175, y=-0.1,
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
# pio.write_html(fig, file=str(cwd.parent.absolute()) + '/docs/_includes/timeline.html', auto_open=False, config=dict(modeBarButtonsToRemove=['autoScale2d']))
# print('\nFinished writing to timeline.html.')
