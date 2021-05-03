from os import getcwd
from pathlib import Path
import numpy as np
import datetime as dt
import plotly.graph_objects as go
import plotly.io as pio
import pandas as pd
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

#  Read-in admissions data
fname_admit = 'lsdata_clean.csv'
df11 = pd.read_csv(fname_admit, low_memory=False)

fname_percentiles = '/Users/Shared/lsmedians.csv'
dff = pd.read_csv(fname_percentiles, low_memory=False)
dff = dff[:20]  # Limit to top twenty schools
dff.loc[:, 'Yield'] = dff['Yield'].map(lambda x: float(x[:-1])/100)
dff.loc[:, 'Acceptance Rate'] = dff['Acceptance Rate'].map(lambda x: float(x[:-1])/100)

#  Convert sent_at and decision_at to datetime
df11.loc[:, 'sent_at'] = pd.to_datetime(df11['sent_at'])
df11.loc[:, 'decision_at'] = pd.to_datetime(df11['decision_at'])

cycles = [18, 19, 20, 21]

T11 = ['Yale University', 'Harvard University', 'Stanford University', 'University of Chicago',
       'Columbia University', 'New York University', 'University of Pennsylvania', 'University of Virginia',
       'University of Michigan', 'University of California—Berkeley', 'Northwestern University']

T11_short = ['Yale', 'Harvard', 'Stanford', 'UChicago', 'Columbia', 'NYU', 'UPenn', 'Virginia', 'Michigan',
             'Berkeley', 'Northwestern ']

fig = go.Figure()

#  Percentage of applicants who reported sent_at date on LSData
total_a = 0
total_app = 0
for school in T11:
    school_t_a = dff[dff['School'] == school]['1st Yr Class'].values[0]/dff[dff['School'] == school]['Yield'].values[0]
    total_a += school_t_a
    school_t_app = school_t_a/dff[dff['School'] == school]['Acceptance Rate'].values[0]
    total_app += school_t_app

    fig.add_trace(go.Scatter(
        x=cycles,
        y=[100*df11[(df11['school_name'] == school) & (df11['cycle'] == c)].shape[0]/school_t_app for c in cycles],
        mode='lines+markers',
        name=school,
        meta=school,
        hovertemplate='%{meta}<br>20%{x} - %{y:.1f}%<extra></extra>'
        )
    )

#  Across the T11
fig.add_trace(go.Scatter(
    x=cycles,
    y=[100*df11[(df11['school_name'].str.contains('|'.join(T11))) & (df11['cycle'] == c)].shape[0]/total_app for c in cycles],
    mode='lines+markers',
    name='Top 11',
    meta='Top 11',
    hovertemplate='%{meta}<br>20%{x} - %{y:.1f}%<extra></extra>'
    )
)

updatemenu = []
buttons = []

x1 = np.tile(cycles, (len(T11)+1, 1))
x2 = np.tile(cycles, (len(T11)+1, 1))
y1 = []
y2 = []

for school in T11:
    school_t_a = dff[dff['School'] == school]['1st Yr Class'].values[0]/dff[dff['School'] == school]['Yield'].values[0]
    school_t_app = school_t_a/dff[dff['School'] == school]['Acceptance Rate'].values[0]
    y1.append([100*df11[(df11['school_name'] == school) & (df11['cycle'] == c)].shape[0]/school_t_app for c in cycles])
    y2.append([100*df11[(df11['school_name'] == school) & (df11['cycle'] == c) & (df11['decision'] == 'A')].shape[0]/school_t_a for c in cycles])

y1.append([100*df11[(df11['school_name'].str.contains('|'.join(T11))) & (df11['cycle'] == c)].shape[0]/total_app for c in cycles])
y2.append([100*df11[(df11['school_name'].str.contains('|'.join(T11))) & (df11['cycle'] == c) & (df11['decision'] == 'A')].shape[0]/total_a for c in cycles])

buttons = list([
    dict(                   # All applicants
        method='update',
        visible=True,
        args=[{
            'x': x1,
            'y': y1,
            'mode': 'lines+markers',
            'name': T11_short + ['Top 11']
            }],
        label='All Applicants'),
    dict(                   # Accepted applicants
        method='update',
        visible=True,
        args=[{
            'x': x2,
            'y': y2,
            'mode': 'lines+markers',
            'name': T11_short + ['Top 11']
            }],
        label='Accepted Applicants')
]
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
    xaxis_title='Cycle',
    yaxis_title='Percentage',
    legend_title='App. Cycle',
    title={
        'text': 'LSData Applicant Pool vs. Total Applicant Pool',
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
        traceorder='normal'
    )
)

fig.update_xaxes(
    tickmode='array',
    tickvals=cycles,
    range=[min(cycles)-0.25, max(cycles)+0.25]
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
# pio.write_html(fig, file=str(cwd.parent.absolute()) + '/docs/_includes/poolscatter.html', auto_open=False, config=dict(modeBarButtonsToRemove=['autoScale2d']))
# print('\nFinished writing to poolscatter.html.')
