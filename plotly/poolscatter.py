from os import getcwd
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()

#  Read-in admissions data
fname_admit = 'lsdata_clean.csv'
df11 = pd.read_csv(fname_admit, low_memory=False)

#  Read-in medians/schools data
fname_meds = 'lsmedians.csv'
dfmeds = pd.read_csv(fname_meds, low_memory=False)

cycles = [18, 19, 20, 21]

T11 = ['Yale University', 'Harvard University', 'Stanford University', 'University of Chicago',
       'Columbia University', 'New York University', 'University of Pennsylvania', 'University of Virginia',
       'University of Michigan', 'University of California—Berkeley', 'Northwestern University']

T11_short = ['Yale', 'Harvard', 'Stanford', 'UChicago', 'Columbia', 'NYU', 'UPenn', 'Virginia', 'Michigan',
             'Berkeley', 'Northwestern']

#  Set 2021 medians data to 2020 medians data, until 2021 data becomes available
for school in T11:
    index = dfmeds[(dfmeds['school_name'] == school) & (dfmeds['cycle'] == 21)].index[0]
    dfmeds.loc[index] = \
        list(dfmeds[(dfmeds['school_name'] == school) & (dfmeds['cycle'] == 20)].values[0][:-1]) + [21]

fig = go.Figure()

#  Plot percentage of all applicants who reported LSAT, GPA, and a submitted application, in T11
for i, school in enumerate(T11):
    fig.add_trace(go.Scatter(
        x=cycles,
        y=[100 * df11[(df11['school_name'] == school) & (df11['cycle'] == c)].shape[0] /
           dfmeds[(dfmeds['school_name'] == school) & (dfmeds['cycle'] == c)]['total_applicants'].values[0] for c in cycles],
        mode='lines+markers',
        name=T11_short[i],
        meta=[T11_short[i] + '<br>' + 'LSData Volume: ' +
              str(df11[(df11['school_name'] == school) & (df11['cycle'] == c)].shape[0]) for c in cycles],
        hovertemplate='%{meta}<br>Pct. of Total: %{y:.1f}%<extra></extra>'
        )
    )

#  Same percentage as above, across the T11 in aggregate
fig.add_trace(go.Scatter(
    x=cycles,
    y=[100 * df11[(df11['school_name'].str.contains('|'.join(T11))) & (df11['cycle'] == c)].shape[0] /
       dfmeds[dfmeds['cycle'] == c]['total_applicants'].sum() for c in cycles],
    mode='lines+markers',
    name='Top 11',
    meta=['Top 11' + '<br>' + 'LSData Volume: ' +
          str(df11[(df11['school_name'].str.contains('|'.join(T11))) & (df11['cycle'] == c)].shape[0]) for c in cycles],
    hovertemplate='%{meta}<br>Pct. of Total: %{y:.1f}%<extra></extra>'
    )
)

#  Add dropdown buttons
x1 = np.tile(cycles, (len(T11)+1, 1))
x2 = np.tile(cycles, (len(T11)+1, 1))
y1 = []
y2 = []

#  For individual schools
for school in T11:
    #  All applicants
    y1.append([100 * df11[(df11['school_name'] == school) & (df11['cycle'] == c)].shape[0] /
               dfmeds[(dfmeds['school_name'] == school) & (dfmeds['cycle'] == c)]['total_applicants'].values[0] for c in cycles])

    #  Accepted applicants
    y2.append([100 * df11[(df11['school_name'] == school) & (df11['cycle'] == c) & (df11['decision'] == 'A')].shape[0] /
               dfmeds[(dfmeds['school_name'] == school) & (dfmeds['cycle'] == c)]['total_accepted'].values[0] for c in cycles])

#  For T11 as collective: all and accepted applicants
y1.append([100 * df11[(df11['school_name'].str.contains('|'.join(T11))) & (df11['cycle'] == c)].shape[0] /
           dfmeds[dfmeds['cycle'] == c]['total_applicants'].sum() for c in cycles])
y2.append([100 * df11[(df11['school_name'].str.contains('|'.join(T11))) & (df11['cycle'] == c) &
                    (df11['decision'] == 'A')].shape[0] /
           dfmeds[dfmeds['cycle'] == c]['total_accepted'].sum() for c in cycles])

meta1 = [[T11_short[i] + '<br>' + 'LSData Volume: ' +
          str(df11[(df11['school_name'] == school) & (df11['cycle'] == c)].shape[0]) for c in cycles]
         for i, school in enumerate(T11)]
meta1.append(['Top 11' + '<br>' + 'LSData Volume: ' +
              str(df11[(df11['school_name'].str.contains('|'.join(T11))) &
                       (df11['cycle'] == c)].shape[0]) for c in cycles])

meta2 = [[T11_short[i] + '<br>' + 'LSData Volume: ' +
          str(df11[(df11['school_name'] == school) & (df11['decision'] == 'A') &
                   (df11['cycle'] == c)].shape[0]) for c in cycles] for i, school in enumerate(T11)]
meta2.append(['Top 11' + '<br>' + 'LSData Volume: ' +
              str(df11[(df11['school_name'].str.contains('|'.join(T11))) &
                       (df11['decision'] == 'A') &
                       (df11['cycle'] == c)].shape[0]) for c in cycles])

buttons = list([
    dict(                   # All applicants
        method='update',
        visible=True,
        args=[{
            'x': x1,
            'y': y1,
            'name': T11_short + ['Top 11'],
            'meta': meta1,
            }],
        label='All Applicants'),
    dict(                   # Accepted applicants
        method='update',
        visible=True,
        args=[{
            'x': x2,
            'y': y2,
            'name': T11_short + ['Top 11'],
            'meta': meta2,
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

#  Adjust layout
fig.update_layout(
    updatemenus=updatemenu,
    xaxis_title='Cycle',
    yaxis_title='Percentage',
    legend_title='School',
    autosize=True,
    margin=dict(l=75, r=100, autoexpand=True),
    height=700,
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

# fig.show(config=dict(modeBarButtonsToRemove=['autoScale2d']))

cwd = Path(getcwd())
pio.write_html(fig,
               file=str(cwd.parent.absolute()) + '/docs/_includes/poolscatter.html',
               auto_open=False,
               config=dict(modeBarButtonsToRemove=['autoScale2d']))
print('\nFinished writing to poolscatter.html.')
