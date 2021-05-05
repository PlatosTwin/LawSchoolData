from os import getcwd
from pathlib import Path
import datetime as dt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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
             'Berkeley', 'Northwestern']

#  Create 2x2 figure with origin at upper-left quadrant
fig = make_subplots(rows=2, cols=2, start_cell='top-left')

current_of = max(df11[df11['cycle'] == 21]['decision_at'])

#  Set initial histogram traces
row = 1
col = 1
showlegend = False

for d in ['all', 'A', 'R', 'WL']:
    for c in cycles:
        if d == 'all':
            df_temp = df11[(df11['school_name'] == T11[0]) & (df11['cycle'] == c) & (df11['decision_at'] <= current_of)]
            showlegend = True
        else:
            df_temp = df11[(df11['school_name'] == T11[0]) & (df11['cycle'] == c) &
                           (df11['decision_at'] <= current_of) & (df11['decision'] == d)]
            showlegend = False

        fig.add_trace(go.Histogram(
            x=df_temp['wait'],
            name=str(c-1) + '/' + str(c) + ' (n=%0.f' % df11[(df11['school_name'] == T11[0]) & (df11['cycle'] == c) &
                                                             (df11['decision_at'] <= current_of)].shape[0] + ')',
            xbins=dict(
                start=min(df_temp['wait']),
                end=max(df_temp['wait']),
                size=7
                ),
            meta='Yale, ' + str(c-1) + '/' + str(c) + ' (n=%0.f' % df_temp.shape[0] +
                 ')<br>Avg. Wait: %0.f' % df_temp['wait'].mean(),
            hovertemplate='%{meta}<br>%{x} days<extra></extra>',
            showlegend=showlegend,
            legendgroup=str(c)
            ),
            row=row,
            col=col
        )

    #  Advance to the next quadrant, clockwise
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

    for d in ['all', 'A', 'R', 'WL']:
        for c in cycles:
            if d == 'all':
                df_temp = df11[(df11['school_name'] == school) &
                               (df11['decision_at'] <= current_of) & (df11['cycle'] == c)]
            else:
                df_temp = df11[(df11['school_name'] == school) & (df11['decision_at'] <= current_of) &
                               (df11['cycle'] == c) & (df11['decision'] == d)]

            x.append(df_temp['wait'])
            name.append(str(c-1) + '/' + str(c) + ' (n=%0.f' % df11[(df11['school_name'] == school) &
                                                                    (df11['cycle'] == c) &
                                                                    (df11['decision_at'] <= current_of)].shape[0] + ')')
            meta.append(T11_short[i] + ', ' + str(c-1) + '/' + str(c) +
                        ' (n=%0.f' % df_temp.shape[0] + ')<br>Avg. Wait: %0.f' % df_temp['wait'].mean())

    button_schools.append(
        dict(
            method='update',
            label=T11_short[i] + ' (n=%0.f' % df11[(df11['school_name'] == school) &
                                                   (df11['decision_at'] <= current_of)].shape[0] + ')',
            visible=True,
            args=[
                dict(
                    x=x,
                    name=name,
                    meta=meta
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

#  Adjust layout
fig.update_layout(
    updatemenus=updatemenu,
    barmode='stack',
    legend_title='App. Cycle',
    colorway=['violet', 'seagreen', 'coral', 'cornflowerblue']*4,
    autosize=True,
    margin=dict(l=75, r=100, autoexpand=True),
    height=700,
    title={
        'text': 'Wait Times Distribution Through ' + str(current_of.month) + '/' + str(current_of.day) + ' Each Cycle',
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

# Update xaxis properties
fig.update_xaxes(title_text='Wait Time (days) - All', range=[10, 200], row=1, col=1)
fig.update_xaxes(title_text='Wait Time (days) - Accepted', range=[10, 200], row=1, col=2)
fig.update_xaxes(title_text='Wait Time (days) - Rejected', range=[10, 200], row=2, col=2)
fig.update_xaxes(title_text='Wait Time (days) - Waitlisted', range=[10, 200], row=2, col=1)

# Update yaxis properties
fig.update_yaxes(title_text='Count', row=1, col=1)
fig.update_yaxes(title_text='Count', row=1, col=2)
fig.update_yaxes(title_text='Count', row=2, col=1)
fig.update_yaxes(title_text='Count', row=2, col=2)

# fig.show(config=dict(modeBarButtonsToRemove=['autoScale2d']))

cwd = Path(getcwd())
pio.write_html(fig, file=str(cwd.parent.absolute()) + '/docs/_includes/waithistogram.html', auto_open=False, config=dict(modeBarButtonsToRemove=['autoScale2d']))
print('\nFinished writing to waithistogram.html.')
