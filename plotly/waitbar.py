from os import getcwd
from pathlib import Path
import datetime as dt
import plotly.graph_objects as go
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

dfwait = pd.DataFrame(columns=['school_name', 'cycle', 'decision', 'wait', 'stdev', 'n'])

#  Calculate mean wait times and other wait statistics
current_of = max(df11[df11['cycle'] == 21]['decision_at'])
dec = ['all', 'A', 'R', 'WL']
for school in T11:
    for d in dec:
        for c in cycles:
            if d == 'all':
                df_temp = df11[(df11['decision'].str.contains('|'.join(['A', 'R', 'WL']))) &
                               (df11['school_name'] == school) & (df11['cycle'] == c) &
                               (df11['decision_at'] <= current_of)]
            else:
                df_temp = df11[(df11['decision'] == d) & (df11['school_name'] == school) &
                               (df11['cycle'] == c) & (df11['decision_at'] <= current_of)]

            index = len(dfwait)
            dfwait.loc[index] = [school, c, d, df_temp['wait'].mean(), df_temp['wait'].std(), df_temp['wait'].shape[0]]

dfwait['wait'] = dfwait.apply(lambda row: int(row['wait']), axis=1)
dfwait['stdev'] = dfwait.apply(lambda row: int(row['stdev']), axis=1)

#  Set initial traces
fig = go.Figure()

x = ['Total', 'Accepted', 'Rejected', 'Waitlisted']

for c in cycles:
    y = [dfwait[(dfwait['school_name'] == T11[0]) & (dfwait['cycle'] == c) &
                (dfwait['decision'] == d)]['wait'].values[0] for d in dec]
    eb = [dfwait[(dfwait['school_name'] == T11[0]) & (dfwait['cycle'] == c) &
                 (dfwait['decision'] == d)]['stdev'].values[0] for d in dec]
    fig.add_trace(go.Bar(
        x=x,
        y=y,
        name=str(c-1) + '/' + str(c),
        text=y,
        textposition='auto',
        meta=[dfwait[(dfwait['school_name'] == T11[0]) & (dfwait['cycle'] == c) &
                     (dfwait['decision'] == d)]['n'].values[0] for d in dec],
        hovertemplate='%{y} days<br>(n=%{meta})<extra></extra>',
        error_y=dict(
            type='data',
            array=eb,
            color='darkgray',
            thickness=1.5
        )
    )
    )

updatemenu = []
button_schools = []

#  Button with one option for each school
for i, school in enumerate(T11):
    y = []
    name = []
    eb = []
    meta = []

    for c in cycles:
        y.append([dfwait[(dfwait['school_name'] == school) & (dfwait['cycle'] == c) &
                         (dfwait['decision'] == d)]['wait'].values[0] for d in dec])
        name.append(str(c - 1) + '/' + str(c))
        eb.append(
            dict(
                type='data',
                array=[dfwait[(dfwait['school_name'] == school) & (dfwait['cycle'] == c) &
                              (dfwait['decision'] == d)]['stdev'].values[0] for d in dec],
                color='darkgray',
                thickness=1.5
                )
            )
        meta.append([dfwait[(dfwait['school_name'] == school) & (dfwait['cycle'] == c) &
                            (dfwait['decision'] == d)]['n'].values[0] for d in dec])

    button_schools.append(
        dict(
            method='update',
            label=T11_short[i],
            visible=True,
            args=[
                dict(
                    y=y,
                    name=name,
                    meta=meta,
                    text=y,
                    error_y=eb
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
    barmode='group',
    legend_title='App. Cycle',
    title={
        'text': 'Wait Times Through ' + str(current_of.month) + '/' + str(current_of.day) + ' Each Cycle',
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

fig.update_xaxes(title_text='Decision')
fig.update_yaxes(title_text='Wait Time (days)')

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
# pio.write_html(fig, file=str(cwd.parent.absolute()) + '/docs/_includes/waitbar.html', auto_open=False, config=dict(modeBarButtonsToRemove=['autoScale2d']))
# print('\nFinished writing to waitbar.html.')
