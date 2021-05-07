from os import getcwd
from pathlib import Path
import datetime as dt
import plotly.graph_objects as go
import plotly.io as pio
import pandas as pd
from pandas.plotting import register_matplotlib_converters
from plotly.subplots import make_subplots
import math
register_matplotlib_converters()

#  Read-in admissions data
fname_admit = 'lsdata_clean.csv'
df11 = pd.read_csv(fname_admit, low_memory=False)

df11 = df11.dropna(subset=['sent_at', 'decision_at'])

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

current_of = max(df11[df11['cycle'] == 21]['decision_at'])

#  Calculate mean wait times and other wait statistics
dftime = pd.DataFrame(columns=['school_name', 'cycle', 'month',
                               'acceptance_rate',
                               'lsat_mean', 'lsat_std',
                               'gpa_mean', 'gpa_std',
                               'n', 'na'])

months = ['September', 'October', 'November', 'December', 'January', 'February', 'March']
for school in T11:
    for i, m in enumerate(months):
        for c in cycles:
            if i < 3:
                t_min = dt.datetime(2017, i + 9, 1)
                t_max = dt.datetime(2017, i + 10, 1)
            elif i == 3:
                t_min = dt.datetime(2017, 12, 1)
                t_max = dt.datetime(2018, 1, 1)
            else:
                t_min = dt.datetime(2018, i - 3, 1)
                t_max = dt.datetime(2018, i - 2, 1)

            df_temp = df11[(df11['school_name'] == school) & (df11['cycle'] == c) &
                           (df11['sent_at'] >= t_min) &
                           (df11['sent_at'] < t_max)]

            try:
                acceptance_rate = round(
                    100*df_temp[df_temp['decision'] == 'A'].shape[0] /
                    df_temp[df_temp['decision'].str.contains('|'.join(['A', 'R', 'WL']))].shape[0], 2)
            except ZeroDivisionError:
                acceptance_rate = 0.0

            lsat_mean = df_temp['lsat'].mean()
            lsat_std = round(df_temp['lsat'].std(), 1) if not math.isnan(df_temp['lsat'].std()) else 0.0
            gpa_mean = df_temp['gpa'].mean()
            gpa_std = round(df_temp['gpa'].std(), 1) if not math.isnan(df_temp['gpa'].std()) else 0.0
            n = df_temp.shape[0]
            na = df_temp[df_temp['decision'] == 'A'].shape[0]

            index = len(dftime)
            dftime.loc[index] = [school, c, m, acceptance_rate, lsat_mean, lsat_std, gpa_mean, gpa_std, n, na]

#  Set initial traces
fig = make_subplots(rows=2, cols=2, start_cell='top-left')

row = 1
col = 1
for b in [['acceptance_rate', '%'], ['lsat_mean', ' LSAT', 'lsat_std'], ['gpa_mean', ' GPA', 'gpa_std']]:
    for c in cycles:
        y = [dftime[(dftime['month'] == m) & (dftime['cycle'] == c) &
                    (dftime['school_name'] == T11[0])].groupby(['school_name']).mean()[b[0]].values[0] for m in months]
        y = [round(elem, 1) for elem in y]

        if b[0] == 'acceptance_rate':
            showlegend = True
            meta = [b[1] + '<br>(n=%i' % dftime[(dftime['month'] == m) &
                                         (dftime['cycle'] == c) &
                                         (dftime['school_name'] == T11[0])]['na'] + ')' for m in months]
            eb = []
        else:
            showlegend = False
            meta = [' ' + b[1] + '<br>(n=%i' % dftime[(dftime['month'] == m) &
                                                      (dftime['cycle'] == c) &
                                                      (dftime['school_name'] == T11[0])]['n'] + ')' for m in months]
            eb = [dftime[(dftime['month'] == m) & (dftime['cycle'] == c) &
                         (dftime['school_name'] == T11[0])].groupby(['school_name']).mean()[b[2]].values[0]
                  for m in months]

        fig.add_trace(go.Bar(
            x=months,
            y=y,
            name=str(c-1) + '/' + str(c) + ' (n=%0.f' % df11[(df11['school_name'] == T11[0]) &
                                                             (df11['cycle'] == c)].shape[0] + ')',
            text=y,
            textposition='auto',
            meta=meta,
            hovertemplate='%{y}%{meta}<extra></extra>',
            error_y=dict(
                type='data',
                array=eb,
                color='darkgray',
                thickness=1.5
                ),
            legendgroup=str(c),
            showlegend=showlegend
            ),
            row=row,
            col=col
        )

    #  Advance to the next quadrant, clockwise
    if (row == 1) & (col == 2):
        row += 1

    if (row == 1) & (col == 1):
        col += 1

#  Add fourth plot
for c in cycles:
    y = [100*dftime[(dftime['month'] == m) & (dftime['cycle'] == c) &
                    (dftime['school_name'] == T11[0])]['n'].values[0] /
         dftime[(dftime['school_name'] == T11[0]) & (dftime['cycle'] == c)]['n'].sum() for m in months]
    y = [round(elem, 1) for elem in y]

    meta = ['%' + '<br>(n=%i' % dftime[(dftime['month'] == m) &
                                       (dftime['cycle'] == c) &
                                       (dftime['school_name'] == T11[0])]['n'] + ')' for m in months]

    fig.add_trace(go.Bar(
        x=months,
        y=y,
        name=str(c-1) + '/' + str(c) + ' (n=%0.f' % df11[(df11['school_name'] == T11[0]) &
                                                         (df11['cycle'] == c)].shape[0] + ')',
        text=y,
        textposition='auto',
        meta=meta,
        hovertemplate='%{y}%{meta}<extra></extra>',
        legendgroup=str(c),
        showlegend=showlegend
        ),
        row=2,
        col=1
    )

updatemenu = []
button_schools = []

#  Button with one option for each school
for i, school in enumerate(T11):
    y = []
    name = []
    eb = []
    meta = []

    #  For plots 1, 2, and 3
    for b in [['acceptance_rate', '%'], ['lsat_mean', ' LSAT', 'lsat_std'], ['gpa_mean', ' GPA', 'gpa_std']]:
        for c in cycles:
            y_temp = [round(dftime[(dftime['month'] == m) & (dftime['cycle'] == c) &
                                   (dftime['school_name'] == school)].groupby(['school_name']).mean()[b[0]].values[0], 1)
                      for m in months]
            y.append(y_temp)

            name.append(str(c-1) + '/' + str(c) + ' (n=%0.f' % df11[(df11['school_name'] == school) &
                                                                    (df11['cycle'] == c)].shape[0] + ')')
            if b[0] == 'acceptance_rate':
                meta.append([b[1] + '<br>(n=%i' % dftime[(dftime['month'] == m) &
                                         (dftime['cycle'] == c) &
                                         (dftime['school_name'] == T11[0])]['na'] + ')' for m in months])
                eb.append(
                    dict(
                        type='data',
                        array=[],
                        color='darkgray',
                        thickness=1.5
                        )
                    )
            else:
                meta.append([' ' + b[1] + '<br>(n=%i' % dftime[(dftime['month'] == m) &
                                                      (dftime['cycle'] == c) &
                                                      (dftime['school_name'] == school)]['n'] + ')' for m in months])
                eb.append(
                    dict(
                        type='data',
                        array=[dftime[(dftime['month'] == m) & (dftime['cycle'] == c) &
                                      (dftime['school_name'] == school)].groupby(['school_name']).mean()[b[2]].values[0]
                               for m in months],
                        color='darkgray',
                        thickness=1.5
                    )
                )

    #  For plot 4
    for c in cycles:
        y_temp = [round(100*dftime[(dftime['month'] == m) & (dftime['cycle'] == c) &
                                   (dftime['school_name'] == school)]['n'].values[0] /
                        dftime[(dftime['school_name'] == school) & (dftime['cycle'] == c)]['n'].sum(), 1)
                  for m in months]
        y.append(y_temp)

        meta.append(['%' + '<br>(n=%i' % dftime[(dftime['month'] == m) &
                                          (dftime['cycle'] == c) &
                                          (dftime['school_name'] == school)]['n'] + ')' for m in months])

        name.append(str(c-1) + '/' + str(c) + ' (n=%0.f' % df11[(df11['school_name'] == school) &
                                                                (df11['cycle'] == c)].shape[0] + ')')

    button_schools.append(
        dict(
            method='update',
            label=T11_short[i] + ' (n=%0.f' % df11[df11['school_name'] == school].shape[0] + ')',
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
    autosize=True,
    colorway=['violet', 'seagreen', 'coral', 'cornflowerblue'],
    margin=dict(l=75, r=100, autoexpand=True),
    height=700,
    title={
        'text': 'Application Timeline Correlations',
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
fig.update_xaxes(title_text='Month Application Sent')

# Update yaxis properties
fig.update_yaxes(title_text='Acceptance Rate', row=1, col=1)
fig.update_yaxes(title_text='LSAT (all apps.)', row=1, col=2)
fig.update_yaxes(title_text='GPA (all apps.)', row=2, col=2)
fig.update_yaxes(title_text='Pct. of All Apps. Sent', row=2, col=1)

# fig.show(config=dict(modeBarButtonsToRemove=['autoScale2d']))

fname_html = 'timelinebar.html'
cwd = Path(getcwd())
pio.write_html(fig,
               file=str(cwd.parent.absolute()) + '/docs/_includes/' + fname_html,
               auto_open=False,
               config=dict(modeBarButtonsToRemove=['autoScale2d']))
print('\nFinished writing to ' + fname_html + '.')
