from os import getcwd
from pathlib import Path
import datetime as dt
import matplotlib
import matplotlib.colors
import matplotlib.cm as cm
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.colors import n_colors
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

current_of = max(df11[df11['cycle'] == 21]['decision_at'])

#  Calculate regular and splitter acceptance rates
dfpct = pd.DataFrame(columns=['school_name', 'cycle', 'Regular', 'Splitters', 'Rev. Splitters', 'rn', 'sn', 'rsn'])

for school in T11:
    for c in cycles + ['all']:
        if c == 'all':
            df_temp = df11[df11['school_name'] == school]
        else:
            df_temp = df11[(df11['school_name'] == school) & (df11['cycle'] == c)]

        reg = 100*df_temp[(df_temp['decision'] == 'A') &
                          (df_temp['splitter'] != 'blue') &
                          (df_temp['splitter'] != 'black')].shape[0] / \
              df_temp[(df_temp['decision'].str.contains('|'.join(['A', 'R', 'WL']))) &
                      (df_temp['splitter'] != 'blue') &
                      (df_temp['splitter'] != 'black')].shape[0]

        rn = df_temp[(df_temp['decision'] == 'A') &
                     (df_temp['splitter'] != 'blue') &
                     (df_temp['splitter'] != 'black')].shape[0]

        split = 100*df_temp[(df_temp['decision'] == 'A') &
                            (df_temp['splitter'] == 'blue') &
                            (df_temp['splitter'] != 'black')].shape[0] / \
                df_temp[(df_temp['decision'].str.contains('|'.join(['A', 'R', 'WL']))) &
                        (df_temp['splitter'] == 'blue') &
                        (df_temp['splitter'] != 'black')].shape[0]

        sn = df_temp[(df_temp['decision'] == 'A') &
                     (df_temp['splitter'] == 'blue') &
                     (df_temp['splitter'] != 'black')].shape[0]

        rsplit = 100*df_temp[(df_temp['decision'] == 'A') &
                             (df_temp['splitter'] != 'blue') &
                             (df_temp['splitter'] == 'black')].shape[0] / \
                 df_temp[(df_temp['decision'].str.contains('|'.join(['A', 'R', 'WL']))) &
                         (df_temp['splitter'] != 'blue') &
                         (df_temp['splitter'] == 'black')].shape[0]

        rsn = df_temp[(df_temp['decision'] == 'A') &
                      (df_temp['splitter'] != 'blue') &
                      (df_temp['splitter'] == 'black')].shape[0]

        index = len(dfpct)
        dfpct.loc[index] = [school, c, reg, split, rsplit, rn, sn, rsn ]

#  Set initial traces
# fig = go.Figure()

fig = make_subplots(
    rows=2, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.15,
    specs=[[{'type': 'scatter'}],
           [{'type': 'table'}]]
)

for s in [['Regular', 'lime', 'rn'], ['Splitters', 'dodgerblue', 'sn'], ['Rev. Splitters', 'black', 'rsn']]:
    fig.add_trace(
        go.Scatter(
            x=cycles,
            y=dfpct[dfpct['school_name'] == T11[0]][s[0]],
            mode='markers',
            name=s[0],
            meta=[dfpct[(dfpct['school_name'] == T11[0]) & (dfpct['cycle'] == c)][s[2]] for c in cycles],
            hovertemplate='%{y:.2f}%<br>(n=%{meta})<extra></extra>',
            marker=dict(
                size=10,
                color=[s[1]]*4,
                )
            ),
        row=1,
        col=1
    )

index_s = [round(x, 2) for x in dfpct['Splitters'].values/dfpct['Regular'].values]
index_r = [round(x, 2) for x in dfpct['Rev. Splitters'].values/dfpct['Regular'].values]

colors_pos = n_colors('rgb(200, 255, 200)', 'rgb(0, 200, 0)', 9, colortype='rgb')
colors_neg = n_colors('rgb(255, 200, 200)', 'rgb(200, 0, 0)', 9, colortype='rgb')

norm_s = matplotlib.colors.Normalize(vmin=min(index_s)*0.9, vmax=max(index_s)*1.1, clip=True)
m_s_pos = cm.ScalarMappable(norm=norm_s, cmap=cm.Greens)
m_s_neg = cm.ScalarMappable(norm=norm_s, cmap=cm.Reds_r)

norm_r = matplotlib.colors.Normalize(vmin=min(index_r)*0.9, vmax=max(index_r)*1.1, clip=True)
m_r_pos = cm.ScalarMappable(norm=norm_r, cmap=cm.Greens)
m_r_neg = cm.ScalarMappable(norm=norm_r, cmap=cm.Reds_r)

index_s_c = []
for idx in index_s:
    if idx < 1:
        index_s_c.append(matplotlib.colors.rgb2hex(m_s_neg.to_rgba(idx)))
    else:
        index_s_c.append(matplotlib.colors.rgb2hex(m_s_pos.to_rgba(idx)))

index_r_c = []
for idx in index_r:
    if idx < 1:
        index_r_c.append(matplotlib.colors.rgb2hex(m_r_neg.to_rgba(idx)))
    else:
        index_r_c.append(matplotlib.colors.rgb2hex(m_r_pos.to_rgba(idx)))

alternating_color = ['#f0f0f0', '#e6e6e6', '#e2e2e2', '#dcdcdc', '#d2d2d2']*len(T11)
fill_color = [
    np.concatenate([['white']*5 + ['lightgrey']*5]*5 + [['white']*5]).ravel().tolist(),  # School
    alternating_color,  # Cycle
    alternating_color,  # Regular %
    alternating_color,  # Regular n=
    alternating_color,  # Splitters %
    alternating_color,  # Splitters n=
    alternating_color,  # Rev. Splitters %
    alternating_color,  # Rev. Splitters n=
    index_s_c,  # Splitter index
    index_r_c  # Rev. Splitter index
]

#  Add table
fig.add_trace(
    go.Table(
        columnwidth=[80] + [45] + [100]*8,
        header=dict(
            values=['School', 'Cycle',
                    'Regular (%)', 'Regular (n=)',
                    'Splitters (%)', 'Splitters (n=)',
                    'Rev. Splitters (%)', 'Rev. Splitters (n=)',
                    'Splitter Index',
                    'Rev. Splitter Index'],
            font=dict(size=11),
            align=['left']*2 + ['center']*8
        ),
        cells=dict(
            values=[np.array([[name] + ['']*4 for name in T11_short]).flatten(),  # School
                    (cycles + ['All'])*len(T11),  # Cycle
                    [int(x) for x in dfpct['Regular'].values],  # Regular %
                    dfpct['rn'].values,  # Regular n=
                    [int(x) for x in dfpct['Splitters'].values],  # Splitters %
                    dfpct['sn'].values,  # Splitters n=
                    [int(x) for x in dfpct['Rev. Splitters'].values],  # Rev. Splitters %
                    dfpct['rsn'].values,  # Rev. Splitters n=
                    index_s,  # Splitters index
                    index_r],  # Rev. Splitters index
            fill_color=fill_color,
            line_color=fill_color,
            align=['left']*2 + ['center']*8,
            font=dict(
                color=['black']*8 + ['lightgrey']*2,
                size=11
                # family=[['Arial']*10, ['Arial']*10, ['Arial']*10, ['Arial']*10, ['Arial Black']*10]*len(T11)
                )
        )
    ),
    row=2, col=1)

updatemenu = []
button_schools = []

#  Button with one option for each school
for i, school in enumerate(T11):
    y = []
    name = []
    marker = []
    meta = []

    for s in [['Regular', 'lime', 'rn'], ['Splitters', 'dodgerblue', 'sn'], ['Rev. Splitters', 'black', 'rsn']]:
        y.append(dfpct[(dfpct['school_name'] == school)][s[0]])
        name.append(s[0])
        marker.append(
            dict(
                size=10,
                color=[s[1]] * 4
                )
            )
        meta.append([dfpct[(dfpct['school_name'] == school) & (dfpct['cycle'] == c)][s[2]] for c in cycles])

    button_schools.append(
        dict(
            method='update',
            label=T11_short[i],
            visible=True,
            args=[
                dict(
                    y=y,
                    name=name,
                    marker=marker,
                    meta=meta
                )
            ],
        )
    )

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
    height=700,
    title={
        'text': 'Chance of Acceptance by Stats. Type',
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

fig.update_xaxes(
    title_text='Cycle',
    tickmode='array',
    tickvals=cycles,
    range=[min(cycles)-0.25, max(cycles)+0.25]
)

fig.update_yaxes(title_text='Percentage')

# fig.show(config=dict(modeBarButtonsToRemove=['autoScale2d']))

cwd = Path(getcwd())
pio.write_html(fig, file=str(cwd.parent.absolute()) + '/docs/_includes/splitters.html', auto_open=False, config=dict(modeBarButtonsToRemove=['autoScale2d']))
print('\nFinished writing to splitters.html.')
