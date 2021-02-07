import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import numpy as np
import plotly.graph_objects as go


# using-a-slider-and-buttons
# Follow this link - https://plotly.com/python/animations/

dataframe = pd.read_csv(
    'https://gist.githubusercontent.com/chriddyp/c78bf172206ce24f77d6363a2d754b59/raw/c353e8ef842413cae56ae3920b8fd78468aa4cb2/usa-agricultural-exports-2011.csv')


def generate_table(dataframe, max_rows=10):
    return html.Table([
        html.Thead(
            html.Tr([html.Th(col) for col in dataframe.columns])
        ),
        html.Tbody([
            html.Tr([
                html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
            ]) for i in range(min(len(dataframe), max_rows))
        ])
    ])


population = []

# Differential Evolution


def de(fobj, bounds, mut=0.8, crossp=0.7, popsize=20, its=1000, flag=True):
    dimensions = len(bounds)
    # pop = []
    # if (flag == True):
    #     pop = np.random.rand(popsize, dimensions)
    # else:
    #     pop = flag
    pop = np.random.rand(popsize, dimensions)
    min_b, max_b = np.asarray(bounds).T
    diff = np.fabs(min_b - max_b)
    pop_denorm = min_b + pop * diff
    fitness = np.asarray([fobj(ind) for ind in pop_denorm])
    best_idx = np.argmin(fitness)
    best = pop_denorm[best_idx]
    for i in range(its):
        for j in range(popsize):
            idxs = [idx for idx in range(popsize) if idx != j]
            a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
            mutant = np.clip(a + mut * (b - c), 0, 1)
            cross_points = np.random.rand(dimensions) < crossp
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dimensions)] = True
            trial = np.where(cross_points, mutant, pop[j])
            trial_denorm = min_b + trial * diff
            f = fobj(trial_denorm)
            if f < fitness[j]:
                fitness[j] = f
                pop[j] = trial
                if f < fitness[best_idx]:
                    best_idx = j
                    best = trial_denorm
        yield best, fitness[best_idx]


external_stylesheets = [
    'https://codepen.io/chriddyp/pen/bWLwgP.css', [dbc.themes.BOOTSTRAP]]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

list1 = [0, 100, 200, 300]

controls = dbc.Card([
    html.H3('Control Panel', className='text-center',
            id='control-panel-heading'),
    dbc.FormGroup([
        dbc.Label('Choose the dimensions - on the right legend'),

    ], className='my-3'),
    html.Hr(),
    dbc.FormGroup([
        dbc.Label('Link to the database'),
        dbc.Input(id='database-url', type='text', value='', disabled=True)
    ], className='my-3'),
    html.Hr(),
    dbc.FormGroup([
        dbc.Label('Slider for iterations'),
        dcc.Slider(
            id='iterations-slider',
            min=0,
            max=300,
            value=0,
            marks={str(it): str(it) for it in list1},
            step=None
        ),
    ]),
    html.Hr(),
    dbc.FormGroup([
        dbc.Label('Mutation Value: (Between 0 and 1)'),
        dbc.Input(id='mutation', type='number',
                  value=0.8, step=0.01, min=0.1, max=0.99)
    ]),
    html.Hr(),
    dbc.FormGroup([
        dbc.Label('Crossover Value: (Between 0 and 1)'),
        dbc.Input(id='crossover', type='number',
                  value=0.7, step=0.01, min=0.01, max=0.99)
    ]),
    html.Hr(),
    dbc.FormGroup([
        dbc.Label('Population Size:'),
        dbc.Input(id='popsize', type='number', value=20)
    ]),
    html.Hr(),
    dbc.Button(
        'Download Video Animation',
        color='danger',
        block=True,
        id='button'
    )

], body=True)

navbar = dbc.NavbarSimple(
    children=[
        dbc.NavItem(dbc.NavLink("Cosine Function", href="#")),
        # dbc.DropdownMenu(
        #     children=[
        #         dbc.DropdownMenuItem("More pages", header=True),
        #         dbc.DropdownMenuItem("Page 2", href="#"),
        #         dbc.DropdownMenuItem("Page 3", href="#"),
        #     ],
        #     nav=True,
        #     in_navbar=True,
        #     label="More",
        # ),
    ],
    brand="Home",
    brand_href="#",
)

# HTML Structure
app.layout = dbc.Container(children=[
    navbar,
    html.H1('Differential Evolution Dashboard',
            className="text-center h1"),
    html.Hr(),

    dbc.Row([
        dbc.Col(controls, md=4),
        dbc.Col(dcc.Graph(id='example-graph', figure={}), md=8)
    ], align="center"),



    dbc.Row([
        html.H2('Visualize Uploaded Data', id='visualize-heading',
                className='text-center'),
        dbc.Container(
            generate_table(dataframe=dataframe),
        ),
    ],
        className="mt-4 pt-4"
    )
])


@ app.callback(
    Output("example-graph", "figure"),
    [
        Input("iterations-slider", "value"),
        Input('mutation', 'value'),
        Input('crossover', 'value'),
        Input('popsize', 'value')
    ]
)
def update_graph(n_iter, mut, cross, psize):
    fig = go.Figure()
    if n_iter == 0:
        return fig

    dimensions = [8, 16, 32, 64]

    for d in dimensions:
        it = list(de(lambda x: sum(
            x ** 2) / d, [(-100, 100)] * d, mut=mut, crossp=cross, popsize=psize, its=n_iter))
        x, f = zip(*it)
        fig.add_trace(go.Scatter(y=f, mode='lines',
                                 name="Dimensions: {}".format(d)))
    return fig


def upload_data(url):
    dataframe = pd.read_csv(url)


if __name__ == '__main__':
    app.run_server(debug=True)