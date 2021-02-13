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


population_data = []

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
        population_data.append(pop)
        yield best, fitness[best_idx]


def de2(fobj, bounds, mut=0.8, crossp=0.7, popsize=20, its=1000):
    dimensions = len(bounds)
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
        yield min_b + pop * diff, fitness, best_idx


def rmse(w):
    y_pred = fmodel(x, w)
    return np.sqrt(sum((y - y_pred) ** 2) / len(y))


def fmodel(x, w):
    return w[0] + w[1]*x + w[2] * x**2 + w[3] * x**3 + w[4] * x**4 + w[5] * x**5


external_stylesheets = [
    'https://codepen.io/chriddyp/pen/bWLwgP.css', [dbc.themes.BOOTSTRAP]]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets,
                suppress_callback_exceptions=True)

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
        'Save the Progress!',
        color='danger',
        block=True,
        id='save_progress_button',
        n_clicks=0
    )

], body=True)

navbar = dbc.NavbarSimple(
    children=[
        dbc.NavItem(dbc.NavLink(dcc.Link('Cosine Function',
                                         href='/cosine', className="mx-4 text-decoration-none text-reset"))),
        dbc.NavItem(dbc.NavLink(dcc.Link('Visualize Uploaded Data',
                                         href='/data', className="mx-4 text-decoration-none text-reset"))),
        dbc.NavItem(dbc.NavLink(dcc.Link('Previous Runs',
                                         href='/prev', className="mx-4 text-decoration-none text-reset"))),
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
    brand_href="/"
)


# HTML Structure - Main Page
app.layout = dbc.Container(children=[
    navbar,
    dcc.Location(id='url', refresh=False),
    dbc.Container(id='page-content')
])


@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/cosine':
        return cosine_page_layout
    elif pathname == '/data':
        return visualize_data_layout
    elif pathname == '/prev':
        return previous_runs_layout
    else:
        return mathematical_function


mathematical_function = dbc.Container([
    html.H1('Differential Evolution Simulation',
            className="text-center h1"),
    html.Hr(),

    dbc.Row([
        dbc.Col(controls, md=4),
        dbc.Col(dcc.Graph(id='example-graph', figure={}), md=8)
    ], align="center")
])


@ app.callback(
    Output("example-graph", "figure"),
    [
        Input("iterations-slider", "value"),
        Input('mutation', 'value'),
        Input('crossover', 'value'),
        Input('popsize', 'value'),
        Input('save_progress_button', 'n_clicks')
    ]
)
def update_graph(n_iter, mut, cross, psize, save_progress_button):
    fig = go.Figure()
    if n_iter == 0:
        return fig

    dimensions = [8, 16, 32, 64]

    store = False
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'save_progress_button' in changed_id:
        store = True

    for d in dimensions:
        it = list(de(lambda x: sum(
            x ** 2) / d, [(-100, 100)] * d, mut=mut, crossp=cross, popsize=psize, its=n_iter))
        x, f = zip(*it)
        fig.add_trace(go.Scatter(y=f, mode='lines',
                                 name="Dimensions: {}".format(d)))

        if(store == True):
            print('Dimension ' + str(d) + ' : ' + str(f) + '\n')
            print('Population Data: \n')
            for pop in population_data:
                print(str(pop))

    return fig


cosine_page_layout = dbc.Container([
    html.H1('Cosine Function', className="text-center h1"),
    html.Hr(),

    dbc.Row([
        html.Video(
            src='https://pablormier.github.io/assets/img/de/curve-fitting.mp4#center',
            autoPlay='autoPlay',
            # controls='controls'
        )], align="center",)
])


# @app.callback(
#     Output('cosine-graph', 'figure')
#     [
#         Input("iterations-slider", "value"),
#         Input('mutation', 'value'),
#         Input('crossover', 'value'),
#         Input('popsize', 'value')
#     ]
# )
# def cosine_graph(n_iter, mut, cross, psize):
#     fig = go.Figure()
#     if (n_iter == 0):
#         return fig

#     result = list(de2(rmse, [(-5, 5) * 6], its=2000))


visualize_data_layout = dbc.Container([
    html.H1('User Uploaded Data',
            className="text-center h1"),

    html.Hr(),
    dbc.Input(id="database_url", placeholder="Link to data", type='text'),
    dbc.Row([
        dbc.Container(id="table"
                      # generate_table(dataframe=dataframe),
                      ),
    ],
        className="mt-4 pt-4"
    )
])


@app.callback(
    Output('table', 'children'),
    [Input('database_url', 'value')])
def generate_table(value, max_rows=100):
    dataframe = pd.read_csv(value)

    if (dataframe.empty):
        return 'Invalid'

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


previous_runs_layout = dbc.Container([
    html.H1('Check Previous Runs', className='text-center'),
    html.Hr(),
    dbc.DropdownMenu(
        label="Previous Runs",
        children=[
            dbc.DropdownMenuItem('Run at 12:00', id='run1'),
            dbc.DropdownMenuItem('Run at 13:30', id='run2'),
            dbc.DropdownMenuItem('Run at 16:00', id='run3')
        ]
    ),
    dbc.Container(id='output_run', className='mt-4 pt-4')
])


@app.callback(
    Output('output_run', 'children'),
    [
        Input('run1', 'n_clicks'),
        Input('run2', 'n_clicks'),
        Input('run3', 'n_clicks'),
    ])
def update_run(run1, run2, run3):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'run1' in changed_id:
        msg = 'Displaying the saved progress of run 1'
    elif 'run2' in changed_id:
        msg = 'Displaying the saved progress of run 2'
    elif 'run3' in changed_id:
        msg = 'Displaying the saved progress of run 3'
    else:
        msg = 'No Session Selected'
        return html.H3(msg)

    return dbc.Container([
        html.H3(msg),
        dcc.Graph(id='prev_session_graph', figure={})
    ])


if __name__ == '__main__':
    app.run_server(debug=True)
