import dash
import dash_html_components as html
import pandas as pd
import dash_core_components as dcc
from dash.dependencies import Input, Output
import plotly.express as px
import numpy as np
import plotly.graph_objects as go


# Differential Evolution
def de(fobj, bounds, mut=0.8, crossp=0.7, popsize=20, its=1000):
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
        yield best, fitness[best_idx]


fig = go.Figure()

dimensions = [8, 16, 32, 64]

for d in dimensions:
    it = list(de(lambda x: sum(x**2)/d, [(-100, 100)] * d, its=300))
    x, f = zip(*it)
    fig.add_trace(go.Scatter(y=f, mode='lines', showlegend=False))


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

colors = {
    'background': '#dfe6e9',
    'text': '#2d3436'
}

# Rendering the data - connects the backend flask with react
app.layout = html.Div(style={'backgroundColor': colors['background']}, children=[
    html.H1('Hello Dash', className="text-center h1 text-dark"),

    html.Div(children='Dash: A web application framework for python.',
             className="text-center h4 text-muted"),

    dcc.Graph(
        id='example-graph',
        figure=fig,
        style={'width': '50%'}
    )
])
if __name__ == "__main__":
    app.run_server(debug=True)
