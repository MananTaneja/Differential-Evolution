import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px

df = px.data.gapminder()
animations = {
    'Scatter': px.scatter(
        df, x="gdpPercap", y="lifeExp", animation_frame="year",
        animation_group="country", size="pop", color="continent",
        hover_name="country", log_x=True, size_max=55,
        range_x=[100, 100000], range_y=[25, 90]),
    'Bar': px.bar(
        df, x="continent", y="pop", color="continent",
        animation_frame="year", animation_group="country",
        range_y=[0, 4000000000]),
}

app = dash.Dash(__name__)

app.layout = html.Div([
    html.P("Select an animation:"),
    dcc.RadioItems(
        id='selection',
        options=[{'label': x, 'value': x} for x in animations],
        value='Scatter'
    ),
    dcc.Graph(id="graph"),
])


@app.callback(
    Output("graph", "figure"),
    [Input("selection", "value")])
def display_animated_graph(s):
    return animations[s]


app.run_server(debug=True, port=3001)


# Matplotlib vs plotly - comparitive framework study
