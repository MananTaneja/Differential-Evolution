import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import pandas as pd

ext_stylesheets = [
    'https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=ext_stylesheets)

# Can add inline styles here
colors = {
    'background': '#dfe6e9',
    'text': '#2d3436'
}

# Load dataframe
df = pd.DataFrame({
    "Fruit": ["Apples", "Oranges", "Bananas", "Apples", "Oranges", "Bananas"],
    "Amount": [4, 1, 2, 2, 4, 5],
    "City": ["SF", "SF", "SF", "Montreal", "Montreal", "Montreal"]
})

# Make Visualizations with data
fig = px.bar(df, x="Fruit", y="Amount", color="City", barmode="group")

fig.update_layout(
    plot_bgcolor=colors['background'],
    paper_bgcolor=colors['background'],
    font_color=colors['text']
)

# Rendering the data - connects the backend flask with react
app.layout = html.Div(style={'backgroundColor': colors['background']}, children=[
    html.H1('Hello Dash', className="text-center h1 text-dark"),

    html.Div(children='Dash: A web application framework for python.',
             className="text-center h4 text-muted"),

    dcc.Graph(
        id='example-graph',
        figure=fig
    )
])

if __name__ == "__main__":
    app.run_server(debug=True)
