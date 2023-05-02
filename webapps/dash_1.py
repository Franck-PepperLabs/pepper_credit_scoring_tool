# My first dash app

import dash
import dash_html_components as html
from dash.dependencies import Input, Output

# Create the Dash application
app = dash.Dash(__name__)

# Define the app layout
app.layout = html.Div([
    html.Iframe(
        id='dashboard',
        srcDoc=open('dashboard.html', 'r').read(),
        width='100%', height='600')
])

if __name__ == '__main__':
    # Run the Dash app
    app.run_server(debug=True)