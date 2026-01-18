import pandas as pd
import plotly.graph_objects as go

def generate_interactive_forecast(history_df: pd.DataFrame, forecast_df: pd.DataFrame, drug_name: str) -> str:
    """
    Generates an interactive Plotly chart showing historical price and a forecast.

    Args:
        history_df (pd.DataFrame): DataFrame with historical 'date' and 'price'.
        forecast_df (pd.DataFrame): DataFrame with forecasted 'date', 'price', 
                                    'lower_bound', and 'upper_bound'.
        drug_name (str): The name of the drug for the chart title.

    Returns:
        str: An HTML string representing the Plotly figure, ready for embedding.
    """
    fig = go.Figure()

    # Add the confidence interval band for the forecast
    fig.add_trace(go.Scatter(
        x=pd.concat([forecast_df['date'], forecast_df['date'][::-1]]),
        y=pd.concat([forecast_df['upper_bound'], forecast_df['lower_bound'][::-1]]),
        fill='toself',
        fillcolor='rgba(255, 165, 0, 0.2)', # Light orange fill
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        showlegend=False,
        name='Confidence Interval'
    ))
    
    # Add the historical data
    fig.add_trace(go.Scatter(
        x=history_df['date'], 
        y=history_df['price'],
        mode='lines',
        name='Historical Price',
        line=dict(color='blue')
    ))

    # Add the forecast line
    fig.add_trace(go.Scatter(
        x=forecast_df['date'], 
        y=forecast_df['price'],
        mode='lines',
        name='Forecasted Price',
        line=dict(color='orange', dash='dash')
    ))

    # Update layout for a professional look
    fig.update_layout(
        title=f'Price History & 4-Week Forecast for: <b>{drug_name}</b>',
        xaxis_title='Date',
        yaxis_title='Price per Unit ($)',
        template='plotly_white',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=40, r=40, t=80, b=40)
    )

    # Return as an HTML string for embedding
    return fig.to_html(full_html=False, include_plotlyjs='cdn', default_height='500px')

if __name__ == '__main__':
    # --- Example Usage ---
    # Create sample dataframes to demonstrate the function
    history = pd.DataFrame({
        'date': pd.to_datetime(['2023-01-01', '2023-01-08', '2023-01-15']),
        'price': [100, 102, 101]
    })
    
    forecast = pd.DataFrame({
        'date': pd.to_datetime(['2023-01-22', '2023-01-29', '2023-02-05']),
        'price': [105, 108, 110],
        'lower_bound': [100, 102, 104],
        'upper_bound': [110, 114, 116]
    })
    
    chart_html = generate_interactive_forecast(history, forecast, "Test-Drug 500mg")
    
    # Save to a file to preview
    with open("interactive_plot_preview.html", "w") as f:
        f.write("<html><head><title>Chart Preview</title></head><body>")
        f.write(chart_html)
        f.write("</body></html>")
        
    print("Generated 'interactive_plot_preview.html' for visual inspection.")