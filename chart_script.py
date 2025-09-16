import plotly.graph_objects as go
import plotly.io as pio

# Data with shortened model names to fit 15 character limit
models = ["Logistic Reg", "Random Forest", "Gradient Boost"]
accuracies = [96.5, 99.5, 99.5]

# Colors from the brand palette
colors = ['#1FB8CD', '#DB4545', '#2E8B57']

# Create horizontal bar chart
fig = go.Figure(go.Bar(
    x=accuracies,
    y=models,
    orientation='h',
    marker_color=colors,
    text=[f'{acc}%' for acc in accuracies],
    textposition='inside'
))

# Update layout
fig.update_traces(cliponaxis=False)

fig.update_layout(
    title="ML Model Performance Comparison",
    xaxis_title="Accuracy (%)",
    yaxis_title="Models"
)

# Save as PNG and SVG
fig.write_image("chart.png")
fig.write_image("chart.svg", format="svg")