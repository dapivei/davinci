import plotly.graph_objects as go
import pandas as pd
from datetime import date

def get_date(pandas_date):
    year, day, month = pandas_date
    return date(year, month, day)
    
def n_weeks(time1,time2):
    if(type(time1) == str):
        time1 = get_date(time1)
    if(type(time2) == str):
        time2 = get_date(time2)
    result = abs((time1-time2).days//7)
    time2 = time1 + pd.Timedelta(days= 7*result)
    return time2
def remove_day(pandas_date):
    year = pandas_date.year
    month = pandas_date.month
    return date(year=year, month=month, day = 1)


def plot_topic_across_time(df, aggregation_level = 'day', speaker = None, title = 'Topics over Time'):
    colors = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#D55E00", "#0072B2", "#CC79A7"]
    df = df.copy()
    df = df[df.Topic != -1]
    if(speaker):
        df = df[df.speaker == speaker]
    if(aggregation_level == 'day'):
        df = df.groupby(['Topic','CustomName', 'timestamp']).size().reset_index(name='counts')
        key = 'timestamp'
    elif(aggregation_level == 'week'):
        
        min_date = df['timestamp'].min()
        df['n_weeks'] = df.apply(lambda x: n_weeks(min_date,x["timestamp"]),axis=1)
        df = df.groupby(['Topic','CustomName', 'n_weeks']).size().reset_index(name='counts')
        key = 'n_weeks'
    else:
        #Aggregation by month
        df["year_month"] = df.timestamp.apply(remove_day)
        df = df.groupby(['Topic','CustomName', 'year_month']).size().reset_index(name='counts')
        key = 'year_month'
        
        
    fig = go.Figure()
    for index, topic in enumerate(df.Topic.unique()):
        trace_data = df.loc[df.Topic == topic, :]
        topic_name = trace_data.CustomName.values[0]
        

        y = trace_data.counts
        fig.add_trace(go.Scatter(x=trace_data[key], y=y,
                                 mode='lines',
                                 marker_color=colors[index % 7],
                                 hoverinfo="text",
                                 name=topic_name,
                                 ))
    # Styling of the visualization
    fig.update_xaxes(showgrid=True)
    fig.update_yaxes(showgrid=True)
    fig.update_layout(
        
        yaxis_title="Frequency",
        title={
            'text': f"<b>{title}</b>",
            'y': .95,
            'x': 0.40,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(
                size=22,
                color="Black")
        },
        template="simple_white",
        width=1250,
        height=450,
        hoverlabel=dict(
            bgcolor="white",
            font_size=16,
            font_family="Rockwell"
        ),
        legend=dict(
            title="<b>Global Topic Representation",
        )
    )
    return fig