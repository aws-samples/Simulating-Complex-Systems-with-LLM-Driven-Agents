
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
######################################################################
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. #
# SPDX-License-Identifier: MIT-0                                #
######################################################################


import streamlit as st
import networkx as nx
import plotly.graph_objects as go
import plotly.subplots as sp
import pickle
from collections import defaultdict
import numpy as np
import pandas as pd


# Streamlit app
def main():
    st.set_page_config(layout="wide")

    st.title("Energy Market Supply Chain Network Simulation")

    # Create a sidebar for the refresh button
    with st.sidebar:
        st.title("Controls")
        if st.button("Refresh Data"):
            st.session_state.refresh = True
        elif 'refresh' not in st.session_state:
            st.session_state.refresh = True

    # Main content area
    if 'refresh' not in st.session_state or st.session_state.refresh:
        all_results, all_subsidies, company_data, demand_history, unmet_demand_history = load_simulation_data()
        if all_results and all_subsidies and company_data and demand_history and unmet_demand_history:
            st.session_state.all_results = all_results
            st.session_state.all_subsidies = all_subsidies
            st.session_state.company_data = company_data
            st.session_state.demand_history = demand_history
            st.session_state.unmet_demand_history = unmet_demand_history
            st.session_state.refresh = False
    else:
        all_results = st.session_state.get('all_results')
        all_subsidies = st.session_state.get('all_subsidies')
        company_data = st.session_state.get('company_data')
        demand_history = st.session_state.get('demand_history')
        unmet_demand_history = st.session_state.get('unmet_demand_history')

    if all(v is not None for v in [all_results, all_subsidies, company_data, demand_history, unmet_demand_history]):
        max_step = len(all_results) - 1

        # Add a slider for step selection
        step = st.slider("Select simulation step", 0, max_step, 0)

        # Create the network visualization
        network_fig = visualize_network(company_data['company_ids'],
                                        company_data['company_types'],
                                        company_data['connections'])

        # Display the network chart
        st.plotly_chart(network_fig)

        # Add the company information table
        st.subheader("Company Information")
        company_df = pd.DataFrame({
            'Company ID': company_data['company_ids'],
            'Company Type': [company_data['company_types'][id] for id in company_data['company_ids']]
        })
        st.table(company_df)

        # Create and display the results chart
        #results_fig = visualize_results(all_results, all_subsidies, step, demand_history)
        #st.plotly_chart(results_fig, use_container_width=True)

        results_fig, individual_resources_fig = visualize_results(all_results, all_subsidies, step, demand_history)
        st.plotly_chart(results_fig, use_container_width=True)
        st.plotly_chart(individual_resources_fig, use_container_width=True)

        # Create and display the demand chart
        demand_fig = visualize_demand(demand_history, unmet_demand_history, step)
        st.plotly_chart(demand_fig, use_container_width=True)


    else:
        st.error("Failed to load simulation data. Please check your data files and try refreshing.")

def load_simulation_data():
    all_results, all_subsidies, company_data, demand_history = None, None, None, None

    try:
        with open('all_results.pkl', 'rb') as f:
            all_results = pickle.load(f)

        with open('all_subsidies.pkl', 'rb') as f:
            all_subsidies = pickle.load(f)

        with open('company_data.pkl', 'rb') as f:
            company_data = pickle.load(f)

        with open('demand_history.pkl', 'rb') as f:
            demand_history = pickle.load(f)

        with open('unmet_demand_history.pkl', 'rb') as f:
            unmet_demand_history = pickle.load(f)

    except FileNotFoundError as e:
        st.error(f"Error: {e}. Make sure the pickle files are in the correct directory.")
    except Exception as e:
        st.error(f"An error occurred while loading the data: {e}")

    return all_results, all_subsidies, company_data, demand_history, unmet_demand_history

def visualize_network(company_ids, company_types, connections):
    G = nx.Graph()

    # Add nodes
    for company_id, company_type in company_types.items():
        G.add_node(company_id, type=company_type)

    # Add edges
    for company_id, company_connections in zip(company_ids, connections):
        for connection in company_connections:
            G.add_edge(company_id, connection)

    # Set up colors for different company types
    color_map = {
        "OilAndGas": "red",
        "RenewableEnergy": "green",
        "NuclearPower": "yellow",
        "CoalMining": "black",
        "UtilityProvider": "blue"
    }

    # Create a list of colors for nodes
    node_colors = [color_map[G.nodes[node]['type']] for node in G.nodes()]

    # Set up the layout
    pos = nx.spring_layout(G)

    # Create edge trace
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    # Create node trace
    node_x = [pos[node][0] for node in G.nodes()]
    node_y = [pos[node][1] for node in G.nodes()]

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        marker=dict(
            showscale=False,
            color=node_colors,
            size=10,
        ),
        text=[f"{node}<br>{G.nodes[node]['type']}" for node in G.nodes()],
        textposition="top center"
    )

    # Create the figure
    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title='Energy Market Supply Chain Network',
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=0, l=0, r=0, t=40),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )

    return fig

def visualize_results(all_results, all_subsidies, step, demand_history):
    # Process data
    resources_by_type = defaultdict(lambda: defaultdict(list))
    production_by_type = defaultdict(lambda: defaultdict(list))
    cost_to_consumer = defaultdict(list)
    subsidies_over_time = defaultdict(list)

    for s, (results, subsidies) in enumerate(zip(all_results, all_subsidies)):
        for company in results:
            company_type = company['company_type']
            resources_by_type[company_type][s].append(company['resources'])
            production_by_type[company_type][s].append(company['production'])
            if company_type == "UtilityProvider":
                cost_to_consumer[s].append(company['cost_to_consumer'])

        for company_type, subsidy in subsidies.items():
            subsidies_over_time[company_type].append(subsidy)

    # Calculate averages
    avg_resources = {ctype: [sum(step_data)/len(step_data) for step_data in cdata.values()]
                     for ctype, cdata in resources_by_type.items()}
    avg_production = {ctype: [sum(step_data)/len(step_data) for step_data in cdata.values()]
                      for ctype, cdata in production_by_type.items()}
    #avg_cost_to_consumer = [sum(costs)/len(costs) for costs in cost_to_consumer.values()]

    # Calculate average cost to consumer divided by demand
    avg_cost_to_consumer = []
    for s, costs in cost_to_consumer.items():
        if s < len(demand_history) and demand_history[s] > 0:
            avg_cost = sum(costs) / len(costs)
            avg_cost_per_unit = avg_cost / demand_history[s]
            avg_cost_to_consumer.append(avg_cost_per_unit)
        else:
            avg_cost_to_consumer.append(0)  # or some default value

    # Create subplots with more space between them
    fig = sp.make_subplots(rows=2, cols=2, subplot_titles=(
        f'Average Resources by Company Type',
        f'Average Production by Company Type',
        f'Subsidies by Company Type',
        f'Average Cost to Consumer'
    ), vertical_spacing=0.2, horizontal_spacing=0.15)

    # Resources plot
    for company_type, data in avg_resources.items():
        fig.add_trace(go.Scatter(x=list(range(len(data))), y=data, name=company_type, mode='lines', legendgroup="resources", showlegend=True), row=1, col=1)

    # Production plot
    for company_type, data in avg_production.items():
        fig.add_trace(go.Scatter(x=list(range(len(data))), y=data, name=company_type, mode='lines', legendgroup="production", showlegend=True), row=1, col=2)

    # Subsidies plot
    for company_type, data in subsidies_over_time.items():
        fig.add_trace(go.Scatter(x=list(range(len(data))), y=data, name=company_type, mode='lines', legendgroup="subsidies", showlegend=True), row=2, col=1)

    # Cost to Consumer plot
    fig.add_trace(go.Scatter(x=list(range(len(avg_cost_to_consumer))), y=avg_cost_to_consumer, name='Average Cost to Consumer', mode='lines', legendgroup="cost", showlegend=True), row=2, col=2)

    # Update layout
    fig.update_layout(
        height=900,
        width=1200,
        title_text="Energy Market Simulation Results",
        title_x=0.5,
        margin=dict(l=50, r=50, t=100, b=50),
    )

    # Update x-axes
    fig.update_xaxes(title_text="Simulation Step", row=2, col=1)
    fig.update_xaxes(title_text="Simulation Step", row=2, col=2)

    # Update y-axes
    fig.update_yaxes(title_text="Resources", row=1, col=1)
    fig.update_yaxes(title_text="Production", row=1, col=2)
    fig.update_yaxes(title_text="Subsidy Amount", row=2, col=1)
    fig.update_yaxes(title_text="Cost ($/KWh)", row=2, col=2)

    # Add individual legends to each subplot
    fig.update_layout(
        legend1=dict(title="Resources", yanchor="top", y=0.95, xanchor="left", x=1.02, font=dict(size=10)),
        legend2=dict(title="Production", yanchor="top", y=0.95, xanchor="left", x=1.02, font=dict(size=10)),
        legend3=dict(title="Subsidies", yanchor="top", y=0.95, xanchor="left", x=1.02, font=dict(size=10)),
        legend4=dict(title="Cost", yanchor="top", y=0.95, xanchor="left", x=1.02, font=dict(size=10)),
    )

    # Update each trace to use a specific legend
    for i, trace in enumerate(fig.data):
        if i < len(avg_resources):
            trace.update(legend="legend1")
        elif i < len(avg_resources) + len(avg_production):
            trace.update(legend="legend2")
        elif i < len(avg_resources) + len(avg_production) + len(subsidies_over_time):
            trace.update(legend="legend3")
        else:
            trace.update(legend="legend4")

    # return fig

    # Create a new figure for individual company resources
    fig2 = go.Figure()

    for company in all_results[0]:  # Assume company list is consistent across all steps
        company_id = company['company_id']
        resources = [step_results[all_results[0].index(company)]['resources'] for step_results in all_results]

        fig2.add_trace(go.Scatter(
            x=list(range(len(resources))),
            y=resources,
            mode='lines',
            name=company_id
        ))

    fig2.update_layout(
        title="Individual Company Resources Over Time",
        xaxis_title="Simulation Step",
        yaxis_title="Resources",
        legend_title="Companies",
        height=600,
        width=1200
    )

    return fig, fig2  # Return both figures


def visualize_demand(demand_history, unmet_demand_history, step):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=list(range(len(demand_history))),
        y=demand_history,
        mode='lines',
        name='Avg Demand'
    ))

   # Add trace for unmet demand
    fig.add_trace(go.Scatter(
        x=list(range(len(unmet_demand_history))),
        y=unmet_demand_history,
        mode='lines',
        name='Total Unmet Demand',
        line=dict(color='red')
    ))

    fig.update_layout(
        title=f'Avg Market Demand and Total Unmet Demand Over Time',
        xaxis_title='Simulation Step',
        yaxis_title='Demand',
        height=400,
        width=800,
    )

    # Add a vertical line to indicate the current step
    fig.add_vline(x=step, line_dash="dash", line_color="red")

    return fig

if __name__ == "__main__":
    main()