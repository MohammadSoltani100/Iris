import streamlit as st
import networkx as nx
import plotly.graph_objects as go

from utils import (
    load_data_widget,
    show_dataframe_overview,
    numeric_columns,
    download_plotly_html,
    download_dataframe,
    add_common_layout_options
)

st.set_page_config(page_title="Metabolic Pathway", layout="wide")

st.title("Metabolic pathway network")
st.write("Upload an edge list. Required columns: source node and target node. Optional columns: enzyme, weight, activity.")

edge_df = load_data_widget("pathway_edges", "Upload pathway edge list")

if edge_df is None:
    st.stop()

show_dataframe_overview(edge_df)

c1, c2, c3 = st.columns(3)
with c1:
    source_col = st.selectbox("Source column", edge_df.columns.tolist())
with c2:
    target_col = st.selectbox("Target column", edge_df.columns.tolist())
with c3:
    label_col = st.selectbox("Edge label/enzyme column", ["None"] + edge_df.columns.tolist())

label_col = None if label_col == "None" else label_col

num_cols = numeric_columns(edge_df)
activity_col = st.selectbox("Node color/value column in edge table", ["None"] + num_cols)
activity_col = None if activity_col == "None" else activity_col

layout_method = st.selectbox("Network layout", ["spring", "kamada_kawai", "circular", "shell"])
node_size = st.slider("Node size", 10, 80, 35)

G = nx.DiGraph()

for _, row in edge_df.iterrows():
    source = str(row[source_col])
    target = str(row[target_col])
    if source and target and source != "nan" and target != "nan":
        edge_label = str(row[label_col]) if label_col else ""
        G.add_edge(source, target, label=edge_label)

if G.number_of_nodes() == 0:
    st.error("No valid edges were detected.")
    st.stop()

if layout_method == "spring":
    pos = nx.spring_layout(G, seed=42)
elif layout_method == "kamada_kawai":
    pos = nx.kamada_kawai_layout(G)
elif layout_method == "circular":
    pos = nx.circular_layout(G)
else:
    pos = nx.shell_layout(G)

edge_x, edge_y = [], []
for u, v in G.edges():
    x0, y0 = pos[u]
    x1, y1 = pos[v]
    edge_x += [x0, x1, None]
    edge_y += [y0, y1, None]

edge_trace = go.Scatter(
    x=edge_x,
    y=edge_y,
    line=dict(width=1.5, color="#888"),
    hoverinfo="none",
    mode="lines"
)

node_x, node_y, node_text, node_color = [], [], [], []

activity_map = {}
if activity_col:
    for _, row in edge_df.iterrows():
        source = str(row[source_col])
        target = str(row[target_col])
        val = row[activity_col]
        activity_map[source] = val
        activity_map[target] = val

for node in G.nodes():
    x, y = pos[node]
    node_x.append(x)
    node_y.append(y)
    node_text.append(node)
    node_color.append(activity_map.get(node, G.degree(node)))

node_trace = go.Scatter(
    x=node_x,
    y=node_y,
    mode="markers+text",
    text=node_text,
    textposition="top center",
    hoverinfo="text",
    marker=dict(
        size=node_size,
        color=node_color,
        colorscale="Viridis",
        showscale=True,
        colorbar=dict(title="Value / Degree"),
        line=dict(width=1, color="white")
    )
)

fig = go.Figure(data=[edge_trace, node_trace])
fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False)
fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False)
fig = add_common_layout_options(fig, "Metabolic pathway network", height=700)

st.plotly_chart(fig, use_container_width=True)

download_plotly_html(fig, "metabolic_pathway_network.html")
download_dataframe(edge_df, "pathway_edges.csv")