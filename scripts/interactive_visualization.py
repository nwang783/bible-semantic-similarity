import streamlit as st
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import networkx as nx
from collections import defaultdict

# Configuration
EMBEDDINGS_FILE = "../data/pericopes_embeddings.npy"
PASSAGES_FILE = "../data/pericopes_with_text.csv"

# Bible book order and testament
BIBLE_BOOKS_ORDER = [
    # Old Testament
    "GEN", "EXO", "LEV", "NUM", "DEU", "JOS", "JDG", "RUT", "1SA", "2SA",
    "1KI", "2KI", "1CH", "2CH", "EZR", "NEH", "EST", "JOB", "PSA", "PRO",
    "ECC", "SON", "ISA", "JER", "LAM", "EZE", "DAN", "HOS", "JOE", "AMO",
    "OBA", "JON", "MIC", "NAH", "HAB", "ZEP", "HAG", "ZEC", "MAL",
    # New Testament
    "MAT", "MAR", "LUK", "JOH", "ACT", "ROM", "1CO", "2CO", "GAL", "EPH",
    "PHP", "COL", "1TH", "2TH", "1TI", "2TI", "TIT", "PHM", "HEB", "JAM",
    "1PE", "2PE", "1JO", "2JO", "3JO", "JUD", "REV"
]

BOOK_NAMES = {
    "GEN": "Genesis", "EXO": "Exodus", "LEV": "Leviticus", "NUM": "Numbers", "DEU": "Deuteronomy",
    "JOS": "Joshua", "JDG": "Judges", "RUT": "Ruth", "1SA": "1 Samuel", "2SA": "2 Samuel",
    "1KI": "1 Kings", "2KI": "2 Kings", "1CH": "1 Chronicles", "2CH": "2 Chronicles",
    "EZR": "Ezra", "NEH": "Nehemiah", "EST": "Esther", "JOB": "Job", "PSA": "Psalms",
    "PRO": "Proverbs", "ECC": "Ecclesiastes", "SON": "Song of Solomon", "ISA": "Isaiah",
    "JER": "Jeremiah", "LAM": "Lamentations", "EZE": "Ezekiel", "DAN": "Daniel",
    "HOS": "Hosea", "JOE": "Joel", "AMO": "Amos", "OBA": "Obadiah", "JON": "Jonah",
    "MIC": "Micah", "NAH": "Nahum", "HAB": "Habakkuk", "ZEP": "Zephaniah", "HAG": "Haggai",
    "ZEC": "Zechariah", "MAL": "Malachi", "MAT": "Matthew", "MAR": "Mark", "LUK": "Luke",
    "JOH": "John", "ACT": "Acts", "ROM": "Romans", "1CO": "1 Corinthians", "2CO": "2 Corinthians",
    "GAL": "Galatians", "EPH": "Ephesians", "PHP": "Philippians", "COL": "Colossians",
    "1TH": "1 Thessalonians", "2TH": "2 Thessalonians", "1TI": "1 Timothy", "2TI": "2 Timothy",
    "TIT": "Titus", "PHM": "Philemon", "HEB": "Hebrews", "JAM": "James", "1PE": "1 Peter",
    "2PE": "2 Peter", "1JO": "1 John", "2JO": "2 John", "3JO": "3 John", "JUD": "Jude",
    "REV": "Revelation"
}

OLD_TESTAMENT_BOOKS = BIBLE_BOOKS_ORDER[:39]
NEW_TESTAMENT_BOOKS = BIBLE_BOOKS_ORDER[39:]

@st.cache_data
def load_data():
    """Load embeddings and passages data."""
    embeddings = np.load(EMBEDDINGS_FILE)
    df = pd.read_csv(PASSAGES_FILE)

    # Handle missing passage text
    df['Passage Text'] = df['Passage Text'].fillna('')
    df['Summary'] = df['Summary'].fillna('')

    # Filter out passages with blank text
    valid_mask = df['Passage Text'].str.strip() != ''
    df = df[valid_mask]  # Keep original indices - don't reset!
    embeddings = embeddings[valid_mask]

    # Add full reference
    df['Reference'] = df.apply(
        lambda row: f"{row['Book']} {row['Chapter']}:{row['Start Verse']}" +
        (f"-{row['End Verse']}" if row['Start Verse'] != row['End Verse'] else ""),
        axis=1
    )

    # Add testament
    df['Testament'] = df['Book'].apply(lambda x: 'Old Testament' if x in OLD_TESTAMENT_BOOKS else 'New Testament')

    return embeddings, df

@st.cache_data
def compute_similarity_matrix(embeddings):
    """Compute cosine similarity matrix."""
    return cosine_similarity(embeddings)

@st.cache_data
def get_connections(similarity_matrix, threshold, passages_df):
    """Get connections above threshold."""
    edges = []
    # Get list of actual index values to map positional indices to DataFrame indices
    idx_list = passages_df.index.tolist()

    for i in range(len(similarity_matrix)):
        for j in range(i + 1, len(similarity_matrix)):
            sim_score = similarity_matrix[i][j]
            if sim_score >= threshold:
                source_text = str(passages_df.iloc[i]['Passage Text'])
                target_text = str(passages_df.iloc[j]['Passage Text'])

                edges.append({
                    "source_idx": idx_list[i],  # Use actual DataFrame index
                    "target_idx": idx_list[j],  # Use actual DataFrame index
                    "similarity": sim_score,
                    "source_ref": passages_df.iloc[i]['Reference'],
                    "target_ref": passages_df.iloc[j]['Reference'],
                    "source_text": source_text[:200] + ("..." if len(source_text) > 200 else ""),
                    "target_text": target_text[:200] + ("..." if len(target_text) > 200 else ""),
                    "source_book": passages_df.iloc[i]['Book'],
                    "target_book": passages_df.iloc[j]['Book'],
                })
    return pd.DataFrame(edges)

def get_book_boundaries(passages_df):
    """Get the start and end indices for each book."""
    boundaries = {}
    for book in BIBLE_BOOKS_ORDER:
        book_indices = passages_df[passages_df['Book'] == book].index.tolist()
        if book_indices:
            boundaries[book] = {
                'start': min(book_indices),
                'end': max(book_indices),
                'middle': (min(book_indices) + max(book_indices)) / 2,
                'count': len(book_indices)
            }
    return boundaries

def create_arc_diagram(passages_df, edges_df, book_boundaries, show_book_labels=True):
    """Create an interactive arc diagram using Plotly."""
    fig = go.Figure()

    # Color mapping for testaments
    testament_colors = {
        'Old Testament': '#3498db',  # Blue
        'New Testament': '#e74c3c',  # Red
    }

    # Add book boundaries and labels
    if show_book_labels:
        for book, bounds in book_boundaries.items():
            testament = 'Old Testament' if book in OLD_TESTAMENT_BOOKS else 'New Testament'
            color = testament_colors[testament]

            # Add vertical line at book start
            fig.add_shape(
                type="line",
                x0=bounds['start'], y0=-10, x1=bounds['start'], y1=0,
                line=dict(color=color, width=1, dash="dot"),
                opacity=0.3
            )

            # Add book label
            fig.add_annotation(
                x=bounds['middle'],
                y=-15,
                text=BOOK_NAMES.get(book, book),
                showarrow=False,
                font=dict(size=8, color=color),
                textangle=-45,
                xanchor='right'
            )

    # Add arcs for connections
    for _, edge in edges_df.iterrows():
        x1, x2 = edge['source_idx'], edge['target_idx']
        width = abs(x2 - x1)
        height = width * 0.5

        # Determine if connection is inter-testament
        source_testament = 'Old Testament' if edge['source_book'] in OLD_TESTAMENT_BOOKS else 'New Testament'
        target_testament = 'Old Testament' if edge['target_book'] in OLD_TESTAMENT_BOOKS else 'New Testament'

        if source_testament == target_testament:
            color = testament_colors[source_testament]
        else:
            color = '#9b59b6'  # Purple for cross-testament connections

        # Create arc path
        t = np.linspace(0, np.pi, 50)
        x = np.linspace(x1, x2, 50)
        y = height * np.sin(t)

        # Hover text
        hover_text = (
            f"<b>Connection (Similarity: {edge['similarity']:.3f})</b><br><br>"
            f"<b>{edge['source_ref']}</b><br>"
            f"{edge['source_text']}<br><br>"
            f"<b>{edge['target_ref']}</b><br>"
            f"{edge['target_text']}"
        )

        fig.add_trace(go.Scatter(
            x=x, y=y,
            mode='lines',
            line=dict(color=color, width=0.5),
            opacity=0.6,
            hovertemplate=hover_text + '<extra></extra>',
            showlegend=False
        ))

    # Add passage points
    testament_traces = {}
    for testament in ['Old Testament', 'New Testament']:
        testament_df = passages_df[passages_df['Testament'] == testament]

        hover_text = [
            f"<b>{row['Reference']}</b><br>{row['Summary']}<br><br>{row['Passage Text'][:200]}..."
            for _, row in testament_df.iterrows()
        ]

        fig.add_trace(go.Scatter(
            x=testament_df.index,
            y=np.zeros(len(testament_df)),
            mode='markers',
            marker=dict(size=3, color=testament_colors[testament]),
            name=testament,
            hovertemplate='%{hovertext}<extra></extra>',
            hovertext=hover_text
        ))

    # Layout
    fig.update_layout(
        title=dict(
            text=f'Biblical Intertextual Connections<br><sub>{len(edges_df)} connections found</sub>',
            x=0.5,
            xanchor='center'
        ),
        xaxis=dict(
            title="Passage Index (Genesis → Revelation)",
            showgrid=False,
            zeroline=False,
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
        ),
        height=800,
        hovermode='closest',
        plot_bgcolor='white',
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        )
    )

    return fig

def create_network_graph(edges_df, passages_df, max_nodes=200):
    """Create a network graph visualization."""
    # Limit to top connections to avoid overcrowding
    top_edges = edges_df.nlargest(max_nodes, 'similarity')

    # Create network
    G = nx.Graph()

    # Add edges
    for _, edge in top_edges.iterrows():
        G.add_edge(
            edge['source_idx'],
            edge['target_idx'],
            weight=edge['similarity']
        )

    # Layout
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

    # Create edge traces
    edge_traces = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]

        edge_info = top_edges[
            ((top_edges['source_idx'] == edge[0]) & (top_edges['target_idx'] == edge[1])) |
            ((top_edges['source_idx'] == edge[1]) & (top_edges['target_idx'] == edge[0]))
        ].iloc[0]

        edge_trace = go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            mode='lines',
            line=dict(width=edge_info['similarity'] * 2, color='#888'),
            hoverinfo='none',
            showlegend=False
        )
        edge_traces.append(edge_trace)

    # Create node trace
    node_x = []
    node_y = []
    node_text = []
    node_colors = []

    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

        passage_info = passages_df.iloc[node]
        node_text.append(
            f"<b>{passage_info['Reference']}</b><br>"
            f"{passage_info['Summary']}<br>"
            f"Connections: {G.degree(node)}"
        )

        # Color by testament
        if passage_info['Testament'] == 'Old Testament':
            node_colors.append('#3498db')
        else:
            node_colors.append('#e74c3c')

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hovertemplate='%{text}<extra></extra>',
        text=node_text,
        marker=dict(
            size=[G.degree(node) * 3 + 5 for node in G.nodes()],
            color=node_colors,
            line=dict(width=1, color='white')
        ),
        showlegend=False
    )

    # Create figure
    fig = go.Figure(data=edge_traces + [node_trace])

    fig.update_layout(
        title=f'Network Graph of Top {len(G.edges())} Connections',
        showlegend=False,
        hovermode='closest',
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=700,
        plot_bgcolor='white'
    )

    return fig

def create_heatmap(similarity_matrix, passages_df, sample_size=500):
    """Create a heatmap of similarities (sampled for performance)."""
    # Sample passages for visualization
    step = max(1, len(passages_df) // sample_size)
    sampled_indices = list(range(0, len(passages_df), step))

    sampled_matrix = similarity_matrix[np.ix_(sampled_indices, sampled_indices)]
    sampled_refs = [passages_df.iloc[i]['Reference'] for i in sampled_indices]

    fig = go.Figure(data=go.Heatmap(
        z=sampled_matrix,
        x=sampled_refs,
        y=sampled_refs,
        colorscale='Viridis',
        hovertemplate='%{x}<br>%{y}<br>Similarity: %{z:.3f}<extra></extra>'
    ))

    fig.update_layout(
        title=f'Similarity Heatmap (sampled every {step} passages)',
        xaxis=dict(showticklabels=False),
        yaxis=dict(showticklabels=False),
        height=700
    )

    return fig

def show_statistics(edges_df, passages_df):
    """Display statistics about connections."""
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Connections", len(edges_df))

    with col2:
        avg_sim = edges_df['similarity'].mean() if len(edges_df) > 0 else 0
        st.metric("Average Similarity", f"{avg_sim:.3f}")

    with col3:
        # Count cross-testament connections
        cross_testament = edges_df[
            (edges_df['source_book'].isin(OLD_TESTAMENT_BOOKS) & edges_df['target_book'].isin(NEW_TESTAMENT_BOOKS)) |
            (edges_df['source_book'].isin(NEW_TESTAMENT_BOOKS) & edges_df['target_book'].isin(OLD_TESTAMENT_BOOKS))
        ]
        st.metric("Cross-Testament", len(cross_testament))

    with col4:
        max_sim = edges_df['similarity'].max() if len(edges_df) > 0 else 0
        st.metric("Max Similarity", f"{max_sim:.3f}")

    # Most connected passages
    st.subheader("Most Connected Passages")

    # Count connections per passage
    source_counts = edges_df['source_idx'].value_counts()
    target_counts = edges_df['target_idx'].value_counts()
    total_counts = (source_counts + target_counts).sort_values(ascending=False).head(10)

    top_passages = []
    for idx, count in total_counts.items():
        passage = passages_df.iloc[idx]
        top_passages.append({
            'Reference': passage['Reference'],
            'Summary': passage['Summary'],
            'Connections': count
        })

    st.dataframe(pd.DataFrame(top_passages), use_container_width=True)

    # Book statistics
    st.subheader("Connections by Book")

    book_stats = defaultdict(int)
    for _, edge in edges_df.iterrows():
        book_stats[edge['source_book']] += 1
        book_stats[edge['target_book']] += 1

    book_df = pd.DataFrame([
        {'Book': BOOK_NAMES.get(book, book), 'Connections': count}
        for book, count in sorted(book_stats.items(), key=lambda x: x[1], reverse=True)
    ]).head(15)

    fig = px.bar(book_df, x='Book', y='Connections',
                 title='Top 15 Books by Number of Connections',
                 color='Connections', color_continuous_scale='Viridis')
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

def main():
    st.set_page_config(
        page_title="Bible Semantic Similarity Explorer",
        page_icon="📖",
        layout="wide"
    )

    st.title("📖 Biblical Intertextual Connections Explorer")
    st.markdown("""
    Explore semantic connections between Bible passages using AI embeddings.
    This tool visualizes passages with similar meanings across the entire Bible.
    """)

    # Load data
    with st.spinner("Loading embeddings and passages..."):
        embeddings, passages_df = load_data()
        similarity_matrix = compute_similarity_matrix(embeddings)
        book_boundaries = get_book_boundaries(passages_df)

    st.success(f"Loaded {len(passages_df)} passages from {len(book_boundaries)} books")

    # Sidebar controls
    st.sidebar.header("⚙️ Controls")

    threshold = st.sidebar.slider(
        "Similarity Threshold",
        min_value=0.5,
        max_value=1.0,
        value=0.8,
        step=0.01,
        help="Minimum similarity score to show connections"
    )

    visualization_type = st.sidebar.radio(
        "Visualization Type",
        ["Arc Diagram", "Network Graph", "Heatmap", "Statistics"]
    )

    show_book_labels = st.sidebar.checkbox("Show Book Labels", value=True)

    # Search functionality
    st.sidebar.header("🔍 Search")
    search_query = st.sidebar.text_input("Search passages", placeholder="e.g., Genesis, creation, love")

    # Get connections
    with st.spinner("Computing connections..."):
        edges_df = get_connections(similarity_matrix, threshold, passages_df)

    # Filter by search
    if search_query:
        mask = (
            edges_df['source_ref'].str.contains(search_query, case=False) |
            edges_df['target_ref'].str.contains(search_query, case=False) |
            edges_df['source_text'].str.contains(search_query, case=False) |
            edges_df['target_text'].str.contains(search_query, case=False)
        )
        edges_df = edges_df[mask]
        st.info(f"Found {len(edges_df)} connections matching '{search_query}'")

    # Display visualization
    if visualization_type == "Arc Diagram":
        fig = create_arc_diagram(passages_df, edges_df, book_boundaries, show_book_labels)
        st.plotly_chart(fig, use_container_width=True)

        # Show connection details
        st.subheader("Connection Details")
        display_df = edges_df[['source_ref', 'target_ref', 'similarity']].copy()
        display_df.columns = ['From', 'To', 'Similarity']
        st.dataframe(
            display_df.sort_values('Similarity', ascending=False),
            use_container_width=True,
            height=300
        )

    elif visualization_type == "Network Graph":
        max_nodes = st.sidebar.slider("Max connections to show", 50, 500, 200, 50)
        if len(edges_df) > 0:
            fig = create_network_graph(edges_df, passages_df, max_nodes)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No connections found at this threshold.")

    elif visualization_type == "Heatmap":
        sample_size = st.sidebar.slider("Sample size", 100, 1000, 500, 100)
        fig = create_heatmap(similarity_matrix, passages_df, sample_size)
        st.plotly_chart(fig, use_container_width=True)

    elif visualization_type == "Statistics":
        if len(edges_df) > 0:
            show_statistics(edges_df, passages_df)
        else:
            st.warning("No connections found at this threshold.")

    # Detailed passage comparison
    if len(edges_df) > 0:
        st.sidebar.header("📊 Passage Comparison")

        if st.sidebar.checkbox("Show detailed comparison"):
            st.subheader("Detailed Passage Comparison")

            connection_options = [
                f"{row['source_ref']} ↔ {row['target_ref']} ({row['similarity']:.3f})"
                for _, row in edges_df.nlargest(20, 'similarity').iterrows()
            ]

            if connection_options:
                selected = st.selectbox("Select a connection to explore:", connection_options)
                selected_idx = connection_options.index(selected)
                selected_edge = edges_df.nlargest(20, 'similarity').iloc[selected_idx]

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown(f"### {selected_edge['source_ref']}")
                    source_passage = passages_df.loc[selected_edge['source_idx']]
                    st.markdown(f"**{source_passage['Summary']}**")
                    st.text_area("Passage Text", source_passage['Passage Text'], height=200, key=f"source_{selected_edge['source_idx']}")

                with col2:
                    st.markdown(f"### {selected_edge['target_ref']}")
                    target_passage = passages_df.loc[selected_edge['target_idx']]
                    st.markdown(f"**{target_passage['Summary']}**")
                    st.text_area("Passage Text", target_passage['Passage Text'], height=200, key=f"target_{selected_edge['target_idx']}")

                st.metric("Similarity Score", f"{selected_edge['similarity']:.4f}")

if __name__ == "__main__":
    main()
