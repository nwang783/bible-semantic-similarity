import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.path import Path
import matplotlib.patches as patches

EMBEDDINGS_FILE = "../data/pericopes_embeddings.npy"
PASSAGES_FILE = "../data/pericopes_with_text.csv"
THRESHOLD = 0.8


def load_embeddings(file_path):
    return np.load(file_path)

def get_passages(file_path):
    import pandas as pd
    df = pd.read_csv(file_path)
    return df['Passage Text'].tolist()


def main():
    embeddings = load_embeddings(EMBEDDINGS_FILE)
    print(f"Loaded embeddings with shape: {embeddings.shape}")

    # Load passages
    passages = get_passages(PASSAGES_FILE)

    # Create a similarity matrix
    similarity_matrix = cosine_similarity(embeddings)

    # Find and print pairs with similarity above the threshold
    edges = []
    for i in range(len(similarity_matrix)):
        for j in range(i + 1, len(similarity_matrix)):
            sim_score = similarity_matrix[i][j]
            if sim_score >= THRESHOLD:
                edges.append({
                    "source": i,
                    "target": j,
                    "weight": sim_score,
                    "source_text": passages[i],
                    "target_text": passages[j]
                })

    edges_df = pd.DataFrame(edges)
    print(f"Total edges found: {len(edges_df)}")

    # 4. Create arc diagram visualization
    fig, ax = plt.subplots(figsize=(20, 10))

    # Plot passages as points along x-axis
    x_positions = np.arange(len(passages))
    ax.scatter(x_positions, np.zeros(len(passages)), s=2, c='black', zorder=3)

    # Draw arcs for each edge
    for _, edge in edges_df.iterrows():
        x1, x2 = edge['source'], edge['target']
        width = abs(x2 - x1)
        center = (x1 + x2) / 2
        
        # Color based on similarity strength
        similarity = edge['weight']
        color = plt.cm.viridis((similarity - THRESHOLD) / (1 - THRESHOLD))

        # Create arc (height proportional to distance)
        height = width * 0.5
        
        arc = mpatches.FancyBboxPatch(
            (min(x1, x2), 0), width, height,
            boxstyle=f"round,pad=0,rounding_size={height}",
            linewidth=0.5,
            edgecolor=color,
            facecolor='none',
            alpha=0.6
        )
        
        verts = [
            (x1, 0),  # Start point
            (x1, height),  # Control point 1
            (x2, height),  # Control point 2
            (x2, 0),  # End point
        ]
        codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]
        path = Path(verts, codes)
        patch = patches.PathPatch(path, facecolor='none', edgecolor=color, 
                                linewidth=0.5, alpha=0.6)
        ax.add_patch(patch)

    # Styling
    ax.set_xlim(-10, len(passages) + 10)
    ax.set_ylim(-50, max(edges_df['target'] - edges_df['source']) * 0.3)
    ax.set_xlabel('Passage Index (Genesis → Revelation)', fontsize=12)
    ax.set_title(f'Biblical Intertextual Connections (similarity > {THRESHOLD})', fontsize=16)
    ax.axis('off')

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap='viridis', 
                            norm=plt.Normalize(vmin=THRESHOLD, vmax=1.0))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation='horizontal', pad=0.05)
    cbar.set_label('Semantic Similarity', fontsize=10)

    plt.tight_layout()
    plt.savefig(f'../bible_connections_{THRESHOLD}.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()