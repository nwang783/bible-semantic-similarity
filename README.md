# Interactive Bible Semantic Similarity Explorer

An interactive web-based visualization tool for exploring semantic connections between Bible passages using AI embeddings.

## Features

### 🎨 Multiple Visualization Types

1. **Arc Diagram** - Beautiful arc visualization showing connections between passages
   - Hover over arcs to see passage texts and similarity scores
   - Color-coded by Testament (Blue=OT, Red=NT, Purple=Cross-Testament)
   - Book boundaries and labels for easy navigation
   - Interactive zoom and pan

2. **Network Graph** - Force-directed network showing highly connected passages
   - Node size indicates number of connections
   - Interactive exploration of passage clusters
   - Displays top connections for better performance

3. **Heatmap** - Matrix view of similarity scores
   - Quick overview of similarity patterns
   - Sampled for performance with large datasets

4. **Statistics Dashboard** - Detailed analytics
   - Most connected passages
   - Connections by book
   - Cross-testament connection counts
   - Average and maximum similarity scores

### 🔍 Interactive Features

- **Adjustable Threshold Slider** - Dynamically filter connections by similarity (0.5-1.0)
- **Search Functionality** - Search for specific passages, books, or keywords
- **Hover Tooltips** - View passage text and metadata by hovering
- **Detailed Passage Comparison** - Side-by-side view of connected passages
- **Book Labels** - Toggle book boundaries and names
- **Testament Color Coding** - Visual distinction between Old and New Testament

### 📊 Analytics

- Total connections found
- Average similarity scores
- Cross-testament connections
- Most connected passages
- Books with most connections
- Interactive charts and graphs

## Installation

The required packages are already installed in your virtual environment:
- plotly
- streamlit
- networkx
- numpy
- pandas
- scikit-learn

## Usage

1. **Activate your virtual environment** (if not already activated):
   ```bash
   cd scripts
   source venv/bin/activate
   ```

2. **Run the Streamlit app**:
   ```bash
   streamlit run interactive_visualization.py
   ```

3. **Open your browser** - Streamlit will automatically open at `http://localhost:8501`

## How to Use

### Basic Exploration
1. Use the **Similarity Threshold** slider in the sidebar to adjust which connections are shown
2. Select different **Visualization Types** to explore the data in different ways
3. **Hover** over connections or passages to see detailed information
4. **Zoom and pan** in the visualizations to explore specific regions

### Search
1. Enter keywords in the **Search** box (e.g., "Genesis", "creation", "love")
2. The visualization will filter to show only matching connections

### Detailed Comparison
1. Check **"Show detailed comparison"** in the sidebar
2. Select a connection from the dropdown
3. View both passages side-by-side with their full text

### Tips
- Start with a higher threshold (0.8-0.9) to see the strongest connections
- Lower the threshold to discover more subtle connections
- Use the Network Graph to identify clusters of related passages
- Check the Statistics view to find the most connected passages

## Performance Notes

- The Arc Diagram may be slow with very low thresholds (many connections)
- Network Graph limits to top N connections for better performance
- Heatmap uses sampling for faster rendering
- Data is cached for quick threshold adjustments

## Data Files

The tool requires these files (already in your project):
- `../data/pericopes_embeddings.npy` - Passage embeddings
- `../data/pericopes_with_text.csv` - Passage metadata and text

## Customization

You can customize the visualization by modifying these parameters in the code:
- `THRESHOLD` - Default similarity threshold
- Color schemes for testaments
- Maximum nodes in network graph
- Heatmap sample size
- Arc diagram height and styling
