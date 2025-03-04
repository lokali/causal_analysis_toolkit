import numpy as np
from causallearn.utils.cit import CIT
from conditional_independence import hsic_test
"""
    Input: data, test_method
    Output: matrix of unconditional independent test results.
"""
class CITestRunner:
    def __init__(self, data):
        self.data = data
        # self.labels = labels
        self.num_sample = data.shape[0]
        self.num_vars = data.shape[1]
    
    def run_csl(self, method):
        """
        Run a conditional independence test from Causal-learn.
        Supported methods: 'kci', 'rcit', 'fisherz'
        """
        results = np.zeros((self.num_vars, self.num_vars))
        np.fill_diagonal(results, 1)

        cit_obj = CIT(self.data, method)
        for i in range(self.num_vars):
            for j in range(i, self.num_vars):
                p_value = cit_obj(i, j)
                results[i, j] = results[j, i] = round(p_value, 4)
                
        return results
    
    def run_cit(self, method, x, y, z=None):
        """
        Run a conditional independence test from Causal-learn.
        Supported methods: 'kci', 'rcit', 'fisherz'
        """
        results = np.zeros((self.num_vars, self.num_vars))
        np.fill_diagonal(results, 1)

        cit_obj = CIT(self.data, method)
        p_value = cit_obj(x, y, z)
        return p_value
    
    def run_hsic(self):
        """
        Run the HSIC-based independence test.
        """
        results = np.zeros((self.num_vars, self.num_vars))
        np.fill_diagonal(results, 1)
        
        for i in range(self.num_vars):
            for j in range(i + 1, self.num_vars):
                p_value = hsic_test(self.data, i, j)['p_value']
                results[i, j] = results[j, i] = round(p_value, 4)
                
        return results
    

import seaborn as sns
import os 
import matplotlib.pyplot as plt
def plot_histograms_single(data, figsize=(12, 8)):
    sns.histplot(data, bins='auto', kde=True)
    if not os.path.exists("results/"):
        os.mkdir("results/")
    plt.savefig("results/histogram_one.pdf")
    plt.show()


import seaborn as sns
import matplotlib.pyplot as plt
import os 
def plot_histograms_all(df, column_size = 4, figsize=(12, 8)):
    """
    Plots histograms for all numerical columns in a DataFrame.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    bins (int, str, or list): Number of bins for the histograms (default: 'auto').
    figsize (tuple): Size of the overall figure (default: (12, 8)).
    """
    num_cols = df.select_dtypes(include=['number']).columns  # Select only numeric columns
    num_plots = len(num_cols)
    
    if num_plots == 0:
        print("No numeric columns found in the DataFrame.")
        return

    # Set up subplots dynamically
    rows = (num_plots // column_size) + (num_plots % column_size > 0)  # Arrange in a grid of 3 columns
    fig, axes = plt.subplots(rows, column_size, figsize=figsize)
    axes = axes.flatten()  # Flatten the axes array for easy iteration

    for i, col in enumerate(num_cols):
        sns.histplot(df[col], bins='auto', ax=axes[i], kde=True)
        axes[i].set_title(f'{col}')
        axes[i].set_xlabel('')
        # axes[i].set_xticklabels([])
        # axes[i].xaxis.set_visible(False)
        # axes[i].set_xticks([])

    
    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    plt.tight_layout()
    if not os.path.exists("results/"):
        os.mkdir("results/")
    plt.savefig("results/histogram_all.pdf")
    plt.show()



import seaborn as sns
import matplotlib.pyplot as plt
import os 
def run_scatterplot(df, kci_matrix=None, hsic_matrix=None, rci_matrix=None):
    g = sns.PairGrid(df, corner=True)
    # Define a custom plotting function
    def scatterplot_with_correlation(x, y, **kwargs):
        # Create the scatter plot
        ax = plt.gca()
        sns.scatterplot(x=x, y=y, **kwargs)
        
        # Get the indices for the labels from the DataFrame
        label_x = ax.get_xlabel()
        label_y = ax.get_ylabel()
        idx_x = df.columns.tolist().index(label_x)
        idx_y = df.columns.tolist().index(label_y)

        text = ""
        # Retrieve the correlation coefficient using indices
        if kci_matrix is not None:
            kci = kci_matrix[idx_y, idx_x]
            text += f"KCI: {kci:.2f}"
        if hsic_matrix is not None:
            hsic = hsic_matrix[idx_y, idx_x]
            text += f"\nHSIC: {hsic:.2f}"
        if rci_matrix is not None:
            rcit = rci_matrix[idx_y, idx_x]
            text += f"\nRCIT: {rcit:.2f}"
        
        # Annotate the correlation coefficient on the plot
        # text = f"KCI: {kci:.2f}\nHSIC: {hsic:.2f}\nRCIT: {rcit:.2f}"
        if kci<=0.05:
            ax.text(0.75, 0.95, text, transform=ax.transAxes, verticalalignment='top', 
                    horizontalalignment='left', fontsize=8, 
                    bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='lightgreen', alpha=0.5))
        else:
            ax.text(0.75, 0.95, text, transform=ax.transAxes, verticalalignment='top', 
                    horizontalalignment='left', fontsize=8, 
                    bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='lightsalmon', alpha=0.5))
        ax.set_title(label_x, fontsize=10)

    # Map the custom function to the lower triangles
    g.map_lower(scatterplot_with_correlation)

    # Optionally, map histograms to the diagonal
    g.map_diag(sns.histplot)

    if not os.path.exists("results/"):
        os.mkdir("results/")
    plt.savefig("results/scatterplot.pdf")
    plt.show()



from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.GraphUtils import GraphUtils
import os 
import numpy as np 
def run_pc(data, alpha, indep_test, label):
    """
    Runs the PC algorithm on the given data and saves the resulting graph.

    Parameters:
    - data (numpy.ndarray or pd.DataFrame.values): The dataset to process.
    - alpha (float): Significance level for independence tests.
    - indep_test (str): Name of the independence test ('kci', 'fisherz', etc.).

    Returns:
    - cg.G (causallearn graph object): The learned causal graph.
    """
    print(f"Running PC algorithm on data with shape: {data.shape}")

    # Run the PC algorithm
    cg = pc(data, alpha, indep_test)

    # Ensure the results directory exists
    if not os.path.exists("results/pc"):
        os.mkdir("results/pc")

    # Sanitize filename (replace '.' in alpha with '_')
    alpha_str = str(alpha).replace(".", "_")
    filename = f"{indep_test}_alpha{alpha_str}_D{data.shape[1]}_N{data.shape[0]}"
    file_path = os.path.join("results/pc/", filename)

    # Save the graph
    np.save(file_path + '.npy', cg.G.graph)

    # save plain graph
    cg.draw_pydot_graph(label) # show on display
    pyd = GraphUtils.to_pydot(cg.G, labels=label)
    pyd.write_png(file_path + '.png')
    return cg, file_path




from causallearn.graph.Edge import Edge
from causallearn.graph.Endpoint import Endpoint
from causallearn.graph.NodeType import NodeType
import pydot 
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import io
import matplotlib.colors as mcolors
import seaborn as sns 
def to_my_pydot(G, edges=None, labels=None, colors=None, title="", dpi=500):
    '''
    Convert a graph object to a DOT object, with nodes colored based on their labels.

    Parameters
    ----------
    G : Graph
        A graph object of causal-learn
    edges : list, optional
        Edges list of graph G
    labels_all : dict, optional
        Dictionary where keys are categories and values are lists of node labels
    colors : dict, optional
        Dictionary mapping categories to colors
    title : str, optional
        The name of the graph
    dpi : float, optional
        The dots per inch of dot object
    Returns
    -------
    pydot_g : pydot.Dot
        A dot object ready for rendering
    '''
    nodes = G.get_nodes()
    if labels is not None:
        assert len(labels) == len(nodes)

    pydot_g = pydot.Dot(title, graph_type="digraph", fontsize=18)
    pydot_g.obj_dict["attributes"]["dpi"] = dpi
    nodes = G.get_nodes()
    for i, node in enumerate(nodes):
        node_name = labels[i] if labels is not None else node.get_name()
        pydot_g.add_node(pydot.Node(i, label=node.get_name()))
        if node.get_node_type() == NodeType.LATENT:
            pydot_g.add_node(pydot.Node(i, label=node_name, style="filled", fillcolor=colors[node_name], shape='square'))
        else:
            pydot_g.add_node(pydot.Node(i, label=node_name, style="filled", fillcolor=colors[node_name]))

    def get_g_arrow_type(endpoint):
        if endpoint == Endpoint.TAIL:
            return 'none'
        elif endpoint == Endpoint.ARROW:
            return 'normal'
        elif endpoint == Endpoint.CIRCLE:
            return 'odot'
        else:
            raise NotImplementedError()

    if edges is None:
        edges = G.get_graph_edges()

    for edge in edges:
        node1 = edge.get_node1()
        node2 = edge.get_node2()
        node1_id = nodes.index(node1)
        node2_id = nodes.index(node2)
        dot_edge = pydot.Edge(node1_id, node2_id, dir='both', arrowtail=get_g_arrow_type(edge.get_endpoint1()),
                              arrowhead=get_g_arrow_type(edge.get_endpoint2()))

        if Edge.Property.dd in edge.properties:
            dot_edge.obj_dict["attributes"]["color"] = "green3"
        if Edge.Property.nl in edge.properties:
            dot_edge.obj_dict["attributes"]["penwidth"] = 2.0

        pydot_g.add_edge(dot_edge)

    return pydot_g

def to_pydot_color(general_graph, labels, label_to_color_dict, path):
    pyd = to_my_pydot(general_graph, labels=labels, colors=label_to_color_dict)
    tmp_png = pyd.create_png(f="png")
    fp = io.BytesIO(tmp_png)
    img = mpimg.imread(fp, format='png')
    plt.rcParams["figure.figsize"] = [20, 12]
    plt.rcParams["figure.autolayout"] = True
    plt.axis('off')
    plt.imshow(img)
    plt.savefig(path+'_color.png', dpi=500)
    plt.show()

def get_feature_to_color(category_to_features):  
    """
    Assigns colors to features based on their category.

    Parameters:
    - category_to_features (dict): A dictionary mapping categories to features.
    - color (str): Color palette to use. Options: ['pastel', 'dark', 'muted', 'bright']

    Returns:
    - feature_to_color (dict): Mapping from feature names to colors.
    """

    css_light_colors = [
    'lightblue', 'lightcoral', 'lightgreen', 'lightsalmon', 'lightpink', 'lightyellow',
    'lavender', 'thistle', 'honeydew', 'mintcream', 'azure', 'aliceblue',
    'beige', 'peachpuff', 'moccasin', 'palegoldenrod', 'powderblue', 'khaki',
    'wheat', 'blanchedalmond', 'papayawhip', 'mistyrose', 'lemonchiffon', 'seashell',
    'cornsilk', 'ivory', 'ghostwhite', 'floralwhite', 'aquamarine', 'lightcyan',
    'lightgoldenrodyellow', 'lightskyblue', 'lightsteelblue', 'mediumaquamarine',
    'paleturquoise', 'palegreen', 'pink', 'plum', 'skyblue', 'springgreen', 'turquoise']

    # Mapping from feature to category
    feature_to_category = {
        feature: category for category, features in category_to_features.items() for feature in features
    }

    # Assign the new pastel/muted/dark colors to categories
    category_to_color = dict(zip(category_to_features.keys(), css_light_colors))  # {'class1': pastel1, 'class2': pastel2}

    # Assign colors to features based on category
    feature_to_color = {
        feature: category_to_color[feature_to_category[feature]] for feature in feature_to_category.keys()
    }
    return feature_to_color


import numpy as np
from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.graph.GraphNode import GraphNode
from causallearn.graph.Edge import Edge
from causallearn.graph.Endpoint import Endpoint
def build_general_graph_from_adjacency_matrix(A, labels):
    directed_edges = {(i, j) for i in range(A.shape[0]) for j in range(A.shape[0]) if A[j, i] == 1 and A[i, j] == -1}
    undirected_edges = {(i, j) for i in range(A.shape[0]) for j in range(i + 1, A.shape[0]) if A[j, i] != 0 and A[i, j] != 0}

    nodes = [GraphNode(i) for i in labels]

    general_graph = GeneralGraph(nodes=nodes)
    for i, j in directed_edges: general_graph.add_edge(Edge(nodes[i], nodes[j], Endpoint.TAIL, Endpoint.ARROW))
    for i, j in undirected_edges: general_graph.add_edge(Edge(nodes[i], nodes[j], Endpoint.TAIL, Endpoint.TAIL))
    return general_graph



import matplotlib.pyplot as plt
import seaborn as sns 
import os 
def plot_heatmap(matrix):
    plt.figure(figsize=(12, 10))
    # Alternative: "Blues", "Reds", "coolwarm", "plasma"
    sns.heatmap(matrix, cmap="Blues", annot=True, fmt=".2f", linewidths=0.5, square=True)
    plt.title('P-value Matrix Heatmap')
    plt.xlabel('Variables')
    plt.ylabel('Variables')

    if not os.path.exists("results/"):
        os.mkdir("results/")
    plt.savefig("results/heatmap.pdf")
    plt.show()