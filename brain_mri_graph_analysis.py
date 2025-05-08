"""
Brain MRI Graph Analysis
=======================
This project demonstrates the application of graph theory to analyze MRI brain images.
It includes segmentation, graph construction, and metric analysis.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import cv2
import tensorflow as tf
import torch
from scipy import ndimage
from sklearn.cluster import KMeans
from skimage import measure, segmentation, filters, feature

class BrainMRIGraphAnalysis:
    def __init__(self, input_dir="data/input", output_dir="data/output"):
        """
        Initialize the MRI Graph Analysis framework
        
        Parameters:
        -----------
        input_dir : str
            Directory containing MRI images
        output_dir : str
            Directory to save output results
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize graph
        self.graph = None
        
        # Default segmentation parameters
        self.seg_threshold = 0.3
        self.min_size = 100
        
        # Default graph construction parameters
        self.connectivity_threshold = 0.6
        
    def load_image(self, image_path):
        """Load and preprocess MRI image"""
        # Read image
        if torch.is_tensor(image_path):
            # If tensor is passed directly
            image = image_path.cpu().numpy()
        elif isinstance(image_path, np.ndarray):
            # If numpy array is passed directly
            image = image_path
        else:
            # If path is provided
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            
        # Normalize to 0-1 range
        image = image.astype(np.float32) / np.max(image)
        
        return image
    
    def segment_image(self, image, method="otsu"):
        """
        Segment the brain regions in MRI image
        
        Parameters:
        -----------
        image : ndarray
            Input MRI image
        method : str
            Segmentation method ('otsu', 'kmeans', 'watershed')
            
        Returns:
        --------
        ndarray
            Segmented image with labeled regions
        """
        if method == "otsu":
            # Apply Gaussian blur to reduce noise
            blurred = filters.gaussian(image, sigma=1.0)
            
            # Apply Otsu's thresholding
            thresh = filters.threshold_otsu(blurred)
            binary = blurred > thresh
            
            # Remove small objects and fill holes
            binary = ndimage.binary_opening(binary, structure=np.ones((3, 3)))
            binary = ndimage.binary_closing(binary, structure=np.ones((3, 3)))
            
            # Label connected regions
            labeled_image, num_features = ndimage.label(binary)
            
        elif method == "kmeans":
            # Reshape for KMeans
            pixels = image.reshape(-1, 1)
            
            # Apply KMeans clustering
            kmeans = KMeans(n_clusters=4, random_state=42)
            labels = kmeans.fit_predict(pixels)
            
            # Reshape back to image size
            segmented = labels.reshape(image.shape)
            
            # Keep only brain regions (assuming higher intensity clusters)
            brain_clusters = np.argsort(kmeans.cluster_centers_.flatten())[-2:]
            mask = np.isin(segmented, brain_clusters)
            
            # Label connected regions
            labeled_image, num_features = ndimage.label(mask)
            
        elif method == "watershed":
            # Apply gradient for watershed
            gradient = filters.sobel(image)
            
            # Create markers
            markers = np.zeros_like(image, dtype=int)
            markers[image < self.seg_threshold/2] = 1    # Background
            markers[image > self.seg_threshold*1.5] = 2  # Foreground
            
            # Apply watershed
            labeled_image = segmentation.watershed(gradient, markers)
            labeled_image = labeled_image - 1  # Make background 0
            
            # Remove small regions
            for region in np.unique(labeled_image):
                if region == 0:  # Skip background
                    continue
                region_size = np.sum(labeled_image == region)
                if region_size < self.min_size:
                    labeled_image[labeled_image == region] = 0
                    
            # Relabel consecutively
            labeled_image, num_features = ndimage.label(labeled_image > 0)
            
        return labeled_image
    
    def extract_region_features(self, image, labeled_image):
        """Extract features for each segmented region"""
        regions = measure.regionprops(labeled_image, intensity_image=image)
        
        # Extract centroids, areas, mean intensities
        centroids = [region.centroid for region in regions]
        areas = [region.area for region in regions]
        intensities = [region.mean_intensity for region in regions]
        
        # Calculate texture features (GLCM)
        textures = []
        for region in regions:
            bbox = region.bbox
            roi = image[bbox[0]:bbox[2], bbox[1]:bbox[3]]
            if roi.size > 0 and roi.shape[0] > 1 and roi.shape[1] > 1:
                glcm = feature.graycomatrix(
                    (roi * 255).astype(np.uint8), 
                    distances=[1], 
                    angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                    levels=8,
                    symmetric=True,
                    normed=True
                )
                contrast = feature.graycoprops(glcm, 'contrast').mean()
                homogeneity = feature.graycoprops(glcm, 'homogeneity').mean()
                textures.append((contrast, homogeneity))
            else:
                textures.append((0, 0))
        
        return {
            'centroids': centroids,
            'areas': areas,
            'intensities': intensities,
            'textures': textures
        }
    
    def construct_graph(self, features, connectivity="distance"):
        """
        Construct a graph from region features
        
        Parameters:
        -----------
        features : dict
            Dictionary of region features
        connectivity : str
            Method to determine edge connections ('distance', 'intensity', 'hybrid')
            
        Returns:
        --------
        networkx.Graph
            Constructed graph
        """
        G = nx.Graph()
        
        # Add nodes with attributes
        for i in range(len(features['centroids'])):
            G.add_node(i, 
                       pos=features['centroids'][i],
                       area=features['areas'][i],
                       intensity=features['intensities'][i],
                       texture=features['textures'][i] if i < len(features['textures']) else (0, 0))
        
        # Add edges based on connectivity method
        if connectivity == "distance":
            # Connect regions based on centroid distance
            for i in range(len(features['centroids'])):
                for j in range(i+1, len(features['centroids'])):
                    centroid_i = np.array(features['centroids'][i])
                    centroid_j = np.array(features['centroids'][j])
                    distance = np.linalg.norm(centroid_i - centroid_j)
                    
                    # Connect if below threshold (normalized by image size)
                    max_distance = np.sqrt(np.sum(np.array(image.shape)**2))
                    if distance / max_distance < self.connectivity_threshold:
                        weight = 1.0 - (distance / max_distance)
                        G.add_edge(i, j, weight=weight, type="distance")
        
        elif connectivity == "intensity":
            # Connect regions with similar intensities
            for i in range(len(features['intensities'])):
                for j in range(i+1, len(features['intensities'])):
                    intensity_diff = abs(features['intensities'][i] - features['intensities'][j])
                    
                    # Connect if intensity difference is small
                    if intensity_diff < self.connectivity_threshold:
                        weight = 1.0 - intensity_diff
                        G.add_edge(i, j, weight=weight, type="intensity")
        
        elif connectivity == "hybrid":
            # Combine distance and intensity
            for i in range(len(features['centroids'])):
                for j in range(i+1, len(features['centroids'])):
                    # Distance component
                    centroid_i = np.array(features['centroids'][i])
                    centroid_j = np.array(features['centroids'][j])
                    distance = np.linalg.norm(centroid_i - centroid_j)
                    max_distance = np.sqrt(np.sum(np.array(image.shape)**2))
                    norm_distance = distance / max_distance
                    
                    # Intensity component
                    intensity_diff = abs(features['intensities'][i] - features['intensities'][j])
                    
                    # Combined metric (50% distance, 50% intensity)
                    combined_metric = 0.5 * norm_distance + 0.5 * intensity_diff
                    
                    if combined_metric < self.connectivity_threshold:
                        weight = 1.0 - combined_metric
                        G.add_edge(i, j, weight=weight, type="hybrid")
        
        self.graph = G
        return G
    
    def calculate_graph_metrics(self):
        """Calculate important graph metrics for the constructed graph"""
        if self.graph is None:
            raise ValueError("Graph not constructed yet. Call construct_graph first.")
        
        G = self.graph
        metrics = {}
        
        # Basic graph properties
        metrics['num_nodes'] = G.number_of_nodes()
        metrics['num_edges'] = G.number_of_edges()
        metrics['density'] = nx.density(G)
        
        # Component analysis
        metrics['connected_components'] = nx.number_connected_components(G)
        
        # Centrality measures
        metrics['degree_centrality'] = nx.degree_centrality(G)
        metrics['betweenness_centrality'] = nx.betweenness_centrality(G)
        metrics['closeness_centrality'] = nx.closeness_centrality(G)
        
        # Clustering and community structure
        metrics['avg_clustering'] = nx.average_clustering(G)
        
        # Path-related metrics
        if nx.is_connected(G):
            metrics['avg_shortest_path'] = nx.average_shortest_path_length(G)
            metrics['diameter'] = nx.diameter(G)
        else:
            # Calculate for largest connected component
            largest_cc = max(nx.connected_components(G), key=len)
            largest_subgraph = G.subgraph(largest_cc).copy()
            
            metrics['avg_shortest_path'] = nx.average_shortest_path_length(largest_subgraph)
            metrics['diameter'] = nx.diameter(largest_subgraph)
            metrics['largest_component_size'] = len(largest_cc)
            
        return metrics
    
    def visualize_graph(self, labeled_image, save_path=None):
        """Visualize the graph overlaid on the segmented image"""
        if self.graph is None:
            raise ValueError("Graph not constructed yet. Call construct_graph first.")
        
        plt.figure(figsize=(12, 10))
        
        # Plot segmented image
        plt.subplot(1, 2, 1)
        plt.imshow(labeled_image, cmap='nipy_spectral')
        plt.title('Segmented Regions')
        plt.axis('off')
        
        # Plot graph
        plt.subplot(1, 2, 2)
        G = self.graph
        
        # Get node positions from graph
        pos = nx.get_node_attributes(G, 'pos')
        
        # Get node sizes based on area (normalized)
        areas = np.array([G.nodes[n]['area'] for n in G.nodes()])
        node_sizes = 50 + 500 * (areas / np.max(areas))
        
        # Get node colors based on intensity
        intensities = np.array([G.nodes[n]['intensity'] for n in G.nodes()])
        node_colors = intensities
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, 
                              node_color=node_colors, cmap='viridis',
                              alpha=0.8)
        
        # Get edge weights
        weights = np.array([G[u][v]['weight'] for u, v in G.edges()])
        edge_widths = 0.5 + 3 * weights
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.6)
        
        # Draw node labels
        nx.draw_networkx_labels(G, pos, font_size=8)
        
        plt.title('Brain Region Graph')
        plt.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def analyze_community_structure(self):
        """Detect and analyze communities in the brain graph"""
        if self.graph is None:
            raise ValueError("Graph not constructed yet. Call construct_graph first.")
        
        # Apply Louvain community detection
        from community import community_louvain
        
        partition = community_louvain.best_partition(self.graph)
        communities = {}
        
        # Group nodes by community
        for node, community_id in partition.items():
            if community_id not in communities:
                communities[community_id] = []
            communities[community_id].append(node)
        
        # Calculate community metrics
        community_metrics = {}
        for comm_id, nodes in communities.items():
            subgraph = self.graph.subgraph(nodes).copy()
            community_metrics[comm_id] = {
                'size': len(nodes),
                'density': nx.density(subgraph),
                'avg_intensity': np.mean([self.graph.nodes[n]['intensity'] for n in nodes]),
                'avg_clustering': nx.average_clustering(subgraph) if len(nodes) > 2 else 0
            }
        
        return communities, community_metrics, partition
    
    def detect_anomalies(self, reference_metrics=None):
        """
        Detect anomalies in the brain structure based on graph properties
        
        Parameters:
        -----------
        reference_metrics : dict, optional
            Reference metrics from a normal brain for comparison
            
        Returns:
        --------
        dict
            Dictionary of detected anomalies
        """
        if self.graph is None:
            raise ValueError("Graph not constructed yet. Call construct_graph first.")
        
        # Get current metrics
        metrics = self.calculate_graph_metrics()
        
        anomalies = {}
        
        # If reference metrics are provided, compare against them
        if reference_metrics:
            # Check global properties
            for key in ['density', 'avg_clustering', 'avg_shortest_path']:
                if key in metrics and key in reference_metrics:
                    # Calculate percent difference
                    diff = (metrics[key] - reference_metrics[key]) / reference_metrics[key] * 100
                    if abs(diff) > 20:  # If more than 20% difference
                        anomalies[key] = {
                            'current': metrics[key],
                            'reference': reference_metrics[key],
                            'percent_diff': diff
                        }
            
            # Check component structure
            if metrics['connected_components'] != reference_metrics['connected_components']:
                anomalies['components'] = {
                    'current': metrics['connected_components'],
                    'reference': reference_metrics['connected_components']
                }
        
        # Detect nodes with unusual centrality (outliers within this graph)
        # Betweenness centrality outliers
        betweenness = list(metrics['betweenness_centrality'].values())
        mean_b = np.mean(betweenness)
        std_b = np.std(betweenness)
        
        outlier_nodes = []
        for node, value in metrics['betweenness_centrality'].items():
            z_score = (value - mean_b) / std_b if std_b > 0 else 0
            if abs(z_score) > 2:  # More than 2 standard deviations
                outlier_nodes.append((node, value, z_score))
        
        if outlier_nodes:
            anomalies['centrality_outliers'] = outlier_nodes
            
        return anomalies
    
    def process_image(self, image_path, segmentation_method="otsu", 
                     connectivity_method="hybrid", visualize=True):
        """Full processing pipeline from image to graph analysis"""
        # Load and preprocess image
        image = self.load_image(image_path)
        
        # Segment the image
        labeled_image = self.segment_image(image, method=segmentation_method)
        
        # Extract features from regions
        features = self.extract_region_features(image, labeled_image)
        
        # Construct graph
        self.construct_graph(features, connectivity=connectivity_method)
        
        # Calculate metrics
        metrics = self.calculate_graph_metrics()
        
        # Detect communities
        communities, community_metrics, partition = self.analyze_community_structure()
        
        # Visualize if requested
        if visualize:
            save_path = os.path.join(self.output_dir, "graph_visualization.png")
            self.visualize_graph(labeled_image, save_path)
        
        return {
            'labeled_image': labeled_image,
            'graph': self.graph,
            'metrics': metrics,
            'communities': communities,
            'community_metrics': community_metrics
        }

# Example usage
if __name__ == "__main__":
    # Create sample synthetic MRI image for testing
    def create_synthetic_mri(size=(256, 256), num_regions=8):
        """Create a synthetic MRI image for testing"""
        image = np.zeros(size)
        
        # Add simulated brain tissue with different regions
        for _ in range(num_regions):
            center_x = np.random.randint(size[0] // 4, size[0] * 3 // 4)
            center_y = np.random.randint(size[1] // 4, size[1] * 3 // 4)
            radius = np.random.randint(10, 40)
            intensity = np.random.uniform(0.3, 1.0)
            
            y, x = np.ogrid[-center_x:size[0]-center_x, -center_y:size[1]-center_y]
            mask = x*x + y*y <= radius*radius
            image[mask] = intensity
        
        # Add noise
        noise = np.random.normal(0, 0.05, size)
        image += noise
        image = np.clip(image, 0, 1)
        
        return image
    
    # Create the directory structure
    os.makedirs("data/input", exist_ok=True)
    os.makedirs("data/output", exist_ok=True)
    
    # Generate a synthetic MRI image
    synthetic_mri = create_synthetic_mri()
    
    # Save the synthetic image
    plt.imsave("data/input/synthetic_mri.png", synthetic_mri, cmap='gray')
    
    # Process the image
    analyzer = BrainMRIGraphAnalysis()
    results = analyzer.process_image("data/input/synthetic_mri.png", visualize=True)
    
    # Print some results
    print("Graph Metrics:")
    print(f"Number of nodes: {results['metrics']['num_nodes']}")
    print(f"Number of edges: {results['metrics']['num_edges']}")
    print(f"Graph density: {results['metrics']['density']:.4f}")
    print(f"Average clustering coefficient: {results['metrics']['avg_clustering']:.4f}")
    print(f"Number of communities detected: {len(results['communities'])}")
    
    # Save the metrics to a file
    import json
    
    with open("data/output/graph_metrics.json", "w") as f:
        # Convert numpy values to Python types for JSON serialization
        metrics_for_json = {}
        for key, value in results['metrics'].items():
            if isinstance(value, dict):
                metrics_for_json[key] = {str(k): float(v) for k, v in value.items()}
            elif isinstance(value, (np.float64, np.float32, np.int64, np.int32)):
                metrics_for_json[key] = float(value)
            else:
                metrics_for_json[key] = value
                
        json.dump(metrics_for_json, f, indent=4)
        
    print(f"Results saved to {os.path.abspath('data/output')}")