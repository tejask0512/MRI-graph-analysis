"""
MRI Graph Analysis - Main Application
====================================
This is the main application file that integrates the graph analysis and
segmentation model components to analyze MRI brain images.
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import torch
import cv2
import networkx as nx
from scipy import ndimage
import pandas as pd
import time
import json
from skimage import io, color, filters, measure, feature
from mri_graph_segmentation_model import MRISegmentationModel
from brain_mri_graph_analysis import BrainMRIGraphAnalysis


class MRIGraphAnalysisApp:
    def __init__(self, config=None):
        """
        Initialize the MRI Graph Analysis Application
        
        Parameters:
        -----------
        config : dict, optional
            Configuration dictionary
        """
        # Default configuration
        self.config = {
            'input_dir': 'data/input',
            'output_dir': 'data/output',
            'model_dir': 'models',
            'use_deep_learning': True,
            'segmentation_method': 'deep_learning',  # 'otsu', 'kmeans', 'watershed', 'deep_learning'
            'connectivity_method': 'hybrid',  # 'distance', 'intensity', 'hybrid'
            'visualize': True,
            'img_size': (256, 256),
            'batch_size': 8,
            'graph_metrics_file': 'graph_metrics.csv'
        }
        
        # Update with user config if provided
        if config:
            self.config.update(config)
            
        # Create necessary directories
        os.makedirs(self.config['input_dir'], exist_ok=True)
        os.makedirs(self.config['output_dir'], exist_ok=True)
        os.makedirs(self.config['model_dir'], exist_ok=True)
        
        # Initialize components
        self.graph_analyzer = BrainMRIGraphAnalysis(
            input_dir=self.config['input_dir'],
            output_dir=self.config['output_dir']
        )
        
        if self.config['use_deep_learning']:
            self.segmentation_model = MRISegmentationModel(
                input_shape=(*self.config['img_size'], 1),
                model_dir=self.config['model_dir']
            )
        else:
            self.segmentation_model = None
    
    def preprocess_image(self, image_path):
        """
        Preprocess an MRI image for analysis
        
        Parameters:
        -----------
        image_path : str
            Path to the input MRI image
            
        Returns:
        --------
        ndarray
            Preprocessed image
        """
        # Load image
        if isinstance(image_path, np.ndarray):
            image = image_path
        else:
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            
        # Resize to standard size
        image = cv2.resize(image, self.config['img_size'])
        
        # Normalize to 0-1 range
        image = image.astype(np.float32) / 255.0
        
        return image
    
    def segment_brain(self, image):
        """
        Segment brain regions in the MRI image
        
        Parameters:
        -----------
        image : ndarray
            Input MRI image
            
        Returns:
        --------
        ndarray
            Segmented image with labeled regions
        """
        if self.config['segmentation_method'] == 'deep_learning' and self.segmentation_model:
            # Use deep learning model for segmentation
            mask = self.segmentation_model.predict(image)
            mask = mask[..., 0] if len(mask.shape) == 3 else mask
            
            # Label connected components in the mask
            labeled_image, num_features = ndimage.label(mask > 0.5)
            
        else:
            # Use classical segmentation methods
            labeled_image = self.graph_analyzer.segment_image(
                image, method=self.config['segmentation_method'])
            
        return labeled_image
    
    def analyze_single_image(self, image_path, save_results=True):
        """
        Analyze a single MRI image
        
        Parameters:
        -----------
        image_path : str
            Path to the input MRI image
        save_results : bool
            Whether to save results to disk
            
        Returns:
        --------
        dict
            Analysis results
        """
        print(f"Analyzing image: {image_path}")
        
        # Preprocess image
        image = self.preprocess_image(image_path)
        
        # Segment brain regions
        labeled_image = self.segment_brain(image)
        
        # Extract features and build graph
        features = self.graph_analyzer.extract_region_features(image, labeled_image)
        graph = self.graph_analyzer.construct_graph(
            features, connectivity=self.config['connectivity_method'])
        
        # Calculate graph metrics
        metrics = self.graph_analyzer.calculate_graph_metrics()
        
        # Analyze community structure
        communities, community_metrics, partition = self.graph_analyzer.analyze_community_structure()
        
        # Visualize graph if requested
        if self.config['visualize']:
            if save_results:
                filename = os.path.basename(image_path).split('.')[0] if isinstance(image_path, str) else "image"
                save_path = os.path.join(self.config['output_dir'], f"{filename}_graph_viz.png")
            else:
                save_path = None
                
            self.graph_analyzer.visualize_graph(labeled_image, save_path)
        
        # Save metrics to CSV
        if save_results:
            filename = os.path.basename(image_path).split('.')[0] if isinstance(image_path, str) else "image"
            results_path = os.path.join(self.config['output_dir'], f"{filename}_metrics.json")
            
            # Convert complex data types for JSON serialization
            metrics_for_json = {}
            for key, value in metrics.items():
                if isinstance(value, dict):
                    metrics_for_json[key] = {str(k): float(v) for k, v in value.items()}
                elif isinstance(value, (np.float64, np.float32, np.int64, np.int32)):
                    metrics_for_json[key] = float(value)
                else:
                    metrics_for_json[key] = value
            
            with open(results_path, 'w') as f:
                json.dump(metrics_for_json, f, indent=4)
                
            print(f"Results saved to {results_path}")
        
        return {
            'image': image,
            'labeled_image': labeled_image,
            'graph': graph,
            'metrics': metrics,
            'communities': communities,
            'community_metrics': community_metrics
        }
    
    def batch_analyze(self, image_paths):
        """
        Analyze multiple MRI images and compile results
        
        Parameters:
        -----------
        image_paths : list
            List of paths to MRI images
            
        Returns:
        --------
        pandas.DataFrame
            Analysis results for all images
        """
        all_metrics = []
        
        for image_path in image_paths:
            print(f"Processing {image_path}...")
            results = self.analyze_single_image(image_path)
            
            # Extract key metrics
            metrics = results['metrics']
            metrics_dict = {
                'image_name': os.path.basename(image_path),
                'num_nodes': metrics['num_nodes'],
                'num_edges': metrics['num_edges'],
                'density': metrics['density'],
                'avg_clustering': metrics['avg_clustering'],
                'connected_components': metrics['connected_components']
            }
            
            # Add average centrality measures
            if 'degree_centrality' in metrics:
                metrics_dict['avg_degree_centrality'] = np.mean(list(metrics['degree_centrality'].values()))
            if 'betweenness_centrality' in metrics:
                metrics_dict['avg_betweenness_centrality'] = np.mean(list(metrics['betweenness_centrality'].values()))
            if 'closeness_centrality' in metrics:
                metrics_dict['avg_closeness_centrality'] = np.mean(list(metrics['closeness_centrality'].values()))
            
            # Add path metrics if available
            if 'avg_shortest_path' in metrics:
                metrics_dict['avg_shortest_path'] = metrics['avg_shortest_path']
            if 'diameter' in metrics:
                metrics_dict['diameter'] = metrics['diameter']
                
            # Community structure information
            metrics_dict['num_communities'] = len(results['communities'])
            
            all_metrics.append(metrics_dict)
        
        # Create DataFrame
        results_df = pd.DataFrame(all_metrics)
        
        # Save to CSV
        csv_path = os.path.join(self.config['output_dir'], self.config['graph_metrics_file'])
        results_df.to_csv(csv_path, index=False)
        print(f"Batch analysis results saved to {csv_path}")
        
        return results_df
    
    def train_segmentation_model(self, num_samples=200, epochs=30):
        """
        Train the segmentation model on synthetic data
        
        Parameters:
        -----------
        num_samples : int
            Number of samples to generate for training
        epochs : int
            Number of training epochs
            
        Returns:
        --------
        History object
            Training history
        """
        if not self.segmentation_model:
            raise ValueError("Segmentation model not initialized")
        
        print("Generating synthetic dataset...")
        images, masks = self.segmentation_model.prepare_synthetic_dataset(
            num_samples=num_samples, size=self.config['img_size'])
        
        print(f"Training segmentation model on {num_samples} synthetic samples...")
        history = self.segmentation_model.train(
            images, masks, batch_size=self.config['batch_size'], epochs=epochs)
        
        # Visualize some predictions
        vis_idx = np.random.choice(len(images), 5, replace=False)
        self.segmentation_model.visualize_predictions(
            images[vis_idx], masks[vis_idx], 
            save_dir=os.path.join(self.config['output_dir'], 'model_predictions'))
        
        return history
    
    def compare_normal_vs_abnormal(self, normal_image_path, abnormal_image_path):
        """
        Compare graph metrics between normal and abnormal MRI images
        
        Parameters:
        -----------
        normal_image_path : str
            Path to normal MRI image
        abnormal_image_path : str
            Path to abnormal MRI image
            
        Returns:
        --------
        dict
            Comparison results
        """
        # Analyze both images
        normal_results = self.analyze_single_image(normal_image_path, save_results=True)
        abnormal_results = self.analyze_single_image(abnormal_image_path, save_results=True)
        
        # Detect anomalies in abnormal image using normal as reference
        anomalies = self.graph_analyzer.detect_anomalies(normal_results['metrics'])
        
        # Visualize comparison
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # Original images
        axes[0, 0].imshow(normal_results['image'], cmap='gray')
        axes[0, 0].set_title('Normal MRI')
        axes[0, 0].axis('off')
        
        axes[1, 0].imshow(abnormal_results['image'], cmap='gray')
        axes[1, 0].set_title('Abnormal MRI')
        axes[1, 0].axis('off')
        
        # Segmentation results
        axes[0, 1].imshow(normal_results['labeled_image'], cmap='nipy_spectral')
        axes[0, 1].set_title('Normal Segmentation')
        axes[0, 1].axis('off')
        
        axes[1, 1].imshow(abnormal_results['labeled_image'], cmap='nipy_spectral')
        axes[1, 1].set_title('Abnormal Segmentation')
        axes[1, 1].axis('off')
        
        # Graph visualization
        normal_graph = normal_results['graph']
        abnormal_graph = abnormal_results['graph']
        
        pos_normal = nx.get_node_attributes(normal_graph, 'pos')
        nx.draw_networkx(normal_graph, pos=pos_normal, ax=axes[0, 2], 
                         node_size=50, node_color='blue', with_labels=False)
        axes[0, 2].set_title('Normal Graph Structure')
        axes[0, 2].axis('off')
        
        pos_abnormal = nx.get_node_attributes(abnormal_graph, 'pos')
        nx.draw_networkx(abnormal_graph, pos=pos_abnormal, ax=axes[1, 2], 
                         node_size=50, node_color='red', with_labels=False)
        axes[1, 2].set_title('Abnormal Graph Structure')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.config['output_dir'], 'normal_vs_abnormal_comparison.png'), dpi=300)
        plt.close()
        
        # Create comparison table
        comparison = {
            'metrics': {},
            'anomalies': anomalies
        }
        
        for key in normal_results['metrics']:
            if isinstance(normal_results['metrics'][key], (int, float, np.number)):
                comparison['metrics'][key] = {
                    'normal': normal_results['metrics'][key],
                    'abnormal': abnormal_results['metrics'][key],
                    'percent_diff': (abnormal_results['metrics'][key] - normal_results['metrics'][key]) / 
                                    normal_results['metrics'][key] * 100 if normal_results['metrics'][key] != 0 else float('inf')
                }
        
        # Save comparison results
        with open(os.path.join(self.config['output_dir'], 'comparison_results.json'), 'w') as f:
            # Convert numpy types to Python standard types
            comparison_json = json.dumps(comparison, default=lambda x: float(x) if isinstance(x, np.number) else x)
            f.write(comparison_json)
        
        return comparison


# Synthetic data creation function
def create_synthetic_abnormal_mri(size=(256, 256), tumor_radius=25):
    """Create a synthetic MRI image with an abnormal region (tumor)"""
    image = np.zeros(size)
    
    # Create brain outline (ellipse)
    center_x = size[0] // 2
    center_y = size[1] // 2
    a = 85  # semi-major axis
    b = 105  # semi-minor axis
    
    y, x = np.ogrid[-center_x:size[0]-center_x, -center_y:size[1]-center_y]
    brain_mask = (x*x)/(a*a) + (y*y)/(b*b) <= 1
    
    # Add brain tissue
    image[brain_mask] = 0.6
    
    # Add normal regions
    for _ in range(5):
        # Random position inside the brain
        while True:
            pos_x = np.random.randint(0, size[0])
            pos_y = np.random.randint(0, size[1])
            if brain_mask[pos_x, pos_y]:
                break
                
        radius = np.random.randint(10, 20)
        
        # Create region mask
        y, x = np.ogrid[-pos_x:size[0]-pos_x, -pos_y:size[1]-pos_y]
        region_mask = x*x + y*y <= radius*radius
        
        # Ensure region is inside brain
        region_mask = np.logical_and(region_mask, brain_mask)
        
        # Add region to image
        image[region_mask] = 0.8
    
    # Add abnormal region (tumor)
    tumor_x = center_x + np.random.randint(-30, 30)
    tumor_y = center_y + np.random.randint(-30, 30)
    
    y, x = np.ogrid[-tumor_x:size[0]-tumor_x, -tumor_y:size[1]-tumor_y]
    tumor_mask = x*x + y*y <= tumor_radius*tumor_radius
    
    # Ensure tumor is inside brain
    tumor_mask = np.logical_and(tumor_mask, brain_mask)
    
    # Add tumor with higher intensity
    image[tumor_mask] = 1.0
    
    # Add noise
    noise = np.random.normal(0, 0.05, size)
    image = image + noise
    image = np.clip(image, 0, 1)
    
    return image


def main():
    parser = argparse.ArgumentParser(description='MRI Graph Analysis Application')
    parser.add_argument('--input', type=str, help='Input MRI image or directory')
    parser.add_argument('--output', type=str, default='data/output', help='Output directory')
    parser.add_argument('--train', action='store_true', help='Train segmentation model')
    parser.add_argument('--compare', action='store_true', help='Compare normal vs abnormal MRI')
    parser.add_argument('--batch', action='store_true', help='Process multiple images')
    parser.add_argument('--method', type=str, default='deep_learning', 
                        choices=['otsu', 'kmeans', 'watershed', 'deep_learning'],
                        help='Segmentation method')
    parser.add_argument('--connectivity', type=str, default='hybrid',
                        choices=['distance', 'intensity', 'hybrid'],
                        help='Graph connectivity method')
    
    args = parser.parse_args()
    
    # Configuration
    config = {
        'output_dir': args.output,
        'segmentation_method': args.method,
        'connectivity_method': args.connectivity
    }
    
    # Initialize application
    app = MRIGraphAnalysisApp(config)
    
    # Create demo directories
    os.makedirs('data/input', exist_ok=True)
    os.makedirs('data/output', exist_ok=True)
    
    if args.train:
        # Train segmentation model
        app.train_segmentation_model(num_samples=200, epochs=30)
        
    elif args.compare:
        # Generate synthetic normal and abnormal MRI images
        normal_mri = create_synthetic_mri(size=(256, 256), num_regions=6)
        abnormal_mri = create_synthetic_abnormal_mri(size=(256, 256), tumor_radius=25)
        
        # Save images
        normal_path = os.path.join('data/input', 'synthetic_normal.png')
        abnormal_path = os.path.join('data/input', 'synthetic_abnormal.png')
        
        plt.imsave(normal_path, normal_mri, cmap='gray')
        plt.imsave(abnormal_path, abnormal_mri, cmap='gray')
        
        # Compare images
        comparison = app.compare_normal_vs_abnormal(normal_path, abnormal_path)
        print("Comparison completed. Results saved to output directory.")
        
    elif args.batch and args.input:
        # Process multiple images
        input_path = args.input
        
        if os.path.isdir(input_path):
            # Process all images in directory
            image_files = [os.path.join(input_path, f) for f in os.listdir(input_path)
                          if f.endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp'))]
            
            if not image_files:
                print(f"No image files found in {input_path}")
                return
            
            app.batch_analyze(image_files)
            
        else:
            print("For batch processing, input must be a directory")
    
    elif args.input:
        # Process single image
        if os.path.isfile(args.input):
            results = app.analyze_single_image(args.input)
            print("Analysis completed. Results saved to output directory.")
        else:
            print(f"Input file {args.input} not found")
    
    else:
        # Demo mode - create and analyze synthetic image
        print("Running in demo mode with synthetic MRI images...")
        
        # Create synthetic MRI
        from brain_mri_graph_analysis import create_synthetic_mri
        synthetic_mri = create_synthetic_mri()
        
        # Save image
        demo_path = os.path.join('data/input', 'demo_synthetic_mri.png')
        plt.imsave(demo_path, synthetic_mri, cmap='gray')
        
        # Analyze image
        results = app.analyze_single_image(demo_path)
        print("Demo analysis completed. Results saved to output directory.")


if __name__ == "__main__":
    main()