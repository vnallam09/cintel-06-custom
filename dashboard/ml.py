import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Machine Learning imports
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, silhouette_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Global variables to store processed data
_processed_data = None
_ml_results = None

def preprocess_penguin_data():
    """
    Preprocess penguin data and train models
    Returns processed dataframe with all results
    """
    global _processed_data, _ml_results
    
    if _processed_data is not None:
        return _processed_data, _ml_results
    
    # Load and clean data
    penguins_data = sns.load_dataset('penguins')
    df_clean = penguins_data.dropna().reset_index(drop=True)
    
    # Features and target
    feature_columns = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']
    X = df_clean[feature_columns].copy()
    
    # Encode species
    le_species = LabelEncoder()
    y = le_species.fit_transform(df_clean['species'])
    species_names = le_species.classes_
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # PCA for visualization
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    
    # K-Means clustering
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)
    
    # SVM classification
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42, stratify=y)
    svm_model = SVC(kernel='rbf', random_state=42, probability=True)
    svm_model.fit(X_train, y_train)
    svm_predictions = svm_model.predict(X_scaled)
    svm_probabilities = svm_model.predict_proba(X_scaled)
    
    # Create results dataframe
    results_df = pd.DataFrame({
        'bill_length_mm': df_clean['bill_length_mm'],
        'bill_depth_mm': df_clean['bill_depth_mm'],
        'flipper_length_mm': df_clean['flipper_length_mm'],
        'body_mass_g': df_clean['body_mass_g'],
        'species': df_clean['species'],
        'island': df_clean['island'],
        'sex': df_clean['sex'],
        'pca_1': X_pca[:, 0],
        'pca_2': X_pca[:, 1],
        'true_species_encoded': y,
        'kmeans_cluster': cluster_labels,
        'svm_prediction': svm_predictions,
        'svm_confidence': np.max(svm_probabilities, axis=1)
    })
    
    # Map clusters to species
    cluster_species_mapping = {}
    for cluster in range(3):
        cluster_mask = results_df['kmeans_cluster'] == cluster
        most_common_species = results_df.loc[cluster_mask, 'species'].mode()[0]
        cluster_species_mapping[cluster] = most_common_species
    
    results_df['kmeans_species'] = results_df['kmeans_cluster'].map(cluster_species_mapping)
    results_df['svm_species'] = le_species.inverse_transform(results_df['svm_prediction'])
    
    # Store ML results
    ml_results = {
        'kmeans_model': kmeans,
        'svm_model': svm_model,
        'scaler': scaler,
        'pca': pca,
        'le_species': le_species,
        'species_names': species_names,
        'cluster_mapping': cluster_species_mapping,
        'X_train': X_train, 'X_test': X_test, 'y_train': y_train, 'y_test': y_test,
        'silhouette_score': silhouette_score(X_scaled, cluster_labels),
        'svm_train_acc': svm_model.score(X_train, y_train),
        'svm_test_acc': svm_model.score(X_test, y_test)
    }
    
    _processed_data = results_df
    _ml_results = ml_results
    
    return results_df, ml_results

# Define consistent colors
SPECIES_COLORS = {'Adelie': '#FF6B6B', 'Chinstrap': '#4ECDC4', 'Gentoo': '#45B7D1'}

# ========================================
# INDIVIDUAL PLOT FUNCTIONS
# ========================================

def plot_true_species_distribution():
    """Plot 1: True species distribution using PCA"""
    df, ml_results = preprocess_penguin_data()
    
    fig = px.scatter(
        df, x='pca_1', y='pca_2', color='species',
        title='<b>🐧 True Species Distribution</b><br><span style="font-size:12px">Ground Truth via PCA</span>',
        color_discrete_map=SPECIES_COLORS,
        hover_data=['bill_length_mm', 'body_mass_g', 'island']
    )
    
    fig.update_traces(marker=dict(size=10, opacity=0.8, line=dict(width=1, color='white')))
    fig.update_layout(
        height=350,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor='rgba(248,249,250,0.8)'
    )
    
    return fig

def plot_kmeans_clustering():
    """Plot 2: K-Means clustering results"""
    df, ml_results = preprocess_penguin_data()
    
    # Create color mapping for clusters
    cluster_colors = [SPECIES_COLORS[ml_results['cluster_mapping'][i]] for i in range(3)]
    df['cluster_color'] = df['kmeans_cluster'].map({i: cluster_colors[i] for i in range(3)})
    
    fig = go.Figure()
    
    symbols = ['diamond', 'star', 'hexagon']
    for cluster in range(3):
        mask = df['kmeans_cluster'] == cluster
        species_name = ml_results['cluster_mapping'][cluster]
        fig.add_trace(go.Scatter(
            x=df.loc[mask, 'pca_1'],
            y=df.loc[mask, 'pca_2'],
            mode='markers',
            name=f'Cluster {cluster} ({species_name})',
            marker=dict(
                color=SPECIES_COLORS[species_name],
                size=12, opacity=0.8, symbol=symbols[cluster],
                line=dict(width=2, color='white')
            )
        ))
    
    fig.update_layout(
        title='<b>🎯 K-Means Clustering (n=3)</b><br><span style="font-size:12px">Unsupervised Learning</span>',
        xaxis_title='PCA Component 1', yaxis_title='PCA Component 2',
        height=350,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor='rgba(248,249,250,0.8)'
    )
    
    return fig

def plot_svm_classification():
    """Plot 3: SVM classification with confidence"""
    df, ml_results = preprocess_penguin_data()
    
    fig = go.Figure()
    
    for species in ml_results['species_names']:
        mask = df['svm_species'] == species
        confidence_sizes = df.loc[mask, 'svm_confidence'] * 20 + 5
        fig.add_trace(go.Scatter(
            x=df.loc[mask, 'pca_1'],
            y=df.loc[mask, 'pca_2'],
            mode='markers',
            name=f'{species}',
            marker=dict(
                color=SPECIES_COLORS[species],
                size=confidence_sizes, opacity=0.7, symbol='square',
                line=dict(width=1, color='white')
            ),
            customdata=df.loc[mask, 'svm_confidence'],
            hovertemplate=f'<b>{species}</b><br>Confidence: %{{customdata:.3f}}<br><extra></extra>'
        ))
    
    fig.update_layout(
        title='<b>🤖 SVM Classification</b><br><span style="font-size:12px">Marker size = confidence</span>',
        xaxis_title='PCA Component 1', yaxis_title='PCA Component 2',
        height=350,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor='rgba(248,249,250,0.8)'
    )
    
    return fig

def plot_performance_metrics():
    """Plot 4: Model performance comparison"""
    df, ml_results = preprocess_penguin_data()
    
    # Calculate accuracies
    kmeans_accuracy = (df['species'] == df['kmeans_species']).mean()
    svm_accuracy = (df['species'] == df['svm_species']).mean()
    
    metrics = {
        'K-Means<br>Silhouette': ml_results['silhouette_score'],
        'K-Means<br>Accuracy': kmeans_accuracy,
        'SVM<br>Train Acc': ml_results['svm_train_acc'],
        'SVM<br>Test Acc': ml_results['svm_test_acc']
    }
    
    colors = ['#FF9999', '#FFB366', '#66B2FF', '#99FF99']
    
    fig = go.Figure(data=[
        go.Bar(
            x=list(metrics.keys()), y=list(metrics.values()),
            marker=dict(color=colors, line=dict(color='white', width=2)),
            text=[f'<b>{v:.3f}</b>' for v in metrics.values()],
            textposition='outside'
        )
    ])
    
    fig.update_layout(
        title='<b>📊 Performance Metrics</b><br><span style="font-size:12px">Model Comparison</span>',
        yaxis_title='Score', yaxis=dict(range=[0, 1.1]),
        height=350,
        plot_bgcolor='rgba(248,249,250,0.8)'
    )
    
    return fig

def plot_confusion_matrix():
    """Plot 5: SVM Confusion Matrix"""
    df, ml_results = preprocess_penguin_data()
    
    # Create confusion matrix
    cm = confusion_matrix(df['species'], df['svm_species'])
    species_names = ml_results['species_names']
    
    fig = go.Figure(data=go.Heatmap(
        z=cm, x=species_names, y=species_names,
        colorscale='Blues', showscale=True,
        text=cm, texttemplate="%{text}", textfont={"size": 16}
    ))
    
    fig.update_layout(
        title='<b>🎯 SVM Confusion Matrix</b><br><span style="font-size:12px">Predicted vs Actual</span>',
        xaxis_title='Predicted Species', yaxis_title='Actual Species',
        height=350
    )
    
    return fig

def plot_feature_importance():
    """Plot 6: Feature correlation heatmap"""
    df, ml_results = preprocess_penguin_data()
    
    features = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']
    corr_matrix = df[features].corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=['Bill Length', 'Bill Depth', 'Flipper Length', 'Body Mass'],
        y=['Bill Length', 'Bill Depth', 'Flipper Length', 'Body Mass'],
        colorscale='RdBu', zmid=0,
        text=np.round(corr_matrix.values, 2),
        texttemplate="%{text}", textfont={"size": 12}
    ))
    
    fig.update_layout(
        title='<b>🔗 Feature Correlations</b><br><span style="font-size:12px">Pearson Correlation</span>',
        height=350
    )
    
    return fig

def plot_cluster_centers():
    """Plot 7: K-Means cluster centers"""
    df, ml_results = preprocess_penguin_data()
    
    # Get cluster centers in original scale
    cluster_centers = ml_results['scaler'].inverse_transform(ml_results['kmeans_model'].cluster_centers_)
    features = ['Bill Length', 'Bill Depth', 'Flipper Length', 'Body Mass']
    
    fig = go.Figure()
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    for i in range(3):
        species_name = ml_results['cluster_mapping'][i]
        fig.add_trace(go.Scatterpolar(
            r=cluster_centers[i],
            theta=features,
            fill='toself',
            name=f'Cluster {i} ({species_name})',
            line_color=colors[i],
            fillcolor=colors[i],
            opacity=0.6
        ))
    
    fig.update_layout(
        title='<b>🎯 Cluster Centers</b><br><span style="font-size:12px">Feature Profiles</span>',
        polar=dict(radialaxis=dict(visible=True, range=[0, None])),
        height=350
    )
    
    return fig

def plot_species_by_island():
    """Plot 8: Species distribution by island"""
    df, ml_results = preprocess_penguin_data()
    
    # Create count data
    island_species = df.groupby(['island', 'species']).size().reset_index(name='count')
    
    fig = px.bar(
        island_species, x='island', y='count', color='species',
        title='<b>🏝️ Species by Island</b><br><span style="font-size:12px">Geographic Distribution</span>',
        color_discrete_map=SPECIES_COLORS
    )
    
    fig.update_layout(
        height=350,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor='rgba(248,249,250,0.8)'
    )
    
    return fig

def plot_pca_explained_variance():
    """Plot 9: PCA explained variance"""
    df, ml_results = preprocess_penguin_data()
    
    # Calculate PCA with more components
    pca_full = PCA(random_state=42)
    pca_full.fit(ml_results['scaler'].transform(df[['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']]))
    
    explained_var = pca_full.explained_variance_ratio_
    cumulative_var = np.cumsum(explained_var)
    
    fig = go.Figure()
    fig.add_trace(go.Bar(x=[f'PC{i+1}' for i in range(len(explained_var))], y=explained_var, name='Individual', marker_color='lightblue'))
    fig.add_trace(go.Scatter(x=[f'PC{i+1}' for i in range(len(explained_var))], y=cumulative_var, mode='lines+markers', name='Cumulative', line=dict(color='red')))
    
    fig.update_layout(
        title='<b>📈 PCA Explained Variance</b><br><span style="font-size:12px">Component Analysis</span>',
        xaxis_title='Principal Components', yaxis_title='Explained Variance Ratio',
        height=350,
        plot_bgcolor='rgba(248,249,250,0.8)'
    )
    
    return fig

def plot_confidence_distribution():
    """Plot 10: SVM confidence distribution"""
    df, ml_results = preprocess_penguin_data()
    
    fig = px.histogram(
        df, x='svm_confidence', color='svm_species', nbins=20,
        title='<b>🎲 SVM Confidence Distribution</b><br><span style="font-size:12px">Prediction Certainty</span>',
        color_discrete_map=SPECIES_COLORS
    )
    
    fig.update_layout(
        height=350,
        xaxis_title='Confidence Score', yaxis_title='Count',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor='rgba(248,249,250,0.8)'
    )
    
    return fig

def plot_feature_distributions():
    """Plot 11: Feature distributions by species"""
    df, ml_results = preprocess_penguin_data()
    
    fig = px.box(
        df.melt(id_vars=['species'], value_vars=['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']),
        x='variable', y='value', color='species',
        title='<b>📏 Feature Distributions</b><br><span style="font-size:12px">Box Plots by Species</span>',
        color_discrete_map=SPECIES_COLORS
    )
    
    fig.update_layout(
        height=350,
        xaxis_title='Features', yaxis_title='Values',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor='rgba(248,249,250,0.8)'
    )
    
    return fig

def create_model_summary_dataframe():
    """Create a DataFrame with model summary statistics for @render.data_frame"""
    df, ml_results = preprocess_penguin_data()
    
    # Calculate various metrics
    kmeans_accuracy = (df['species'] == df['kmeans_species']).mean()
    svm_accuracy = (df['species'] == df['svm_species']).mean()
    
    # Create summary DataFrame
    summary_data = {
        'Metric': [
            'Dataset Size',
            'Features Used', 
            'PCA Variance Explained',
            'K-Means Silhouette Score',
            'K-Means Species Accuracy',
            'SVM Train Accuracy',
            'SVM Test Accuracy', 
            'SVM Full Dataset Accuracy'
        ],
        'Value': [
            f"{len(df):,}",
            "4",
            f"{ml_results['pca'].explained_variance_ratio_.sum():.3f}",
            f"{ml_results['silhouette_score']:.3f}",
            f"{kmeans_accuracy:.3f}",
            f"{ml_results['svm_train_acc']:.3f}",
            f"{ml_results['svm_test_acc']:.3f}",
            f"{svm_accuracy:.3f}"
        ],
        'Description': [
            'Total penguins after cleaning',
            'Bill length, depth, flipper, mass',
            'Variance captured in 2D projection',
            'Cluster separation quality (0-1)',
            'Accuracy vs true species labels',
            'Training set performance',
            'Holdout test performance',
            'Complete dataset accuracy'
        ]
    }
    
    return pd.DataFrame(summary_data)

def plot_model_summary():
    """Plot 12: Model summary statistics - DEPRECATED, use create_model_summary_dataframe() instead"""
    df, ml_results = preprocess_penguin_data()
    
    # Calculate various metrics
    kmeans_accuracy = (df['species'] == df['kmeans_species']).mean()
    svm_accuracy = (df['species'] == df['svm_species']).mean()
    
    # Create lists for table data
    metric_names = [
        'Dataset Size',
        'Features Used', 
        'PCA Variance Explained',
        'K-Means Silhouette Score',
        'K-Means Accuracy',
        'SVM Train Accuracy',
        'SVM Test Accuracy',
        'SVM Full Accuracy'
    ]
    
    metric_values = [
        str(len(df)),
        '4',
        f"{ml_results['pca'].explained_variance_ratio_.sum():.3f}",
        f"{ml_results['silhouette_score']:.3f}",
        f"{kmeans_accuracy:.3f}",
        f"{ml_results['svm_train_acc']:.3f}",
        f"{ml_results['svm_test_acc']:.3f}",
        f"{svm_accuracy:.3f}"
    ]
    
    fig = go.Figure(data=[go.Table(
        header=dict(
            values=['<b>Metric</b>', '<b>Value</b>'],
            fill_color='#4ECDC4',
            align=['left', 'center'],
            font=dict(size=14, color='white'),
            height=40
        ),
        cells=dict(
            values=[metric_names, metric_values],  # Two separate lists
            fill_color=['#f8f9fa', '#ffffff'],
            align=['left', 'center'],
            font=dict(size=12, color='black'),
            height=30
        )
    )])
    
    fig.update_layout(
        title='<b>📋 Model Summary</b><br><span style="font-size:12px">Key Statistics & Performance</span>',
        height=350,
        margin=dict(l=10, r=10, t=80, b=10)
    )
    
    return fig

# ========================================
# SHINY DASHBOARD INTEGRATION
# ========================================

def create_ml_dashboard_grid():
    """
    Example of how to use these functions in a Shiny dashboard
    Returns a dictionary of all plot functions for easy access
    """
    return {
        'true_species': plot_true_species_distribution,
        'kmeans_clustering': plot_kmeans_clustering,
        'svm_classification': plot_svm_classification,
        'performance_metrics': plot_performance_metrics,
        'confusion_matrix': plot_confusion_matrix,
        'feature_importance': plot_feature_importance,
        'cluster_centers': plot_cluster_centers,
        'species_by_island': plot_species_by_island,
        'pca_variance': plot_pca_explained_variance,
        'confidence_dist': plot_confidence_distribution,
        'feature_distributions': plot_feature_distributions,
        'model_summary': plot_model_summary
    }