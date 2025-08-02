import pandas as pd
import seaborn as sns
import plotly.express as px
import numpy as np
from faicons import icon_svg
from shiny import reactive
from shiny.express import input, render, ui
from shinywidgets import render_plotly

# Load penguins dataset from seaborn
penguins_data = sns.load_dataset('penguins')
penguins_data = penguins_data.dropna()

@reactive.calc
def filtered_penguins_data():
    """Filter penguins data based on all user inputs."""
    df = penguins_data.copy()
    
    # Filter by selected species
    try:
        selected_species = input.species_filter()
        if selected_species and len(selected_species) > 0:
            df = df[df['species'].isin(selected_species)]
    except:
        pass
    
    # Filter by selected islands
    try:
        selected_islands = input.island_filter()
        if selected_islands and len(selected_islands) > 0:
            df = df[df['island'].isin(selected_islands)]
    except:
        pass
    
    # Filter by selected sex
    try:
        selected_sex = input.sex_filter()
        if selected_sex and len(selected_sex) > 0:
            df = df[df['sex'].isin(selected_sex)]
    except:
        pass
    
    # Filter by bill length range
    try:
        bill_range = input.bill_length_range()
        if bill_range and len(bill_range) == 2:
            df = df[(df['bill_length_mm'] >= bill_range[0]) & 
                    (df['bill_length_mm'] <= bill_range[1])]
    except:
        pass
    
    # Filter by minimum body mass
    try:
        min_mass = input.min_body_mass()
        if min_mass is not None:
            df = df[df['body_mass_g'] >= min_mass]
    except:
        pass
    
    return df

@reactive.calc
def summary_stats():
    """Calculate summary statistics for filtered data."""
    try:
        df = filtered_penguins_data()
        if len(df) == 0:
            return {'count': 0, 'avg_bill': 0, 'avg_mass': 0, 'species_count': 0, 'island_count': 0}
        
        return {
            'count': len(df),
            'avg_bill': df['bill_length_mm'].mean(),
            'avg_mass': df['body_mass_g'].mean(),
            'species_count': df['species'].nunique(),
            'island_count': df['island'].nunique()
        }
    except Exception as e:
        # Return safe defaults if there's any error
        return {'count': 0, 'avg_bill': 0, 'avg_mass': 0, 'species_count': 0, 'island_count': 0}

# # Page setup
# ui.page_opts(title="Teja's - Palmer Penguins Interactive Analysis Dashboard")

# Sidebar with filter controls
with ui.sidebar():
    ui.h3("Filter Controls")
    ui.hr()
    
    ui.input_checkbox_group(
        "species_filter",
        "Select Penguin Species:",
        choices=list(penguins_data['species'].unique()),
        selected=list(penguins_data['species'].unique())
    )
    
    ui.hr()
    
    ui.input_checkbox_group(
        "island_filter",
        "Select Islands:",
        choices=list(penguins_data['island'].unique()),
        selected=list(penguins_data['island'].unique())
    )
    
    ui.hr()
    
    ui.input_checkbox_group(
        "sex_filter",
        "Select Sex:",
        choices=list(penguins_data['sex'].unique()),
        selected=list(penguins_data['sex'].unique())
    )
    
    ui.hr()
    
    ui.input_slider(
        "bill_length_range",
        "Bill Length Range (mm):",
        min=float(penguins_data['bill_length_mm'].min()),
        max=float(penguins_data['bill_length_mm'].max()),
        value=[float(penguins_data['bill_length_mm'].min()),
               float(penguins_data['bill_length_mm'].max())],
        step=0.5
    )
    
    ui.input_slider(
        "min_body_mass",
        "Minimum Body Mass (g):",
        min=int(penguins_data['body_mass_g'].min()),
        max=int(penguins_data['body_mass_g'].max()),
        value=int(penguins_data['body_mass_g'].min()),
        step=50
    )

# Summary statistics cards
with ui.layout_columns(col_widths=[12]):
    with ui.card():
        ui.card_header("🐧 PENGUIN ANALYSIS INSIGHTS", style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; text-align: center; font-weight: 900; font-size: 28px; text-shadow: 2px 2px 4px rgba(0,0,0,0.3); letter-spacing: 2px; padding: 25px;")
        
        # Dynamic Summary Section
        ui.div("📊 Dynamic Summary", style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; font-weight: bold; padding: 6px; text-align: center; font-size: 16px; margin: 5px 0; border-radius: 8px;")
        
        with ui.layout_columns(gap="5px"):
            @render.ui
            def total_penguins_box():
                stats = summary_stats()
                count = stats['count']
                if count > 300:
                    theme = "bg-gradient-blue-purple"
                    icon_name = "users"
                elif count > 200:
                    theme = "bg-gradient-indigo-purple"
                    icon_name = "user-group"
                else:
                    theme = "bg-gradient-purple-pink"
                    icon_name = "user"
                return ui.value_box(
                    title="Total Penguins",
                    value=f"{count:,}",
                    showcase=icon_svg(icon_name),
                    theme=theme,
                    height="150px"
                )
            
            @render.ui
            def species_diversity_box():
                stats = summary_stats()
                species_count = stats['species_count']
                if species_count >= 3:
                    status = "High"
                    theme = "bg-gradient-green-blue"
                    icon_name = "leaf"
                elif species_count == 2:
                    status = "Medium"
                    theme = "bg-gradient-yellow-orange"
                    icon_name = "seedling"
                else:
                    status = "Low"
                    theme = "bg-gradient-red-orange"
                    icon_name = "tree"
                return ui.value_box(
                    title="🧬 Species Diversity",
                    value=f"{species_count}",
                    showcase=icon_svg(icon_name),
                    theme=theme,
                    height="150px"
                )
            
            @render.ui
            def bill_length_box():
                stats = summary_stats()
                avg_bill = stats['avg_bill']
                if avg_bill > 45:
                    status = "Large"
                    theme = "bg-gradient-orange-red"
                    icon_name = "arrow-up"
                elif avg_bill > 40:
                    status = "Medium"
                    theme = "bg-gradient-indigo-blue"
                    icon_name = "minus"
                else:
                    status = "Small"
                    theme = "bg-gradient-blue-cyan"
                    icon_name = "arrow-down"
                return ui.value_box(
                    title="Avg Bill Length",
                    value=f"{avg_bill:.1f}mm" if avg_bill > 0 else "0mm",
                    showcase=icon_svg(icon_name),
                    theme=theme,
                    height="150px"
                )
            
            @render.ui
            def body_mass_box():
                stats = summary_stats()
                avg_mass = stats['avg_mass']
                if avg_mass > 4500:
                    status = "Heavy"
                    theme = "bg-gradient-red-pink"
                    icon_name = "star"
                elif avg_mass > 3500:
                    status = "Medium"
                    theme = "bg-gradient-purple-blue"
                    icon_name = "circle"
                else:
                    status = "Light"
                    theme = "bg-gradient-cyan-blue"
                    icon_name = "feather"
                return ui.value_box(
                    title="Avg Body Mass",
                    value=f"{avg_mass:.0f}g" if avg_mass > 0 else "0g",
                    showcase=icon_svg(icon_name),
                    theme=theme,
                    height="150px"
                )
            
            ui.value_box(
                title="Study Location",
                value="Antarctica",
                showcase=icon_svg("map-pin"),
                theme="bg-gradient-purple-pink",
                height="150px"
            )
      
        # Dataset Overview Section
        ui.div("📈 Detailed Insights", style="background: linear-gradient(135deg, #48c6ef 0%, #6f86d6 100%); color: white; font-weight: bold; padding: 8px; text-align: center; font-size: 16px; margin: 8px 0; border-radius: 8px;")
    
        # Key Statistics Section
        with ui.layout_columns(col_widths=[6, 6], gap="10px"):
            @render.ui
            def top_performers():
                df = filtered_penguins_data()
                if len(df) == 0:
                    return ui.div("No data available")
                
                species_counts = df['species'].value_counts()
                most_common_species = species_counts.index[0]
                island_counts = df['island'].value_counts()
                most_common_island = island_counts.index[0]
                
                return ui.div(
                    ui.div("🏆 Top Performers", style="background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%); color: #333; font-weight: bold; padding: 6px; border-radius: 8px 8px 0 0; text-align: center; font-size: 14px;"),
                    ui.div(
                        ui.p(f"🥇 Most Common Species: {most_common_species}", style="margin: 5px 0; font-size: 13px;"),
                        ui.p(f"📍 Most Common Island: {most_common_island}", style="margin: 5px 0; font-size: 13px;"),
                        ui.p(f"👥 Count: {species_counts[most_common_species]} penguins", style="margin: 5px 0; font-size: 13px;"),
                        style="padding: 8px; background: white; border: 1px solid #ddd; border-radius: 0 0 8px 8px;"
                    )
                )
            
            @render.ui
            def measurement_ranges():
                df = filtered_penguins_data()
                if len(df) == 0:
                    return ui.div("No data available")
                
                return ui.div(
                    ui.div("📏 Measurement Ranges", style="background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%); color: #333; font-weight: bold; padding: 6px; border-radius: 8px 8px 0 0; text-align: center; font-size: 14px;"),
                    ui.div(
                        ui.p(f"⚖️ Body Mass: {df['body_mass_g'].min():.0f}g - {df['body_mass_g'].max():.0f}g", style="margin: 5px 0; font-size: 13px;"),
                        ui.p(f"📐 Bill Length: {df['bill_length_mm'].min():.1f}mm - {df['bill_length_mm'].max():.1f}mm", style="margin: 5px 0; font-size: 13px;"),
                        ui.p(f"🏊 Flipper Length: {df['flipper_length_mm'].min():.1f}mm - {df['flipper_length_mm'].max():.1f}mm", style="margin: 5px 0; font-size: 13px;"),
                        style="padding: 8px; background: white; border: 1px solid #ddd; border-radius: 0 0 8px 8px;"
                    )
                )

        # Record Holders Section  
        with ui.layout_columns(col_widths=[4, 4, 4], gap="8px"):
            @render.ui
            def heaviest_penguin():
                df = filtered_penguins_data()
                if len(df) == 0:
                    return ui.div("No data available")
                
                heaviest = df.loc[df['body_mass_g'].idxmax()]
                
                return ui.value_box(
                    title="🏋️ Heaviest Penguin",
                    value=f"{heaviest['body_mass_g']:.0f}g",
                    showcase=icon_svg("star"),
                    theme="bg-gradient-red-pink",
                    height="150px"
                )
            
            @render.ui
            def longest_bill():
                df = filtered_penguins_data()
                if len(df) == 0:
                    return ui.div("No data available")
                
                longest_bill = df.loc[df['bill_length_mm'].idxmax()]
                
                return ui.value_box(
                    title="📏 Longest Bill",
                    value=f"{longest_bill['bill_length_mm']:.1f}mm",
                    showcase=icon_svg("arrow-up"),
                    theme="bg-gradient-indigo-purple",
                    height="150px"
                )
            
            @render.ui
            def longest_flipper():
                df = filtered_penguins_data()
                if len(df) == 0:
                    return ui.div("No data available")
                
                longest_flipper = df.loc[df['flipper_length_mm'].idxmax()]
                
                return ui.value_box(
                    title="🏊 Longest Flipper", 
                    value=f"{longest_flipper['flipper_length_mm']:.1f}mm",
                    showcase=icon_svg("arrow-up"),
                    theme="bg-gradient-cyan-blue",
                    height="150px"
                )
        
        # Sex Distribution Section
        @render.ui
        def sex_distribution():
            df = filtered_penguins_data()
            if len(df) == 0:
                return ui.div("No data available")
            
            sex_counts = df['sex'].value_counts()
            
            return ui.div(
                ui.div("👥 Sex Distribution", style="background: linear-gradient(135deg, #d299c2 0%, #fef9d7 100%); color: #333; font-weight: bold; padding: 6px; border-radius: 8px 8px 0 0; text-align: center; font-size: 14px;"),
                ui.div(
                    *[ui.p(f"{sex}: {count} penguins ({count/len(df)*100:.1f}%)", 
                           style="margin: 4px 0; font-size: 13px; text-align: center;") 
                      for sex, count in sex_counts.items()],
                    style="padding: 8px; background: white; border: 1px solid #ddd; border-radius: 0 0 8px 8px;"
                )
            )

ui.hr(style="margin-top: 10px; margin-bottom: 10px;")

# Main charts
ui.div("📈 Penguin Charts", style="background: linear-gradient(135deg, #48c6ef 0%, #6f86d6 100%); color: white; font-weight: bold; padding: 8px; text-align: center; font-size: 16px; margin: 8px 0; border-radius: 8px;")                   
with ui.layout_columns():
    with ui.card():
        ui.card_header("Bill Length vs Body Mass by Species")
        
        @render_plotly
        def bill_mass_scatter():
            df = filtered_penguins_data()
            
            if len(df) == 0:
                fig = px.scatter(title="No penguins match current filters")
                fig.update_layout(height=500)
                return fig
            
            fig = px.scatter(
                df,
                x='body_mass_g',
                y='bill_length_mm',
                color='species',
                size='flipper_length_mm',
                hover_data=['island', 'sex', 'bill_depth_mm'],
                title='Penguin Measurements by Species',
                labels={
                    'body_mass_g': 'Body Mass (g)',
                    'bill_length_mm': 'Bill Length (mm)',
                    'flipper_length_mm': 'Flipper Length (mm)'
                }
            )
            
            fig.update_layout(
                height=500, 
                showlegend=True,
                legend=dict(
                    x=0.02,  
                    y=0.98,  
                    xanchor='left',
                    yanchor='top',
                    bgcolor='rgba(255, 255, 255, 0.8)',  
                    bordercolor='rgba(0, 0, 0, 0.2)',    
                    borderwidth=1
                )
            )
            return fig
    
    with ui.card():
        ui.card_header("Species Distribution by Island")
        
        @render_plotly
        def species_island_bar():
            df = filtered_penguins_data()
            
            if len(df) == 0:
                fig = px.bar(title="No penguins match current filters")
                fig.update_layout(height=500)
                return fig
            
            species_island = pd.crosstab(df['island'], df['species']).reset_index()
            species_island_melted = species_island.melt(
                id_vars='island', 
                var_name='species', 
                value_name='count'
            )
            
            fig = px.bar(
                species_island_melted,
                x='island',
                y='count',
                color='species',
                title='Penguin Species Count by Island',
                labels={'count': 'Number of Penguins', 'island': 'Island', 'species': 'Species'}
            )
            
            fig.update_layout(
                height=500, 
                showlegend=True,
                legend=dict(
                    x=0.90,  
                    y=0.98,  
                    xanchor='right',
                    yanchor='top',
                    bgcolor='rgba(255, 255, 255, 0.8)',  
                    bordercolor='rgba(0, 0, 0, 0.2)',    
                    borderwidth=1
                )
            )
            return fig

ui.hr(style="margin-top: 10px; margin-bottom: 10px;")

# Secondary charts
with ui.layout_columns():
    
    with ui.card():
        ui.card_header("Bill Length vs Bill Depth")
        
        @render_plotly
        def bill_dimensions():
            df = filtered_penguins_data()
            
            if len(df) == 0:
                fig = px.scatter(title="No penguins match current filters")
                fig.update_layout(height=500)
                return fig
            
            fig = px.scatter(
                df,
                x='bill_length_mm',
                y='bill_depth_mm',
                color='species',
                hover_data=['island', 'sex'],
                title='Bill Dimensions by Species',
                labels={'bill_length_mm': 'Bill Length (mm)', 'bill_depth_mm': 'Bill Depth (mm)'}
            )
            
            fig.update_layout(
                height=500, 
                showlegend=True,
                legend=dict(
                    x=0.02, 
                    y=0.98, 
                    xanchor='left',
                    yanchor='top',
                    bgcolor='rgba(255, 255, 255, 0.8)',  
                    bordercolor='rgba(0, 0, 0, 0.2)',    
                    borderwidth=1
                )
            )
            return fig
    
    with ui.card():
        ui.card_header("Body Mass Distribution")
        
        @render_plotly
        def mass_histogram():
            df = filtered_penguins_data()
            
            if len(df) == 0:
                fig = px.histogram(title="No penguins match current filters")
                fig.update_layout(height=500)
                return fig
            
            fig = px.histogram(
                df,
                x='body_mass_g',
                color='species',
                title='Body Mass Distribution by Species',
                labels={'body_mass_g': 'Body Mass (g)', 'count': 'Number of Penguins'},
                nbins=20
            )
            
            fig.update_layout(
                height=500, 
                showlegend=True,
                legend=dict(
                    x=0.90, 
                    y=0.98,  
                    xanchor='right',
                    yanchor='top',
                    bgcolor='rgba(255, 255, 255, 0.8)',  
                    bordercolor='rgba(0, 0, 0, 0.2)',    
                    borderwidth=1
                )
            )
            return fig
        
ui.hr(style="margin-top: 10px; margin-bottom: 10px;")

# Data table
ui.div("🐧 Filtered Penguins Dataset", style="background: linear-gradient(135deg, #48c6ef 0%, #6f86d6 100%); color: white; font-weight: bold; padding: 8px; text-align: center; font-size: 16px; margin: 8px 0; border-radius: 8px;")                   
with ui.layout_columns():
    with ui.card():
        # ui.card_header("🐧 Filtered Penguins Dataset", )
        
        @render.data_frame
        def penguins_data_table():
            df = filtered_penguins_data()
            df_display = df.copy()
            numeric_cols = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm']
            for col in numeric_cols:
                if col in df_display.columns:
                    df_display[col] = df_display[col].round(1)
            return render.DataGrid(df_display, width="100%")

ui.hr(style="margin-top: 10px; margin-bottom: 10px;")

# Example 4x4 grid in your Shiny dashboard:
from ml import *

with ui.card():
    ui.card_header("🐧 Complete Penguin ML Analysis Dashboard", style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; text-align: center; font-weight: 900; font-size: 28px; text-shadow: 2px 2px 4px rgba(0,0,0,0.3); letter-spacing: 2px; padding: 25px;")

# Row 1: Core ML Comparison
with ui.layout_columns(col_widths=[6, 6]):
    with ui.card():
        ui.card_header("🐧 True Species Distribution")
        @render_plotly
        def true_species_plot():
            return plot_true_species_distribution()
    
    with ui.card():
        ui.card_header("🎯 K-Means Clustering Results")
        @render_plotly
        def kmeans_plot():
            return plot_kmeans_clustering()

# Row 2: SVM Analysis
with ui.layout_columns(col_widths=[6, 6]):
    with ui.card():
        ui.card_header("🤖 SVM Classification Predictions")
        @render_plotly
        def svm_plot():
            return plot_svm_classification()
    
    with ui.card():
        ui.card_header("🎯 SVM Confusion Matrix")
        @render_plotly
        def confusion_plot():
            return plot_confusion_matrix()

# Row 3: Performance & Feature Analysis
with ui.layout_columns(col_widths=[6, 6]):
    with ui.card():
        ui.card_header("📊 Model Performance Comparison")
        @render_plotly
        def performance_plot():
            return plot_performance_metrics()
    
    with ui.card():
        ui.card_header("🔗 Feature Correlation Matrix")
        @render_plotly
        def correlation_plot():
            return plot_feature_importance()

# Row 4: Cluster & Geographic Analysis
with ui.layout_columns(col_widths=[6, 6]):
    with ui.card():
        ui.card_header("⭐ K-Means Cluster Centers")
        @render_plotly
        def cluster_centers_plot():
            return plot_cluster_centers()
    
    with ui.card():
        ui.card_header("🏝️ Species Distribution by Island")
        @render_plotly
        def island_plot():
            return plot_species_by_island()

# Row 5: Advanced Statistical Analysis
with ui.layout_columns(col_widths=[6, 6]):
    with ui.card():
        ui.card_header("📈 PCA Explained Variance")
        @render_plotly
        def pca_variance_plot():
            return plot_pca_explained_variance()
    
    with ui.card():
        ui.card_header("🎲 SVM Confidence Distribution")
        @render_plotly
        def confidence_plot():
            return plot_confidence_distribution()

# Row 6: Summary & Feature Distributions
with ui.layout_columns(col_widths=[6, 6]):
    with ui.card():
        ui.card_header("📏 Feature Distributions by Species")
        @render_plotly
        def feature_dist_plot():
            return plot_feature_distributions()
    
    with ui.card():
        ui.card_header("📋 Model Summary Statistics")
        @render.data_frame           
        def summary_table():         
            return render.DataGrid(create_model_summary_dataframe(), width="100%")