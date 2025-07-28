# app.py - Module 6 Interactive Palmer Penguins Analysis Dashboard
# Dataset: Palmer Penguins (Antarctic penguin species measurements)

import pandas as pd
import seaborn as sns
import plotly.express as px
import numpy as np
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

# Page setup
ui.page_opts(
    title="Palmer Penguins Interactive Analysis - Module 6",
    fillable=True
)

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
with ui.layout_columns(col_widths=[2, 2, 2, 3, 3]):
    
    with ui.card():
        ui.card_header("Total Penguins")
        @render.text
        def total_count():
            return f"{summary_stats()['count']}"
    
    with ui.card():
        ui.card_header("Species")
        @render.text
        def species_count_display():
            return f"{summary_stats()['species_count']}"
    
    with ui.card():
        ui.card_header("Islands")
        @render.text
        def island_count_display():
            return f"{summary_stats()['island_count']}"
    
    with ui.card():
        ui.card_header("Avg Bill Length")
        @render.text
        def avg_bill_display():
            stats = summary_stats()
            return f"{stats['avg_bill']:.1f} mm" if stats['avg_bill'] > 0 else "0 mm"
    
    with ui.card():
        ui.card_header("Avg Body Mass")
        @render.text
        def avg_mass_display():
            stats = summary_stats()
            return f"{stats['avg_mass']:.0f} g" if stats['avg_mass'] > 0 else "0 g"

# Main charts
with ui.layout_columns(col_widths=[6, 6]):
    
    with ui.card():
        ui.card_header("Bill Length vs Body Mass by Species")
        
        @render_plotly
        def bill_mass_scatter():
            df = filtered_penguins_data()
            
            if len(df) == 0:
                fig = px.scatter(title="No penguins match current filters")
                fig.update_layout(height=400)
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
            
            fig.update_layout(height=400, showlegend=True)
            return fig
    
    with ui.card():
        ui.card_header("Species Distribution by Island")
        
        @render_plotly
        def species_island_bar():
            df = filtered_penguins_data()
            
            if len(df) == 0:
                fig = px.bar(title="No penguins match current filters")
                fig.update_layout(height=400)
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
            
            fig.update_layout(height=400, showlegend=True)
            return fig

# Secondary charts
with ui.layout_columns(col_widths=[6, 6]):
    
    with ui.card():
        ui.card_header("Bill Length vs Bill Depth")
        
        @render_plotly
        def bill_dimensions():
            df = filtered_penguins_data()
            
            if len(df) == 0:
                fig = px.scatter(title="No penguins match current filters")
                fig.update_layout(height=350)
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
            
            fig.update_layout(height=350, showlegend=True)
            return fig
    
    with ui.card():
        ui.card_header("Body Mass Distribution")
        
        @render_plotly
        def mass_histogram():
            df = filtered_penguins_data()
            
            if len(df) == 0:
                fig = px.histogram(title="No penguins match current filters")
                fig.update_layout(height=350)
                return fig
            
            fig = px.histogram(
                df,
                x='body_mass_g',
                color='species',
                title='Body Mass Distribution by Species',
                labels={'body_mass_g': 'Body Mass (g)', 'count': 'Number of Penguins'},
                nbins=20
            )
            
            fig.update_layout(height=350, showlegend=True)
            return fig

# Data table
with ui.layout_columns(col_widths=[12]):
    with ui.card():
        ui.card_header("🐧 Filtered Penguins Dataset")
        
        @render.data_frame
        def penguins_data_table():
            df = filtered_penguins_data()
            df_display = df.copy()
            numeric_cols = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm']
            for col in numeric_cols:
                if col in df_display.columns:
                    df_display[col] = df_display[col].round(1)
            return df_display

# Insights section
with ui.layout_columns(col_widths=[12]):
    with ui.card():
        ui.card_header("Penguin Analysis Insights")
        
        @render.text
        def penguin_insights():
            df = filtered_penguins_data()
            
            if len(df) == 0:
                return "No penguins match the current filter criteria. Try adjusting the filters above."
            
            heaviest = df.loc[df['body_mass_g'].idxmax()]
            longest_bill = df.loc[df['bill_length_mm'].idxmax()]
            longest_flipper = df.loc[df['flipper_length_mm'].idxmax()]
            
            species_counts = df['species'].value_counts()
            most_common_species = species_counts.index[0]
            
            island_counts = df['island'].value_counts()
            most_common_island = island_counts.index[0]
            
            insights = f"""
   Dataset Summary:
• Total penguins analyzed: {len(df)} out of {len(penguins_data)} total in dataset
• Species represented: {', '.join(df['species'].unique())}
• Islands represented: {', '.join(df['island'].unique())}

   Key Statistics:
• Most common species: {most_common_species} ({species_counts[most_common_species]} penguins)
• Most common island: {most_common_island} ({island_counts[most_common_island]} penguins)
• Body mass range: {df['body_mass_g'].min():.0f}g - {df['body_mass_g'].max():.0f}g
• Bill length range: {df['bill_length_mm'].min():.1f}mm - {df['bill_length_mm'].max():.1f}mm

  Record Holders:
• Heaviest penguin: {heaviest['species']} from {heaviest['island']} ({heaviest['body_mass_g']:.0f}g)
• Longest bill: {longest_bill['species']} from {longest_bill['island']} ({longest_bill['bill_length_mm']:.1f}mm)
• Longest flipper: {longest_flipper['species']} from {longest_flipper['island']} ({longest_flipper['flipper_length_mm']:.1f}mm)

   Sex Distribution: {dict(df['sex'].value_counts())}
            """
            
            return insights
