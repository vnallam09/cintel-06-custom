# Palmer Penguins Interactive Dashboard - Module 6

An interactive web dashboard built with PyShiny for exploring Antarctic penguin research data from the Palmer Station.

## About the Dataset

This dashboard analyzes the Palmer Penguins dataset, which contains measurements of three penguin species (Adelie, Chinstrap, and Gentoo) collected from three islands in the Palmer Archipelago, Antarctica. The dataset includes physical measurements like bill dimensions, flipper length, and body mass.

**Data Source:** [Palmer Penguins Dataset](https://github.com/mwaskom/seaborn-data/blob/master/penguins.csv)

## Features

### Interactive Filtering
- **Species Selection**: Filter by penguin species (Adelie, Chinstrap, Gentoo)
- **Island Selection**: Filter by research location (Torgersen, Biscoe, Dream)
- **Sex Selection**: Filter by penguin sex (male, female)
- **Bill Length Range**: Slider to set minimum and maximum bill length
- **Body Mass Threshold**: Set minimum body mass for analysis

### Visualizations
1. **Bill Length vs Body Mass Scatter Plot**: Interactive plot showing relationship between measurements, colored by species
2. **Species Distribution by Island**: Stacked bar chart showing penguin populations across islands
3. **Bill Dimensions Analysis**: Scatter plot comparing bill length and depth
4. **Body Mass Distribution**: Histogram showing mass distribution by species
5. **Interactive Data Table**: Sortable and filterable table of all penguin measurements

### Summary Statistics
- Real-time cards showing:
  - Total penguin count
  - Number of species represented
  - Number of islands represented
  - Average bill length
  - Average body mass

### Data Insights
- Automatically generated insights about filtered data
- Record holders (heaviest penguin, longest bill, etc.)
- Sex distribution statistics
- Species and island breakdowns

## Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Quick Start
1. Clone this repository:
   ```bash
   git clone [your-repo-url]
   cd module6-penguins-dashboard
   ```

2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the dashboard:
   ```bash
   shiny run app.py --reload --launch-browser
   ```


## Requirements

See `requirements.txt` for full dependency list:
- `shiny` - Web application framework
- `pandas` - Data manipulation
- `seaborn` - Dataset and statistical plotting
- `plotly` - Interactive visualizations
- `shinywidgets` - Enhanced UI components
- `numpy` - Numerical computing

## Architecture

### Reactive Programming
The app uses PyShiny's reactive programming model:
- `filtered_penguins_data()`: Main reactive calc that filters data based on all user inputs
- `summary_stats()`: Reactive calc for computing summary statistics
- All outputs automatically update when filters change

### UI Components
- **Sidebar**: Contains all filter controls using checkbox groups and sliders
- **Main Content**: Organized using layout columns and cards
- **Summary Cards**: Display key metrics at the top
- **Charts**: Four interactive Plotly visualizations
- **Data Table**: Filterable table showing raw data
- **Insights**: Text analysis with dynamic statistics

## Data Processing

The app handles data processing through several steps:
1. Load penguins dataset from seaborn
2. Remove rows with missing values for clean analysis  
3. Apply user-selected filters reactively
4. Calculate summary statistics
5. Generate visualizations and insights


---
