import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CSV = os.path.join(BASE_DIR, "data", "khadakwasla_evaporation_dataset.csv")
DEFAULT_OUTPUT_DIR = os.path.join(BASE_DIR, "data", "eda_plots")

def run_eda(csv_path=DEFAULT_CSV, output_dir=DEFAULT_OUTPUT_DIR):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['month_name'] = df['date'].dt.strftime('%b')
    
    print("=== DATASET EDA SUMMARY ===")
    print(f"Total Daily Observations: {len(df)}")
    print(f"Date Range: {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")
    print("\nDescriptive Statistics:")
    print(df[['temp_mean_c', 'humidity_pct', 'wind_speed_2m_ms', 'solar_radiation_mj', 'evaporation_pan_mm', 'penman_evaporation_mm']].describe().round(2))
    
    # 1. Correlation Matrix Heatmap
    plt.figure(figsize=(10, 8))
    cols_for_corr = ['temp_mean_c', 'temp_max_c', 'temp_min_c', 'humidity_pct', 'wind_speed_2m_ms', 
                     'solar_radiation_mj', 'sunshine_hours', 'pressure_hpa', 'vpd_kpa', 'evaporation_pan_mm']
    corr = df[cols_for_corr].corr()
    
    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1.0, vmin=-1.0, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .8}, annot=True, fmt=".2f")
    plt.title("Correlation Matrix: Weather Parameters vs Evaporation Pan Loss", fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "correlation_matrix.png"), dpi=300)
    plt.close()
    
    # 2. 4-Panel Weather Drivers vs Evaporation Scatter Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Impact of Primary Weather Drivers on Reservoir Evaporation Loss (mm/day)", fontsize=16, fontweight='bold')
    
    sns.regplot(ax=axes[0, 0], data=df, x='temp_mean_c', y='evaporation_pan_mm',
                scatter_kws={'alpha':0.4, 'color':'#e74c3c'}, line_kws={'color':'#900c3f', 'linewidth':2})
    axes[0, 0].set_title("Temperature (°C) vs Evaporation", fontweight='bold')
    axes[0, 0].set_xlabel("Mean Temperature (°C)")
    axes[0, 0].set_ylabel("Evaporation Pan (mm/day)")
    
    sns.regplot(ax=axes[0, 1], data=df, x='solar_radiation_mj', y='evaporation_pan_mm',
                scatter_kws={'alpha':0.4, 'color':'#f39c12'}, line_kws={'color':'#d35400', 'linewidth':2})
    axes[0, 1].set_title("Solar Radiation (MJ/m²/day) vs Evaporation", fontweight='bold')
    axes[0, 1].set_xlabel("Solar Radiation (MJ/m²/day)")
    axes[0, 1].set_ylabel("Evaporation Pan (mm/day)")
    
    sns.regplot(ax=axes[1, 0], data=df, x='humidity_pct', y='evaporation_pan_mm',
                scatter_kws={'alpha':0.4, 'color':'#3498db'}, line_kws={'color':'#1f618d', 'linewidth':2})
    axes[1, 0].set_title("Relative Humidity (%) vs Evaporation (Inverse Relationship)", fontweight='bold')
    axes[1, 0].set_xlabel("Relative Humidity (%)")
    axes[1, 0].set_ylabel("Evaporation Pan (mm/day)")
    
    sns.regplot(ax=axes[1, 1], data=df, x='wind_speed_2m_ms', y='evaporation_pan_mm',
                scatter_kws={'alpha':0.4, 'color':'#2ecc71'}, line_kws={'color':'#117864', 'linewidth':2})
    axes[1, 1].set_title("Wind Speed (m/s) vs Evaporation", fontweight='bold')
    axes[1, 1].set_xlabel("Wind Speed at 2m (m/s)")
    axes[1, 1].set_ylabel("Evaporation Pan (mm/day)")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "evaporation_vs_weather.png"), dpi=300)
    plt.close()
    
    # 3. Seasonal & Monthly Evaporation Trends Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Full Time Series
    ax1.plot(df['date'], df['evaporation_pan_mm'], label='Pan Evaporation (mm/day)', color='#e67e22', alpha=0.7, linewidth=1)
    ax1.plot(df['date'], df['penman_evaporation_mm'], label='FAO-56 Penman Monteith (mm/day)', color='#2980b9', alpha=0.8, linewidth=1.2)
    ax1.set_title("Temporal Daily Evaporation Trends (2021 - 2024)", fontsize=14, fontweight='bold')
    ax1.set_ylabel("Evaporation Depth (mm/day)")
    ax1.legend(loc='upper right')
    
    # Monthly Boxplot
    month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    sns.boxplot(ax=ax2, data=df, x='month_name', y='evaporation_pan_mm', order=month_order, palette='YlOrRd')
    ax2.set_title("Monthly Evaporation Pattern (Peak Summer Loss in Apr-May vs Monsoon Dip in Jul-Aug)", fontsize=14, fontweight='bold')
    ax2.set_xlabel("Month")
    ax2.set_ylabel("Evaporation Pan (mm/day)")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "seasonal_evaporation_trend.png"), dpi=300)
    plt.close()
    
    # 4. Feature Distributions Plot
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("Distribution of Key Meteorological Variables & Target Evaporation", fontsize=15, fontweight='bold')
    
    variables = [
        ('temp_mean_c', 'Temperature (°C)', '#e74c3c'),
        ('humidity_pct', 'Relative Humidity (%)', '#3498db'),
        ('wind_speed_2m_ms', 'Wind Speed (m/s)', '#2ecc71'),
        ('solar_radiation_mj', 'Solar Radiation (MJ/m²)', '#f39c12'),
        ('vpd_kpa', 'Vapor Pressure Deficit (kPa)', '#9b59b6'),
        ('evaporation_pan_mm', 'Pan Evaporation (mm/day)', '#e67e22')
    ]
    
    for idx, (var, title, color) in enumerate(variables):
        row, col = idx // 3, idx % 3
        sns.histplot(ax=axes[row, col], data=df, x=var, kde=True, color=color, bins=25)
        axes[row, col].set_title(title, fontweight='bold')
        axes[row, col].set_xlabel("")
        
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "feature_distributions.png"), dpi=300)
    plt.close()
    
    print(f"\nSuccessfully generated and saved all EDA diagnostic plots to: {output_dir}")

if __name__ == "__main__":
    run_eda()
