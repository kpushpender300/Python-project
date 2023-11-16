import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np



def load_data(file_path):
    # Load the Excel data into a DataFrame
    df = pd.read_csv(file_path)
    return df


def print_clean_data_info(df):
    """
    Print information about the DataFrame and clean the data by handling missing values.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.

    Returns:
    - pd.DataFrame: Cleaned DataFrame.
    """
    print(df.shape)
    print(df.info())

    df[['pm2.5', 'pm10', 'o3', 'no2', 'so2', 'co']] = df[['pm2.5', 'pm10', 'o3', 'no2', 'so2', 'co']].apply(pd.to_numeric, errors='coerce')#'coerce' converts non-numeric values to NaN.
    print(df[['pm2.5', 'pm10', 'o3', 'no2', 'so2', 'co']].dtypes)

    df.dropna(inplace=True)      # Clean data: handling missing values
    return df


def set_date_as_index(df):
    # Set 'Date' as the index
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
    df.set_index('Date', inplace=True)
    return df


def plot_monthly_mean_trend(df):
    """
    Plot the monthly mean trends of different pollutants.

    Parameters:
    - df (pd.DataFrame): Input DataFrame with datetime index and pollutant columns.
    """
    # Resample data by month and calculate the mean
    monthly_mean = df.resample('M').mean()

    # Plotting trends for each pollutant in separate figures
    pollutants = ['pm2.5', 'pm10', 'o3', 'no2', 'so2', 'co']

    for i, pollutant in enumerate(pollutants, 1):
        if i % 2 == 1:
            plt.figure(figsize=(17, 6))  # Adjust the figure size as needed

        plt.subplot(2, 1, i % 2 + 1)  #2 row, 1 columns, ith subplot
        plt.plot(monthly_mean.index, monthly_mean[pollutant], marker='o', color='b')
        plt.title(f'Mean of {pollutant.upper()}')
        plt.xlabel('Year')
        plt.ylabel(f'Monthly Mean {pollutant.upper()} Values')
        plt.grid(True)

        if i % 2 == 0 or i == len(pollutants):
            plt.tight_layout()  # Adjust subplot layout for better appearance
            plt.show()



def plot_yearly_mean_values(df):
    # Group data by year and calculate the mean for each pollutant
    yearly_mean = df.groupby(df.index.year).mean()

    # Plotting line graph for all pollutants (yearly mean)
    pollutants = ['pm2.5', 'pm10', 'o3', 'no2', 'so2', 'co']
    plt.figure(figsize=(12, 8))
    for pollutant in pollutants:
        plt.plot(yearly_mean.index, yearly_mean[pollutant], marker='o', label=pollutant.upper())

    plt.title('Mean Values of Pollutants Over the Years')
    plt.xlabel('Year')
    plt.ylabel('Mean Pollutant Values')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_mean_pie_chart(df):

    selected_year = 2022
    # Extracting year   
    df['Year'] = df.index.year
    yearly_data = df[df['Year'] == selected_year]    

    mean_percentages = yearly_data.drop('Year', axis=1).mean() / yearly_data.drop('Year', axis=1).mean().sum() * 100 

    # Exclude 'Year' from the data if present
    if 'Year' in mean_percentages.index:
        mean_percentages = mean_percentages.drop('Year')

    # Prepare data for plt.pie
    values = mean_percentages.values
    labels = mean_percentages.index

    # Plot a pie chart using plt.pie
    plt.pie(values, labels=None, autopct='%1.1f%%')
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
    plt.title(f'Mean Percentage of Pollutants in {selected_year}')

    # Add a legend
    plt.legend(labels, loc='best', bbox_to_anchor=(1, 0.5))
    plt.show()


def plot_correlation_matrix(df):
    pollutants = ['pm2.5', 'pm10', 'o3', 'no2', 'so2', 'co']
    # Correlation matrix
    correlation_matrix = df[pollutants].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix of Pollutants')
    plt.show()



def get_season(month):
    if 4 <= month <= 6:
        return 'Summer'
    elif 7 <= month <= 9:
        return 'Monsoon'
    elif 10 <= month <= 11:
        return 'Autumn'
    else:
        return 'Winter'



def plot_seasonal_changes(df):
    pollutants = ['pm2.5', 'pm10', 'o3', 'no2', 'so2', 'co']
    
    df['season'] = df.index.month.map(get_season) # Map months to respective seasons

    seasonal_data = df.groupby('season')[pollutants].mean()    # Group data by seasons

    # Plot changes in pollutants across seasons
    seasonal_data.T.plot(kind='bar', figsize=(10, 6))
    plt.title('Seasonal Changes in Pollutants')
    plt.xlabel('Pollutants')
    plt.ylabel('Mean Pollution Levels')
    plt.legend(title='Seasons')
    plt.show()


def plot_average_pollutant_levels_by_season(df):
    pollutants = ['pm2.5', 'pm10', 'o3', 'no2', 'so2', 'co']
    # Map months to respective seasons
    df['season'] = df.index.month.map(get_season)

    # Group data by seasons
    seasonal_data = df.groupby('season')[pollutants].mean()

    # Get the number of seasons
    num_seasons = len(seasonal_data)

    # Set up subplots
    fig, axes = plt.subplots(1, num_seasons, figsize=(6 * num_seasons, 6))

    # Plot pie charts for each season
    for i, season in enumerate(seasonal_data.index):
        season_data = seasonal_data.loc[season]

        # Plot in the i-th subplot
        axes[i].pie(season_data, labels=season_data.index, autopct='%1.1f%%')
        axes[i].set_title(f'Average Pollutant Levels in {season}')

    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.show()



def compare_pollutants_between_years(df):
    pollutants = ['pm2.5', 'pm10', 'o3', 'no2', 'so2', 'co']
    years_to_compare = [2015, 2020, 2021, 2022]
    # Extract data for the years to compare
    data_years = []

    for year in years_to_compare:
        data_years.append(df[df.index.year == year][pollutants].mean())

    # Plot a side-by-side bar chart for comparison
    plt.figure(figsize=(10, 6))
    bar_width = 0.20
    index = np.arange(len(pollutants))

    for i, data_year in enumerate(data_years):
        plt.bar(index + i * bar_width, data_year, bar_width, label=str(years_to_compare[i]))

    plt.xlabel('Pollutants')
    plt.ylabel('Mean Pollution Levels')
    plt.title('Comparison of Pollutants Between Years')
    plt.xticks(index + bar_width * (len(data_years) - 1) / 2, pollutants)
    plt.legend()
    plt.show()


# Define the sub-index calculation functions

def get_PM2_5_subindex(x):
## PM2.5 Sub-Index calculation
    if x <= 30:
        return x * 50 / 30
    elif x <= 60:
        return 50 + (x - 30) * 50 / 30
    elif x <= 90:
        return 100 + (x - 60) * 100 / 30
    elif x <= 120:
        return 200 + (x - 90) * 100 / 30
    elif x <= 250:
        return 300 + (x - 120) * 100 / 130
    elif x > 250:
        return 400 + (x - 250) * 100 / 130
    else:
        return 0


def get_PM10_subindex(x):
    if x <= 50:
        return x
    elif 50 < x <= 100:
        return x
    elif 100 < x <= 250:
        return 100 + (x - 100) * 100 / 150
    elif 250 < x <= 350:
        return 200 + (x - 250)
    elif 350 < x <= 430:
        return 300 + (x - 350) * 100 / 80
    elif x > 430:
        return 400 + (x - 430) * 100 / 80
    else:
        return 0

## O3 Sub-Index calculation
def get_O3_subindex(x):
    if x <= 50:
        return x * 50 / 50
    elif x <= 100:
        return 50 + (x - 50) * 50 / 50
    elif x <= 168:
        return 100 + (x - 100) * 100 / 68
    elif x <= 208:
        return 200 + (x - 168) * 100 / 40
    elif x <= 748:
        return 300 + (x - 208) * 100 / 539
    elif x > 748:
        return 400 + (x - 400) * 100 / 539
    else:
        return 0

#NO2 Subindex Calculation:
def get_NO2_subindex(x):
    if x <= 40:
        return x * 50 / 40
    elif x <= 80:
        return 50 + (x - 40) * 50 / 40
    elif x <= 180:
        return 100 + (x - 80) * 100 / 100
    elif x <= 280:
        return 200 + (x - 180) * 100 / 100
    elif x <= 400:
        return 300 + (x - 280) * 100 / 120
    elif x > 400:
        return 400 + (x - 400) * 100 / 120
    else:
        return 0

## SO2 Sub-Index calculation
def get_SO2_subindex(x):
    if x <= 40:
        return x * 50 / 40
    elif x <= 80:
        return 50 + (x - 40) * 50 / 40
    elif x <= 380:
        return 100 + (x - 80) * 100 / 300
    elif x <= 800:
        return 200 + (x - 380) * 100 / 420
    elif x <= 1600:
        return 300 + (x - 800) * 100 / 800
    elif x > 1600:
        return 400 + (x - 1600) * 100 / 800
    else:
        return 0

## CO Sub-Index calculation
def get_CO_subindex(x):
    if x <= 1:
        return x * 50 / 1
    elif x <= 2:
        return 50 + (x - 1) * 50 / 1
    elif x <= 10:
        return 100 + (x - 2) * 100 / 8
    elif x <= 17:
        return 200 + (x - 10) * 100 / 7
    elif x <= 34:
        return 300 + (x - 17) * 100 / 17
    elif x > 34:
        return 400 + (x - 34) * 100 / 17
    else:
        return 0

# Define the function to calculate AQI for each row
def calculate_aqi(row):
    pm25_subindex = get_PM2_5_subindex(row['pm2.5'])
    pm10_subindex = get_PM10_subindex(row['pm10'])
    o3_subindex = get_O3_subindex(row['o3'])
    no2_subindex = get_NO2_subindex(row['no2'])
    so2_subindex = get_SO2_subindex(row['so2'])
    co_subindex = get_CO_subindex(row['co'])

    # Calculate AQI by finding the maximum of the sub-indices
    aqi = max(pm25_subindex, pm10_subindex, o3_subindex, no2_subindex, so2_subindex, co_subindex)
    return aqi


def get_AQI_Category(x):
    if x <= 50:
        return "Good"
    elif 50 < x <= 100:
        return "Satisfactory"
    elif 100 < x <= 200:
        return "Moderate"
    elif 200< x <= 300:
        return "Poor"
    elif 300 < x <= 400:
        return "Very Poor"
    elif x > 400:
        return "Severe"
    else:
        return "Uncategorized"
    


def plot_aqi_categories_over_years(df):
    # Apply the calculate_aqi function
    df['AQI'] = df.apply(calculate_aqi, axis=1)

    # Apply the get_AQI_Category function
    df['AQI_Category'] = df['AQI'].apply(get_AQI_Category)
    yearly_aqi_category_counts = df.groupby(['Year', 'AQI_Category']).size().unstack(fill_value=0)

    # Plotting the line graph for AQI categories over years
    categories = yearly_aqi_category_counts.columns
    yearly_aqi_category_counts[categories].plot(kind='line', figsize=(10, 6))
    plt.title('AQI Category Counts Over Years')
    plt.xlabel('Year')
    plt.ylabel('Number of Days')
    plt.legend(title='AQI Category')
    plt.show()


def plot_aqi_category_pie_by_season(df):
    # Map months to respective seasons
    df['season'] = df.index.month.map(get_season)

    # Create subplots for each season
    seasons = df['season'].unique()
    num_seasons = len(seasons)
    fig,axes = plt.subplots(1, num_seasons, figsize=(6 * num_seasons, 6))

    for i, season in enumerate(seasons):
        # Filter data for the specific season
        season_data = df[df['season'] == season]
        category_counts = season_data['AQI_Category'].value_counts()

        # Plot a pie chart for each season
        axes[i].pie(category_counts, labels=category_counts.index, autopct='%1.1f%%', startangle=140)
        axes[i].set_title(f'AQI Category Distribution in {season}')

    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.show()


def main():
    """
    Main function to execute the entire analysis process.
    """
    file_path = 'c:/Users/Pushpender Kumar/Desktop/Python/Delhiaqi.csv'  # Replace with your file path
    df = load_data(file_path)

    print_clean_data_info(df) # print some information about the data and clean the data

    df = set_date_as_index(df) # make date as index to easy the process of extracting date and month

    plot_monthly_mean_trend(df) #plots monthly trend of all pollutants of all the years

    plot_yearly_mean_values(df) #  make line plot and all pollutant in a single graph taking the mean yearly

    plot_mean_pie_chart(df) #  make a pie chart of a particular year

    plot_correlation_matrix(df) # plot a correlation between different pollutants

    plot_seasonal_changes(df) # plot bar graph how level of pollutants changes according to season

    plot_average_pollutant_levels_by_season(df) # plot the pie chart about the percentage of pollutant levels in different seasons
  
    compare_pollutants_between_years(df) # plot a bar graph by taking different years data
    
    plot_aqi_categories_over_years(df) # Plot AQI categories over years

    plot_aqi_category_pie_by_season(df) # Plot AQI categories over the seasons 

if __name__ == "__main__":
    main()


