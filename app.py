import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
import seaborn as sns
from iso3166 import countries

st.set_page_config(layout="wide")
st.title("SPACE RACE: Global Space Missions from 1957")

st.markdown("###Loading and Cleaning Data")
missions_df = pd.read_csv('mission_launches.csv')

missions_df['Price'] = pd.to_numeric(missions_df.Price, errors="coerce").fillna(0)
missions_df['Price'] = missions_df.Price.replace(",", "", regex=True)
missions_df['Price'] = missions_df.Price.astype(float)
missions_df.drop(['Unnamed: 0.1', 'Unnamed: 0'], axis=1, inplace=True)

missions_df['Date'] = pd.to_datetime(missions_df.Date, utc=True, errors="coerce")
missions_df['Year'] = missions_df.Date.dt.year
missions_df['Year'] = missions_df.Year.fillna(0).astype(int)
missions_df = missions_df[missions_df.Year != 0]

missions_df['Country'] = missions_df.Location.str.split(",").str[-1].str.strip()
missions_df['Country'] = missions_df.Country.replace({
    'Shahrud Missile Test Site': 'Iran, Islamic Republic of',
    'New Mexico': 'United States of America',
    'Yellow Sea': 'China',
    'Pacific Missile Range Facility': 'United States of America',
    'Pacific Ocean': 'United States of America',
    'Barents Sea': 'Russian Federation',
    'Gran Canaria': 'Spain',
    'North Korea': "Korea, Democratic People's Republic of",
    'South Korea': 'Korea, Republic of',
    'Russia': 'Russian Federation',
    'Iran': 'Iran, Islamic Republic of',
    'USA': 'United States of America'
})
missions_df['Country_Code'] = missions_df.Country.apply(lambda x: countries.get(x).alpha3)

def agency_status(org):
    private_orgs = {
        'lockheed', 'land launch', 'northrop', 'arianespace', 'i-space',
        'boeing', 'martin marietta', 'expace', 'exos', 'virgin orbit',
        'spacex', 'general dynamics','douglas', 'rocket lab', 'ils',
        'blue origin', 'sea launch', 'onespace', 'landspace', 'eurockot'
    }
    government_orgs = {
        'casic', 'asi', 'src', 'casc', 'yuzhmash', 'cecles', 'kosmotras',
        'mhi', 'us navy', 'kari', 'okb-586', 'us air force', 'khrunichev',
        'ut', 'jaxa', 'isa', 'irgc', 'roscosmos', 'esa', 'rvsn ussr', 'vks rf',
        'nasa', 'eer', 'rae', 'cnes', 'ula', 'sandia', 'iai', 'starsem',
        'mitt', 'aeb', 'isro', 'amba', "arm??e de l'air", 'isas', 'kcst'
    }
    if org.lower() in private_orgs:
        return 'Private'
    elif org.lower() in government_orgs:
        return 'Government'
    else:
        return 'Unknown'

missions_df['Agency_Type'] = missions_df.Organisation.apply(agency_status)
missions_df['Success_Binary'] = missions_df.Mission_Status.apply(lambda x: 1 if x == 'Success' else 0)

missions_df['Price_Filled'] = missions_df.Price
missions_df.loc[missions_df['Price_Filled'] == 0, 'Price_Filled'] = np.nan
missions_df['Price_Filled'] = missions_df.groupby('Agency_Type').Price_Filled.transform(
    lambda x: x.fillna(x.median())
)

st.markdown("### Annual Mission Count Over Time")
missions_per_year = missions_df.groupby('Year').size().reset_index(name='Mission_Count')
fig1 = px.line(missions_per_year, x='Year', y='Mission_Count',
               title='Number of Space Missions Over Time')
fig1.update_layout(xaxis=dict(rangeslider=dict(visible=True)))
st.plotly_chart(fig1, use_container_width=True)

st.markdown("### Annual Success Rates of Space Missions")
success_rate = missions_df.groupby('Year').Success_Binary.mean().reset_index()
success_rate['Success_Percent'] = success_rate.Success_Binary * 100
fig2, ax = plt.subplots(figsize=(14, 6))
sns.lineplot(data=success_rate, x='Year', y='Success_Percent', ax=ax)
ax.set_title('Annual Mission Success Rate (%) Over Time')
ax.set_ylabel('Success Rate (%)')
ax.set_xlabel('Year')
ax.set_ylim(0, 100)
st.pyplot(fig2)

st.markdown("### Mission Status Trends by Year")
status_counts = missions_df.groupby(['Year', 'Mission_Status']).size().unstack(fill_value=0)
fig3, ax = plt.subplots(figsize=(10, 6))
for status in status_counts.columns:
    ax.plot(status_counts.index, status_counts[status], label=status)
ax.legend()
st.pyplot(fig3)

st.markdown("### Launch Outcome Breakdown by Country")
success_by_country = pd.crosstab(missions_df.Country_Code, missions_df.Mission_Status)
success_by_country['Total_Missions'] = success_by_country.sum(axis=1)
success_by_country.sort_values('Total_Missions', ascending=False, inplace=True)
success_by_country.reset_index(inplace=True)

x = np.arange(len(success_by_country))
fig4, ax = plt.subplots(figsize=(12, 6))
ax.bar(x, success_by_country['Failure'], label='Failure', color='tomato')
ax.bar(x, success_by_country['Success'], bottom=success_by_country['Failure'], label='Success', color='seagreen')
ax.bar(x, success_by_country['Partial Failure'], bottom=success_by_country['Failure'] + success_by_country['Success'], label='Partial Failure', color='gold')
ax.bar(x, success_by_country['Prelaunch Failure'], bottom=success_by_country['Failure'] + success_by_country['Success'] + success_by_country['Partial Failure'], label='Prelaunch Failure', color='gray')
ax.set_xticks(x)
ax.set_xticklabels(success_by_country['Country_Code'], rotation=45)
ax.set_title('Rocket Launch Outcomes by Country')
ax.set_xlabel('Country')
ax.set_ylabel('Number of Launches')
ax.legend()
st.pyplot(fig4)

st.markdown("### Animated View of the Space Race Over Time")
annual_country = missions_df.groupby(['Year', 'Country']).size().reset_index(name='Missions')
all_years = range(annual_country.Year.min(), annual_country.Year.max() + 1)
all_countries = annual_country.Country.unique()
complete_index = pd.MultiIndex.from_product([all_years, all_countries], names=['Year', 'Country'])
annual_country = annual_country.set_index(['Year', 'Country']).reindex(complete_index, fill_value=0).reset_index()
annual_country['Cumulative_Missions'] = annual_country.groupby('Country').Missions.cumsum()

fig5 = px.bar(
    annual_country,
    x="Cumulative_Missions",
    y="Country",
    color="Country",
    animation_frame="Year",
    title="Space Race Over Time (Animated)",
    range_x=[0, annual_country.Cumulative_Missions.max() + 100],
    height=700
)
fig5.update_layout(yaxis={'categoryorder': 'total ascending'})
st.plotly_chart(fig5)

st.markdown("### Global Distribution of Missions by Country")
country_counts = missions_df.Country.value_counts().reset_index()
country_counts.columns = ['Country', 'Mission_Count']
fig6 = px.choropleth(
    country_counts,
    locations='Country',
    locationmode='country names',
    color='Mission_Count',
    color_continuous_scale='Viridis',
    title='Global Distribution of Space Missions'
)
fig6.update_layout(geo=dict(showframe=False, projection_type='natural earth'))
st.plotly_chart(fig6)

st.markdown("### Top Launch Sites Worldwide")
missions_df['Location'] = missions_df.Location.str.strip()
top_sites = missions_df.Location.value_counts().head(20).reset_index()
top_sites.columns = ['Launch_Site', 'Launch_Count']
fig7, ax = plt.subplots(figsize=(12, 6))
sns.barplot(data=top_sites, x='Launch_Count', y='Launch_Site', ax=ax, palette='viridis')
ax.set_title('Top 20 Launch Sites by Number of Missions')
st.pyplot(fig7)

st.markdown("### Launch Activity Heatmap by Top Launch Sites")
site_year = missions_df.groupby(['Location', 'Year']).size().unstack(fill_value=0)
top_launch_sites = missions_df['Location'].value_counts().head(20).index
site_year_top = site_year.loc[top_launch_sites]
fig8, ax = plt.subplots(figsize=(14, 6))
sns.heatmap(site_year_top, cmap='rocket_r', linewidths=0.5, linecolor='gray', ax=ax)
ax.set_title('Heatmap of Launch Activity by Top Launch Sites')
st.pyplot(fig8)

st.markdown("### 🛰️ Launch Volume by Top Agencies")
agency_yearly = missions_df.groupby(['Year', 'Organisation']).size().unstack(fill_value=0)
top_agencies = missions_df.Organisation.value_counts().head(10).index
agency_yearly_top = agency_yearly[top_agencies]
fig9, ax = plt.subplots(figsize=(14, 6))
agency_yearly_top.plot.area(alpha=0.4, ax=ax)
ax.set_title('Launch Volume by Top Agencies Over Time')
st.pyplot(fig9)

st.markdown("### Expenditure by Agency and Country")
agg1 = missions_df.groupby(['Agency_Type', 'Country_Code', 'Organisation']).Price.sum().reset_index()
fig10 = px.sunburst(
    agg1,
    path=['Agency_Type', 'Country_Code', 'Organisation'],
    values='Price',
    color='Agency_Type',
    color_discrete_map={'Private': '#0046FF', 'Government': '#EF553B'},
    title='Expenditure by Agency Type and Organisation'
)
fig10.update_layout(margin=dict(t=50, l=0, r=0, b=0))
st.plotly_chart(fig10)

st.markdown("### Expenditure Estimate (With Median-Filled Prices)")
agg2 = missions_df.groupby(['Agency_Type', 'Country_Code', 'Organisation']).Price_Filled.sum().reset_index()
fig11 = px.sunburst(
    agg2,
    path=['Agency_Type', 'Country_Code', 'Organisation'],
    values='Price_Filled',
    color='Agency_Type',
    color_discrete_map={'Private': '#636EFA', 'Government': '#FFA673'},
    title='Expenditure by Agency Type (Median Estimate)'
)
fig11.update_layout(margin=dict(t=50, l=0, r=0, b=0))
st.plotly_chart(fig11)
