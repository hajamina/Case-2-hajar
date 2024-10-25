#!/usr/bin/env python
# coding: utf-8

# In[677]:


#!/usr/bin/env python
# coding: utf-8

# In[5]:
#!pip install seaborn

import os
import numpy as np
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px

#%%

#%%

employee = pd.read_csv('Employee.csv', delimiter = ',')
performancerate = pd.read_csv('PerformanceRating.csv', delimiter = ',')

#combineren van de datasets 
performancerate['ReviewDate'] = pd.to_datetime(performancerate['ReviewDate'])  # Zorg dat de aanstellingsdatum in datetime-formaat is
recent_performance = performancerate.loc[performancerate.groupby('EmployeeID')['ReviewDate'].idxmax()]
combined_dataset =  pd.merge(employee, recent_performance, on='EmployeeID', how='left')

#%%


#%%

st.markdown("""
    <h1 style='text-align: center; font-size: 40px; color: #0A1172; font-family: Arial, sans-serif;'>
        HR Dashboard 📊🚀
    </h1>
    <h3 style='text-align: center; font-size: 18px; color: #0A1172; font-family: Arial, sans-serif;'>
        Verbeterde versie
    </h3>
""", unsafe_allow_html=True)

# Ondertitel met CSS voor centreren
st.markdown("""
    <h2 style='text-align: center; font-size: 30px; color: #0A1172; font-family: Arial, sans-serif;'>
        Medewerkerstevredenheid en attritie
    </h2>
""", unsafe_allow_html=True)



combined_dataset['ReviewDate'] = pd.to_datetime(combined_dataset['ReviewDate'], errors='coerce')

# 1. Basisinformatie van de dataset
st.write("""
    Dit dashboard omvangt informatie over de werknemerstevredenheid en attritie. We beginnen eerst met inzicht geven over de werktevredenheidsscore
    en leggen vervolgens een verband met de attritie van de werknemers. We analyseren een uitgebreide dataset en gebruiken geavanceerde machine learning-modellen 
    om te voorspellen welke werknemers het grootste risico lopen te vertrekken. Je kunt patronen en verbanden ontdekken tussen variabelen zoals werkervaring, salaris, 
    werktevredenheid en promotiefrequentie, en zien hoe deze bijdragen aan de beslissing om te blijven of te vertrekken.
    
    Voor de HR-afdeling is dit dashboard een krachtig hulpmiddel om potentiële risico’s vroegtijdig te signaleren en gericht in te grijpen. 
    Door inzicht te krijgen in de belangrijkste drijfveren achter werknemerstevredenheid en vertrekrisico's, kan HR strategieën ontwikkelen om talent te behouden, 
    de werktevredenheid te verhogen en de algehele bedrijfsperformance te verbeteren
    """)

#%%
# Tweede lijn toevoegen
st.markdown("<hr style='border: 2px solid #0A1172; margin: 20px 0;'>", unsafe_allow_html=True)

# Tevredenheidsmetrics
st.subheader('Algemene Tevredenheid van Medewerkers')

# Gemiddelde tevredenheidscijfers
col1, col2, col3, col4, col5 = st.columns(5)

avg_job_satisfaction = combined_dataset['JobSatisfaction'].mean()
avg_env_satisfaction = combined_dataset['EnvironmentSatisfaction'].mean()
avg_relation_satisfaction = combined_dataset['RelationshipSatisfaction'].mean()
avg_manager_rating = combined_dataset['ManagerRating'].mean()
avg_self_rating = combined_dataset['SelfRating'].mean()

# Metrics visualiseren
col1.metric("Job Tevredenheid", f"{avg_job_satisfaction:.2f}/5")
col2.metric("Omgeving Tevredenheid", f"{avg_env_satisfaction:.2f}/5")
col3.metric("Relatie Tevredenheid", f"{avg_relation_satisfaction:.2f}/5")
col4.metric("Leidinggevende Beoordeling", f"{avg_manager_rating:.2f}/5")
col5.metric("Zelf Beoordeling", f"{avg_self_rating:.2f}/5")

#%%

# Box plot voor job tevredenheid per functie
st.subheader("Job Tevredenheid per Functie")

# Bereken het gemiddelde van de job tevredenheid
mean_satisfaction = combined_dataset['JobSatisfaction'].mean()

# Boxplot voor job tevredenheid per functie
fig_box = px.box(combined_dataset, x='JobRole', y='JobSatisfaction', 
                 title="Job Tevredenheid Verdeling per Functie", 
                 labels={'JobRole': 'Functie', 'JobSatisfaction': 'Job Tevredenheid'})

# Voeg een gemiddelde lijn toe
fig_box.add_hline(y=mean_satisfaction, line_dash="dash", 
                  annotation_text="Gemiddelde Tevredenheid", 
                  annotation_position="top left")

# Update de layout van de figuur
fig_box.update_layout(xaxis_title='Functie', yaxis_title='Job Tevredenheid')

# Toon de plot in Streamlit
st.plotly_chart(fig_box)

#%%
# Tweede lijn toevoegen
st.markdown("<hr style='border: 2px solid #0A1172; margin: 20px 0;'>", unsafe_allow_html=True)

# 6. Extra: Distributie van numerieke kolommen
import seaborn as sns
import matplotlib.pyplot as plt

# Distributie van numerieke kolommen
st.subheader("Distributie van numerieke kolommen")
numeric_columns = combined_dataset.select_dtypes(include=['float64', 'int64']).columns
selected_column = st.selectbox("Kies een numerieke kolom om te visualiseren:", numeric_columns)

# Voeg een slider toe waarmee de gebruiker het aantal bins kan instellen
bins = st.slider("Selecteer het aantal bins voor het histogram", min_value=5, max_value=50, value=20)

# Plot het histogram
fig, ax = plt.subplots()
sns.histplot(combined_dataset[selected_column], bins=bins, kde=True, ax=ax)
ax.set_title(f"Histogram van {selected_column} (met {bins} bins)")
ax.set_xlabel(selected_column)
ax.set_ylabel("Aantal")

# Toon de plot in Streamlit
st.pyplot(fig)

# Voeg statistieken toe over de gekozen kolom
mean_value = combined_dataset[selected_column].mean()
min_value = combined_dataset[selected_column].min()
max_value = combined_dataset[selected_column].max()

# Toon de statistieken met vetgedrukte getallen en de naam van de geselecteerde kolom
st.write(f"**Gemiddelde van {selected_column} is:** **{mean_value:.2f}**")
st.write(f"**Minimum van {selected_column} is:** **{min_value:.2f}**")
st.write(f"**Maximum van {selected_column} is:** **{max_value:.2f}**")

combined_dataset['HireDate']=combined_dataset['HireDate'].astype('datetime64[ns]')

#%%

combined_dataset.insert(0, 'FullName', combined_dataset['FirstName'] + ' ' + combined_dataset['LastName'])  # Plaats op positie 0 (eerste kolom)

# Verwijder de originele kolommen 'FirstName', 'LastName' en andere onodige kolomen 
combined_dataset.drop(columns=['FirstName', 'LastName','SelfRating','ManagerRating'], inplace=True)

#%%

# Meerdere kolommen verwijderen
combined_dataset = combined_dataset.drop(columns=['EmployeeID', 'PerformanceID'])

#%%
# Tweede lijn toevoegen
st.markdown("<hr style='border: 2px solid #0A1172; margin: 20px 0;'>", unsafe_allow_html=True)

# Bekijk de unieke waarden in de 'Gender' kolom
print(combined_dataset['Gender'].unique())
print(combined_dataset['BusinessTravel'].unique())
print(combined_dataset['Attrition'].unique())
print(combined_dataset['Department'].unique())

# Verwijder leidende en volgende spaties in 'BusinessTravel' en 'Gender'
combined_dataset['BusinessTravel'] = combined_dataset['BusinessTravel'].str.strip()
combined_dataset['Gender'] = combined_dataset['Gender'].str.strip()

# Omzetten van de volgende kolomen om in numerieke warden 
combined_dataset['Gender'] = combined_dataset['Gender'].map({"Prefer Not To Say" :0, "Male": 1, "Female": 2, "Non-Binary":3})
combined_dataset['BusinessTravel'] = combined_dataset['BusinessTravel'].map({"No Travel": 0, "Some Travel": 1, "Frequent Traveller": 2})
combined_dataset['Attrition'] = combined_dataset['Attrition'].map({'Yes': 1, 'No': 0})

#Een nieuwe variabele toevoegen aan dataset
combined_dataset['PromotionFrequency'] = (combined_dataset['YearsAtCompany'] / (combined_dataset['YearsSinceLastPromotion'] + 1)).round().astype(int)

gps_coordinates = {
    'IL': (40.6331, -89.3985),   # Coördinaten van Springfield, IL
    'CA': (36.7783, -119.4179), # Coördinaten van Sacramento, CA
    'NY': (40.7128, -74.0060)     # Coördinaten van Albany, NY
}

# Woordenboeken voor latitude (breedtegraad) en longitude (lengtegraad)
latitude_dict = {State: gps_coordinates[State][0] for State in gps_coordinates}
longitude_dict = {State: gps_coordinates[State][1] for State in gps_coordinates}

# Latitude en longitude kolommen toevoegen aan de DataFrame
combined_dataset['latitude'] = combined_dataset['State'].map(latitude_dict)
combined_dataset['longitude'] = combined_dataset['State'].map(longitude_dict)



# In[678]:


#combined_dataset.head()


# In[681]:


#pip install folium


# In[683]:


import streamlit as st
import pandas as pd
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
from collections import Counter  # Voeg deze regel toe

# Voorbeeld dataset laden (vervang dit met je eigen dataset)
# combined_dataset = pd.read_csv('your_dataset.csv')

# Interactieve widgets voor filtering
selected_states = st.multiselect(
    "Select States:",
    options=combined_dataset['State'].unique(),
    default=combined_dataset['State'].unique()
)

attrition_filter = st.radio(
    "Show Employees Who:",
    ('Stay (Attrition = 0)', 'Leave (Attrition = 1)', 'All')
)

# Dataset filteren op basis van gebruikersselectie
filtered_data = combined_dataset[combined_dataset['State'].isin(selected_states)]

if attrition_filter == 'Stay (Attrition = 0)':
    filtered_data = filtered_data[filtered_data['Attrition'] == 0]
elif attrition_filter == 'Leave (Attrition = 1)':
    filtered_data = filtered_data[filtered_data['Attrition'] == 1]

# Folium kaart maken
m = folium.Map(location=[39.8283, -98.5795], zoom_start=4)  # USA centrale coördinaten

# Marker Cluster toevoegen
marker_cluster = MarkerCluster().add_to(m)

# Markers toevoegen op basis van gefilterde data
for idx, row in filtered_data.iterrows():
    folium.CircleMarker(
        location=[row['latitude'], row['longitude']],
        radius=7,
        color='red' if row['Attrition'] == 1 else 'green',
        fill=True,
        fill_color='red' if row['Attrition'] == 1 else 'green',
        fill_opacity=0.7,
        popup=f"Department: {row['Department']}, Years at Company: {row['YearsAtCompany']}"
    ).add_to(marker_cluster)


# Kaart weergeven in Streamlit
st_folium(m, width=725)

total_employees_in_selected_states = len(filtered_data)
leaving_employees = filtered_data[filtered_data['Attrition'] == 1]
num_leaving_employees = len(leaving_employees)

if num_leaving_employees > 0:
    avg_years_at_company_leaving = leaving_employees['YearsAtCompany'].mean()
    most_common_department = Counter(leaving_employees['Department']).most_common(1)[0][0]
else:
    avg_years_at_company_leaving = 0
    most_common_department = "N/A"

st.markdown(
    """
    <div style='border: 2px solid grey; padding: 10px;'>
        <h4>Legenda</h4>
        <span style="color: green; font-weight: bold;">&#9679;</span> Gebleven<br>
        <span style="color: red; font-weight: bold;">&#9679;</span> Verlaten<br>
    </div>
    """, unsafe_allow_html=True
)
# Samenvatting in het Nederlands onder de kaart weergeven
st.write(f"**Totaal aantal werknemers in geselecteerde staten:** {total_employees_in_selected_states}")
st.write(f"**Aantal vertrekkende werknemers:** {num_leaving_employees}")
st.write(f"**Gemiddeld aantal jaren bij het bedrijf (vertrekkende werknemers):** {avg_years_at_company_leaving:.2f}")
st.write(f"**Meest voorkomende afdeling (vertrekkende werknemers):** {most_common_department}")



# In[684]:


# Tweede lijn toevoegen
st.markdown("<hr style='border: 2px solid #0A1172; margin: 20px 0;'>", unsafe_allow_html=True)
# **Heatmap van correlaties**
st.subheader('Correlatiematrix van Factoren die Werknemersattritie Beïnvloeden')

# Correlatiematrix berekenen
corr = combined_dataset[['Attrition', 'JobSatisfaction', 'WorkLifeBalance', 
                           'Age', 'YearsSinceLastPromotion', 'DistanceFromHome (KM)', 'Salary','PromotionFrequency']].corr()

# Masker maken voor de bovenste helft
mask = np.triu(np.ones_like(corr, dtype=bool))

# Plotfiguur maken
fig, ax = plt.subplots(figsize=(10, 6))

# Heatmap tekenen met masker voor de bovenste helft
sns.heatmap(corr, mask=mask, annot=True, cmap='Blues', ax=ax)

# Heatmap in Streamlit tonen
st.pyplot(fig)


# Alleen correlaties met Attrition extraheren
# Eerst ervoor zorgen dat de 'Attrition' kolom goed is omgezet
if combined_dataset['Attrition'].dtype == 'object':
    combined_dataset['Attrition'] = combined_dataset['Attrition'].map({'No': 0, 'Yes': 1})

# Controleer ook of de andere kolommen die je wilt analyseren de juiste datatypes hebben
# Converteer waar nodig
numeric_columns = ['JobSatisfaction', 'WorkLifeBalance', 'Age', 
                   'YearsSinceLastPromotion', 'DistanceFromHome (KM)', 
                   'Salary', 'PromotionFrequency']

for col in numeric_columns:
    if combined_dataset[col].dtype == 'object':
        combined_dataset[col] = pd.to_numeric(combined_dataset[col], errors='coerce')

# Alleen numerieke waarden behouden en na de conversie de correlatie berekenen
corr_matrix = combined_dataset[['Attrition'] + numeric_columns].corr()

# Correlaties met Attrition extraheren
attrition_corr = corr_matrix['Attrition'].drop('Attrition')

# Correlatiedrempel instellen
threshold = 0.1
strong_corr = attrition_corr[(attrition_corr >= threshold) | (attrition_corr <= -threshold)]

# Tabel weergeven met sterke correlaties
st.subheader("Sterke Verbanden met Attrition")
st.write("""
Uit de onderstaande correlatiematrix kunnen we een aantal belangrijke verbanden met betrekking tot werknemersattritie afleiden. Hieronder een overzicht van de sterkste correlaties:
""")

# Lijst van sterke correlaties in tabelvorm weergeven
st.table(strong_corr)

# Kort overzicht van de verbanden als tekst
if not strong_corr.empty:
    st.write("We zien dat de volgende factoren een sterke correlatie hebben met werknemersattritie:")
    for factor, corr_value in strong_corr.items():
        st.write(f"- **{factor}** heeft een correlatie van **{corr_value:.2f}** met werknemersattritie.")
else:
    st.write("Er zijn geen sterke correlaties (> |0.1|) gevonden tussen de factoren en werknemersattritie.")
st.markdown("<hr style='border: 2px solid #0A1172; margin: 20px 0;'>", unsafe_allow_html=True)

 


# In[685]:


st.subheader('Werknemersatrittie t.o.v. Overuren')

# Visualisatie: gestapeld staafdiagram van OverTime en Attrition
fig, ax = plt.subplots()

# Groepeer de gegevens per OverTime en Attrition
grouped_data = combined_dataset.groupby(['OverTime', 'Attrition']).size().unstack(fill_value=0)

# Maak het gestapeld staafdiagram
grouped_data.plot(kind='bar', stacked=True, color=['green', 'red'], ax=ax)

# Pas de labels op de x-as aan voor de OverTime-categorieën
ax.set_xticklabels(['Geen Overtime', 'Wel Overtime'], rotation=0)

# Voeg titel en labels toe
ax.set_title('Verdeling van Attrition per Overtime')
ax.set_ylabel('Aantal Werknemers')
ax.set_xlabel('OverTime')

custom_legend = [plt.Line2D([0], [0], color='green', lw=4, label='Gebleven'),
                 plt.Line2D([0], [0], color='red', lw=4, label='Verlaten')]
ax.legend(handles=custom_legend, title='Attrition Status')

# Voeg aantallen toe aan de staven
for p in ax.patches:
    width = p.get_width()
    height = p.get_height()
    x, y = p.get_xy() 
    ax.text(x + width / 2, y + height / 2, int(height), ha='center', va='center', color='white')

# Toon de plot
st.pyplot(fig)
st.markdown("<hr style='border: 2px solid #0A1172; margin: 20px 0;'>", unsafe_allow_html=True)


# In[686]:


import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Verwijder de selectiebox en laat alle leeftijdsgroepen zien
st.subheader('Boxplots: Effect van Leeftijd en Werkpromotie op Werknemersattritie')

# Maak een nieuwe kolom voor leeftijdsgroepen
combined_dataset['AgeGroup'] = pd.cut(combined_dataset['Age'], 
                                      bins=[18, 30, 40, 50], 
                                      labels=['18-29', '30-39', '40-49'])

# Maak een FacetGrid voor alle leeftijdsgroepen
g = sns.FacetGrid(combined_dataset, col='AgeGroup', height=4, aspect=1)

# Voeg de boxplot toe zonder 'hue', gebruik 'Attrition' als x-variabele en pas het palet toe
palette = {'0': 'green', '1': 'red'}
g.map(sns.boxplot, 'Attrition', 'YearsSinceLastPromotion', order=['0', '1'], palette=palette, dodge=False)

# Bereken de medianen per leeftijdsgroep en attrition, en voeg observed=False toe
median_values = combined_dataset.groupby(['AgeGroup', 'Attrition'], observed=False)['YearsSinceLastPromotion'].median().unstack()

# Voeg titels en labels toe
g.set_axis_labels('Attrition', 'Years Since Last Promotion')
g.set_titles('Leeftijdsgroep: {col_name}')

# Annotatie van medianen toevoegen aan de boxplots
for ax in g.axes.flatten():
    age_group = ax.get_title().split(': ')[-1]  # Haal de leeftijdsgroep uit de titel
    for i, attrition in enumerate(median_values.columns):
        median = median_values.loc[age_group, attrition]
        ax.annotate(f'Mediaan: {median}', 
                    xy=(i, median),  # xy is de positie op de x- en y-as
                    xytext=(i, median + 1),  # xytext is de positie van de tekst (aangepast voor duidelijkheid)
                    ha='center', va='bottom',  # ha is horizontal alignment, va is vertical alignment
                    color='black', fontsize=10, weight='bold',
                    arrowprops=dict(facecolor='black', shrink=0.05))  # Optionele pijl toevoegen
    
    # Pas de xtick-labels aan naar "Gebleven" en "Verlaten"
    ax.set_xticks([0, 1])  # Stel de locatie van de ticks in
    ax.set_xticklabels(['Gebleven', 'Verlaten'])  # Pas de labels van de ticks aan

# Toon de figuur in Streamlit
st.pyplot(g.fig)
st.write("Over het algemeen hebben werknemers die zijn gebleven een hogere mediaan van (years since last promotion) vergeleken met degenen die zijn vertrokken. Dit suggereert dat werknemers die recentere promoties hebben gehad, een lagere kans op vertrek hebben, vooral zichtbaar in de leeftijdsgroepen 18-29 en 30-39.")

st.markdown("<hr style='border: 2px solid #0A1172; margin: 20px 0;'>", unsafe_allow_html=True)


# In[687]:


# Leeftijd categoriseren
bins = [18, 30, 40, 50]  # Leeftijdsgrenzen
labels = ['18-29', '30-39', '40-49']  # Labels voor de groepen
combined_dataset['Leeftijdsgroep'] = pd.cut(combined_dataset['Age'], bins=bins, labels=labels, right=False)

# Aantal werknemers per leeftijdsgroep en attritiestatus
age_attrition = combined_dataset.groupby(['Leeftijdsgroep', 'Attrition'], observed=False).size().unstack(fill_value=0)

# Bereken het percentage vertrokken werknemers per leeftijdsgroep
age_attrition['Percentage_Verlaten'] = age_attrition[1] / (age_attrition[1] + age_attrition[0]) * 100

st.subheader('Effect van Leeftijd op werknemersattritie')

fig, ax1 = plt.subplots(figsize=(10, 6))

# Staafgrafiek: aantal werknemers per leeftijdsgroep en attritiestatus
age_attrition[[1, 0]].plot(kind='bar', stacked=False, ax=ax1, color=['red', 'green'], alpha=0.7)

# Secundaire y-as voor het percentage vertrokken werknemers
ax2 = ax1.twinx()
ax2.plot(age_attrition.index, age_attrition['Percentage_Verlaten'], color='blue', marker='o', linestyle='-', linewidth=2)
ax2.set_ylabel('Percentage Vertrokken Werknemers', color='blue')

# Grafiek labels
ax1.set_xlabel('Leeftijdsgroep')
ax1.set_ylabel('Aantal Werknemers')
ax1.set_title('Aantal Werknemers per Leeftijdsgroep en Attritie')
ax1.legend(['Verlaten', 'Gebleven'], title='Attritie')
ax1.grid(axis='y')

# Toon de grafiek in Streamlit
st.pyplot(fig)
max_percentage = age_attrition['Percentage_Verlaten'].max()
max_index = age_attrition['Percentage_Verlaten'].idxmax()  # Geeft de index (leeftijdsgroep) met het hoogste percentage

conclusion = f"De leeftijdsgroep **{max_index}** heeft het hoogste percentage vertrokken werknemers, " \
             f"wat kan wijzen op een verhoogde kans op attritie in deze groep."

st.write(conclusion)

#%%

# Tweede lijn toevoegen
st.markdown("<hr style='border: 2px solid #0A1172; margin: 20px 0;'>", unsafe_allow_html=True)


# In[688]:


import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Voorbeeld van het laden van je dataset
# combined_dataset = pd.read_csv('je_dataset.csv')  # Zorg ervoor dat je dataset hier wordt geladen

st.subheader('Heatmap: Effect van Salaris op Werknemersattritie')

# Lijst met unieke afdelingen
afdelingen = combined_dataset['Department'].unique().tolist()
afdelingen.append('Alle')  # Voeg de optie toe voor alle data

# Dropdown voor het selecteren van een afdeling
selected_department = st.selectbox('Selecteer een afdeling:', afdelingen)

# Filter de dataset op basis van de geselecteerde afdeling
if selected_department != 'Alle':
    filtered_data = combined_dataset[combined_dataset['Department'] == selected_department]
else:
    filtered_data = combined_dataset

# Verminder het aantal bins
salary_bins = pd.cut(filtered_data['Salary'], bins=5)

# Genereer heatmap data
heatmap_data = pd.crosstab(salary_bins, filtered_data['Attrition'])

# Pas de figuurgrootte aan om grote plot te vermijden
plt.figure(figsize=(10, 8))
sns.heatmap(heatmap_data, cmap="coolwarm", annot=True, fmt="d")

# Pas de x-ticks aan
plt.title(f'Heatmap Salaris versus Attritie - Afdeling: {selected_department}')
plt.xlabel('Attritie')
plt.ylabel('Salaris Schalen')
plt.xticks(ticks=[0.5, 1.5], labels=['Gebleven', 'Verlaten'], rotation=0)  # Aanpassen van x-ticks

# Toon de plot in Streamlit
st.pyplot(plt)
st.markdown("<hr style='border: 2px solid #0A1172; margin: 20px 0;'>", unsafe_allow_html=True)


# In[689]:


import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
st.subheader('Effect van Salaris op werknemersattritie in verband met leeftijd van werknemers')

# Voeg een slider toe voor de kolom 'Age'
min_age = combined_dataset['Age'].min()
max_age = combined_dataset['Age'].max()
selected_age = st.slider('Selecteer de leeftijd', min_value=min_age, max_value=max_age, value=(min_age, max_age))

# Filter de dataset op basis van de geselecteerde leeftijd
filtered_dataset = combined_dataset[(combined_dataset['Age'] >= selected_age[0]) & (combined_dataset['Age'] <= selected_age[1])]

# Visualiseer de relatie tussen salaris en attritie met de gefilterde dataset
plt.figure(figsize=(10, 6))
sns.scatterplot(data=filtered_dataset, x='Salary', y='Attrition', hue='Attrition', palette={0: 'green', 1: 'red'}, s=100, legend=False)

# Grafiek labels
plt.title('Salaris versus Werknemersattritie')
plt.xlabel('Salaris')
plt.ylabel('Attritie')
plt.yticks([0, 1], ['Gebleven', 'Verlaten'])
plt.grid()
st.pyplot(plt)

# Gemiddelden berekenen
mean_salary_stayed = filtered_dataset[filtered_dataset['Attrition'] == 0]['Salary'].mean()
mean_salary_left = filtered_dataset[filtered_dataset['Attrition'] == 1]['Salary'].mean()

# Conclusie
if mean_salary_stayed > mean_salary_left:
    conclusion = f"Werknemers die zijn gebleven hebben een gemiddeld salaris van **€{mean_salary_stayed:.2f}**, terwijl " \
                 f"werknemers die zijn vertrokken een gemiddeld salaris van **€{mean_salary_left:.2f}** hebben. Dit kan erop wijzen " \
                 "dat hogere salarissen mogelijk bijdragen aan een lagere attritie."
else:
    conclusion = f"Werknemers die zijn vertrokken hebben een gemiddeld salaris van **€{mean_salary_left:.2f}**, terwijl " \
                 f"werknemers die zijn gebleven een gemiddeld salaris van **€{mean_salary_stayed:.2f}** hebben. Dit kan erop wijzen " \
                 "dat lagere salarissen bijdragen aan een hogere attritie."

st.write(conclusion)
st.markdown("<hr style='border: 2px solid #0A1172; margin: 20px 0;'>", unsafe_allow_html=True)



# In[691]:


import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

st.subheader('Effect van Tevredenheid op Attritie van Werknemers')

# Stel de grootte van de figuur in
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Map Attrition 0 naar 'Gebleven' en 1 naar 'Verlaten'
attrition_labels = {0: 'Gebleven', 1: 'Verlaten'}
colors = ['green', 'red']  # Groen voor 'Gebleven', Rood voor 'Verlaten'

# Boxplot voor 'EnvironmentSatisfaction'
sns.boxplot(x='Attrition', y='EnvironmentSatisfaction', data=combined_dataset, ax=axes[0, 0], palette=colors)
axes[0, 0].set_title('Environment Satisfaction vs Attrition')
axes[0, 0].set_xticklabels([attrition_labels[int(x)] for x in axes[0, 0].get_xticks()])
median_env = combined_dataset.groupby('Attrition')['EnvironmentSatisfaction'].median()
for tick, median in zip(axes[0, 0].get_xticks(), median_env):
    axes[0, 0].plot([tick - 0.2, tick + 0.2], [median, median], color='red', lw=2)

# Boxplot voor 'JobSatisfaction'
sns.boxplot(x='Attrition', y='JobSatisfaction', data=combined_dataset, ax=axes[0, 1], palette=colors)
axes[0, 1].set_title('Job Satisfaction vs Attrition')
axes[0, 1].set_xticklabels([attrition_labels[int(x)] for x in axes[0, 1].get_xticks()])
median_job = combined_dataset.groupby('Attrition')['JobSatisfaction'].median()
for tick, median in zip(axes[0, 1].get_xticks(), median_job):
    axes[0, 1].plot([tick - 0.2, tick + 0.2], [median, median], color='red', lw=2)

# Boxplot voor 'RelationshipSatisfaction'
sns.boxplot(x='Attrition', y='RelationshipSatisfaction', data=combined_dataset, ax=axes[1, 0], palette=colors)
axes[1, 0].set_title('Relationship Satisfaction vs Attrition')
axes[1, 0].set_xticklabels([attrition_labels[int(x)] for x in axes[1, 0].get_xticks()])
median_relationship = combined_dataset.groupby('Attrition')['RelationshipSatisfaction'].median()
for tick, median in zip(axes[1, 0].get_xticks(), median_relationship):
    axes[1, 0].plot([tick - 0.2, tick + 0.2], [median, median], color='red', lw=2)

# Boxplot voor 'WorkLifeBalance'
sns.boxplot(x='Attrition', y='WorkLifeBalance', data=combined_dataset, ax=axes[1, 1], palette=colors)
axes[1, 1].set_title('Work-Life Balance vs Attrition')
axes[1, 1].set_xticklabels([attrition_labels[int(x)] for x in axes[1, 1].get_xticks()])
median_worklife = combined_dataset.groupby('Attrition')['WorkLifeBalance'].median()
for tick, median in zip(axes[1, 1].get_xticks(), median_worklife):
    axes[1, 1].plot([tick - 0.2, tick + 0.2], [median, median], color='red', lw=2)

# Layout aanpassen voor een betere weergave
plt.tight_layout()

# Toon de figuur in Streamlit
st.pyplot(fig)
st.markdown("<hr style='border: 2px solid #0A1172; margin: 20px 0;'>", unsafe_allow_html=True)


# In[692]:


from math import pi
import matplotlib.pyplot as plt
import streamlit as st

# Radar Chart setup
st.subheader('Radar Chart: Effect van Tevredenheid op Attritie van Werknemers')

categories = ['EnvironmentSatisfaction', 'JobSatisfaction', 'RelationshipSatisfaction', 'WorkLifeBalance']
attrition_group = combined_dataset.groupby('Attrition')[categories].mean()

# Map Attrition 0 naar 'Gebleven' en 1 naar 'Verlaten'
attrition_labels = {0: 'Gebleven', 1: 'Verlaten'}
attrition_colors = {0: 'green', 1: 'red'}  # Kleuren instellen

# Radar Chart plotten
fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

# Hoeken berekenen voor elke categorie
angles = [n / float(len(categories)) * 2 * pi for n in range(len(categories))]
angles += angles[:1]

# Plot voor elke Attrition categorie met aangepaste kleuren
for i, row in attrition_group.iterrows():
    values = row.values.flatten().tolist()
    values += values[:1]
    ax.plot(angles, values, label=f'Attrition: {attrition_labels[i]}', color=attrition_colors[i])  # Kleuren toevoegen
    ax.fill(angles, values, color=attrition_colors[i], alpha=0.25)  # Kleuren toevoegen

ax.set_yticklabels([])
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories)

plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
st.pyplot(fig)
st.write("De radar chart toont het effect van verschillende tevredenheidsfactoren (JobSatisfaction, EnvironmentSatisfaction, WorkLifeBalance, RelationshipSatisfaction) op de kans dat werknemers blijven of vertrekken. De lijnen van beide groepen (gebleven en vertrokken) liggen zeer dicht bij elkaar, wat suggereert dat er weinig verschil is in tevredenheidsniveaus tussen werknemers die blijven en vertrekken. Dit impliceert dat factoren zoals werkomgeving, werk-privébalans, relaties en werktevredenheid mogelijk geen grote invloed hebben op werknemersattritie in deze dataset")

st.markdown("<hr style='border: 2px solid #0A1172; margin: 20px 0;'>", unsafe_allow_html=True)


# In[698]:


import streamlit as st
import pandas as pd
import plotly.express as px
st.subheader('Impact van Afstand van Huis op Werknemersattritie')


# Voorbeeld dataset (gebruik je eigen 'combined_dataset' dataset in plaats hiervan)
np.random.seed(42)
combined_dataset = pd.DataFrame({
    'DistanceFromHome (KM)': np.random.randint(1, 50, 100),
    'Attrition': np.random.choice([0, 1], 100)
})

# Maak bins voor 'Afstand van Huis (KM)' en zorg voor hele getallen
combined_dataset['Afstand_bins'] = pd.cut(combined_dataset['DistanceFromHome (KM)'], bins=4, precision=0)

# Functie om de bin-intervallen aan te passen naar het gewenste formaat
def format_bin_labels(interval):
    return f"{int(interval.left)} - {int(interval.right)}"

# Pas de functie toe om de labels van de bins te formatteren
combined_dataset['Afstand_bins'] = combined_dataset['Afstand_bins'].apply(format_bin_labels)

# Labels aanpassen voor 'Attrition': 0 wordt 'Gebleven', 1 wordt 'Verlaten'
combined_dataset['AttritionLabel'] = combined_dataset['Attrition'].replace({0: 'Gebleven', 1: 'Verlaten'})

# Bereken het percentage per groep
grouped_data = combined_dataset.groupby(['Afstand_bins', 'AttritionLabel']).size().reset_index(name='Count')
grouped_data['Percentage'] = grouped_data.groupby('Afstand_bins')['Count'].transform(lambda x: x / x.sum() * 100)

# Maak de stacked bar chart met percentages
fig = px.histogram(grouped_data, 
                   x='Afstand_bins', 
                   y='Percentage', 
                   color='AttritionLabel',  # Gebruik de aangepaste labels hier
                   barmode='stack', 
                   labels={'Afstand_bins': 'Afstand van Huis (km)', 'Percentage': 'Percentage (%)', 'AttritionLabel': 'Attritie Status'},
                   title='Gestapelde Staafdiagram van Afstand van Huis versus Attritie (met Percentages)',
                   color_discrete_map={'Gebleven': 'lightgreen', 'Verlaten': 'red'})  # Aangepaste kleuren

# Pas de layout aan voor zwarte lijnen om de balken
fig.update_traces(marker_line_color='black', marker_line_width=1.5)

# Toon de figuur in Streamlit
st.plotly_chart(fig)


# In[49]:


import streamlit as st
import altair as alt
import pandas as pd
import numpy as np

# Maak een sample dataset
np.random.seed(42)
df = pd.DataFrame({
    'WorkLifeBalance': np.random.randint(1, 5, 100),
    'DistanceFromHome (KM)': np.random.randint(1, 50, 100),
    'Attrition': np.random.choice([0, 1], 100)
})

# Tweede lijn toevoegen voor visuele scheiding
st.markdown("<hr style='border: 2px solid #0A1172; margin: 20px 0;'>", unsafe_allow_html=True)

st.subheader('Effect van Werk-Privébalans op Werknemersattritie')

# Definieer de schuifregelaar voor het filteren op Afstand van Huis (KM)
min_afstand, max_afstand = st.slider(
    'Selecteer Afstand van Huis (in km)',
    min_value=int(df['DistanceFromHome (KM)'].min()),  
    max_value=int(df['DistanceFromHome (KM)'].max()),  
    value=(int(df['DistanceFromHome (KM)'].min()), int(df['DistanceFromHome (KM)'].max()))  
)

# Filter de data op basis van de geselecteerde afstand
gefilterde_data = df[
    (df['DistanceFromHome (KM)'] >= min_afstand) &
    (df['DistanceFromHome (KM)'] <= max_afstand)
]

# Knoppen voor het selecteren van Attrition Status
geselecteerde_status = st.radio("Selecteer Attritie Status", ['Gebleven', 'Verlaten'])

# Filter de data op basis van de geselecteerde attritie status
gefilterde_data = gefilterde_data[gefilterde_data['Attrition'] == (1 if geselecteerde_status == 'Verlaten' else 0)]

st.write(f"Gefilterde data op basis van Afstand: {min_afstand} - {max_afstand} km en Attritie Status: {geselecteerde_status}")

# Voeg labels toe voor 'Attrition' (0 wordt 'Gebleven', 1 wordt 'Verlaten')
gefilterde_data['AttritionLabel'] = gefilterde_data['Attrition'].replace({0: 'Gebleven', 1: 'Verlaten'})

# Bereken het percentage van werknemers dat in elke categorie valt
total_count = len(df)
filtered_count = len(gefilterde_data)
percentage_filtered = (filtered_count / total_count) * 100
st.write(f"Gefilterde groep vormt {percentage_filtered:.2f}% van de totale dataset.")

# Maak een heatmap met omrandingen verwijderd en elke balk een eigen kleur
chart = alt.Chart(gefilterde_data).mark_rect().encode(
    x=alt.X('WorkLifeBalance:O', title='Werk-Privébalans', axis=alt.Axis(labelAngle=0, labelFontSize=14)),  # Grotere labels onder de x-as
    y=alt.Y('DistanceFromHome (KM):Q', title='Afstand van Huis (KM)', axis=alt.Axis(labelFontSize=14)),  # Grotere labels voor de y-as
    color=alt.Color('WorkLifeBalance:N', scale=alt.Scale(scheme='category20b'), legend=alt.Legend(title="Werk-Privébalans")),  # Elke balk eigen kleur
    tooltip=['count()', 'WorkLifeBalance', 'DistanceFromHome (KM)', 'AttritionLabel']
).properties(
    title='Werk-Privébalans versus Afstand van Huis (Heatmap)',
    width=800,  # Grotere breedte van de grafiek
    height=500  # Grotere hoogte van de grafiek
)

# Toon de grafiek in Streamlit
st.altair_chart(chart, use_container_width=True)


# In[33]:


import plotly.express as px
import streamlit as st
import pandas as pd
import numpy as np

# Voorbeeld dataset aanmaken
np.random.seed(42)
combined_dataset = pd.DataFrame({
    'PromotionFrequency': np.random.randint(0, 10, 100),
    'Attrition': np.random.choice([0, 1], 100),  # 0 = Gebleven, 1 = Vertrokken
    'Department': np.random.choice(['Verkoop', 'Technologie', 'Personeelszaken'], 100)
})

# Tweede lijn toevoegen
st.markdown("<hr style='border: 2px solid #0A1172; margin: 20px 0;'>", unsafe_allow_html=True)

st.subheader('Effect van Promotiefrequentie op werknemersattritie')

# Maak de checkboxes met standaardwaarde 'Alle Afdelingen' aangevinkt
alle_afdelingen = st.checkbox("Alle Afdelingen", value=True, key="alle_afdelingen")
verkoop = st.checkbox("Verkoop", value=False, key="verkoop")
technologie = st.checkbox("Technologie", value=False, key="technologie")
hr = st.checkbox("Personeelszaken", value=False, key="hr")

# Zorg ervoor dat slechts één checkbox tegelijkertijd kan worden geselecteerd
if alle_afdelingen:
    verkoop, technologie, hr = False, False, False
elif verkoop:
    alle_afdelingen, technologie, hr = False, False, False
elif technologie:
    alle_afdelingen, verkoop, hr = False, False, False
elif hr:
    alle_afdelingen, verkoop, technologie = False, False, False

# Filter de dataset op basis van de geselecteerde checkbox
if alle_afdelingen:
    gefilterde_data = combined_dataset  # Geen filtering toepassen
elif verkoop:
    gefilterde_data = combined_dataset[combined_dataset['Department'] == 'Verkoop']
elif technologie:
    gefilterde_data = combined_dataset[combined_dataset['Department'] == 'Technologie']
elif hr:
    gefilterde_data = combined_dataset[combined_dataset['Department'] == 'Personeelszaken']

# Controleer of er gefilterde data beschikbaar is
if gefilterde_data.empty:
    st.write("Geen data beschikbaar voor de geselecteerde afdeling.")
else:
    # Groepeer de data op basis van Promotiefrequentie en Attritie, en tel het aantal werknemers in elke categorie
    gegroepeerde_data = gefilterde_data.groupby(['PromotionFrequency', 'Attrition']).size().reset_index(name='Aantal')

    # Definieer een aangepaste kleurenkaart voor Promotiefrequentie
    promotie_kleur_kaart = {
        0: 'lightblue',
        1: 'lightgreen',
        2: 'lightcoral',
        3: 'orange',
        4: 'purple',
        5: 'yellow',
        6: 'darkblue',
        7: 'pink',
        8: 'cyan',
        9: 'brown'
    }

    # Maak een donut chart met Plotly
    fig = px.pie(gegroepeerde_data, 
                 names='PromotionFrequency', 
                 values='Aantal', 
                 color='PromotionFrequency', 
                 hole=0.5,  # Maak het een donut chart
                 labels={'PromotionFrequency': 'Promotiefrequentie', 'Attrition': 'Attritie Status'},
                 color_discrete_map=promotie_kleur_kaart,  # Gebruik de aangepaste kleurenkaart
                 title='Donut Chart: Promotiefrequentie vs Attritie')

    # Pas de layout aan voor zwarte lijnen en duidelijke tekst in de legenda
    fig.update_traces(marker_line_color='black', marker_line_width=1.5, textinfo='percent+label')

    # Pas de legenda-tekst aan zodat deze duidelijk is
    fig.update_layout(
        legend_title_text='Promotiefrequentie',
        legend=dict(
            itemsizing='constant',
            title_font_size=14,
            font=dict(size=12)
        )
    )

    # Toon de donut chart in Streamlit
    st.plotly_chart(fig)

    # Voeg de optie toe om de promotiefrequentie te verhogen
    verhoog_waarde = st.number_input("Verhoog de promotiefrequentie met", min_value=0, value=1, step=1)

    # Bereken de gemiddelde promotiefrequentie voor werknemers die gebleven zijn vs. werknemers die vertrokken zijn
    gemiddelde_promotiefrequentie_gebleven = gefilterde_data[gefilterde_data['Attrition'] == 0]['PromotionFrequency'].mean() + verhoog_waarde
    gemiddelde_promotiefrequentie_vertrokken = gefilterde_data[gefilterde_data['Attrition'] == 1]['PromotionFrequency'].mean() + verhoog_waarde

    # Toon conclusie gebaseerd op de gemiddelde promotiefrequentie
    if gemiddelde_promotiefrequentie_gebleven > gemiddelde_promotiefrequentie_vertrokken:
        conclusie = f"Werknemers die zijn gebleven hebben een gemiddelde promotiefrequentie van **{gemiddelde_promotiefrequentie_gebleven:.2f}**, terwijl " \
                    f"werknemers die zijn vertrokken een gemiddelde promotiefrequentie hebben van **{gemiddelde_promotiefrequentie_vertrokken:.2f}**. Dit suggereert dat een hogere promotiefrequentie " \
                    "kan bijdragen aan een lagere attritie."
    else:
        conclusie = f"Werknemers die zijn vertrokken hebben een gemiddelde promotiefrequentie van **{gemiddelde_promotiefrequentie_vertrokken:.2f}**, terwijl " \
                    f"werknemers die zijn gebleven een gemiddelde promotiefrequentie hebben van **{gemiddelde_promotiefrequentie_gebleven:.2f}**. Dit suggereert dat een lagere promotiefrequentie " \
                    "kan bijdragen aan een hogere attritie."

    st.write(conclusie)


# In[35]:


import streamlit as st
import plotly.graph_objects as go

# Functie om de kans op promotie te berekenen met dynamische gewichten
def calculate_promotion_chance(
    YearsAtCompany, YearsInMostRecentRole, YearsSinceLastPromotion,
    TrainingOpportunitiesWithinYear, SelfRating, ManagerRating, 
    w1, w2, w3, w4, w5, w6):
    # Bereken kans op promotie
    chance = (
        w1 * YearsAtCompany
        + w2 * YearsInMostRecentRole
        + w3 * (1 / (YearsSinceLastPromotion + 1))  # Inverse relatie met promotiekans
        + w4 * TrainingOpportunitiesWithinYear
        + w5 * (SelfRating / 10)  # Rating als verhouding tussen 0 en 1
        + w6 * (ManagerRating / 10)  # Rating als verhouding tussen 0 en 1
    ) * 100  # Omzetten naar percentage

    return max(0.0, min(100.0, chance))  # Zorg ervoor dat de kans tussen 0.0 en 100.0 blijft

# Streamlit UI voor gewichten en invoer
st.subheader("Calculator voor Promotiekans")
st.markdown("<h3><b>Bereken je kans op promotie op basis van je invoer.</b></h3>", unsafe_allow_html=True)

# Plaats invoervelden in rijen van twee met behulp van columns
with st.form(key="promotion_form"):
    col1, col2 = st.columns(2)
    with col1:
        YearsAtCompany = st.number_input("Jaren bij het Bedrijf", min_value=0, max_value=30, value=5)
    with col2:
        YearsInMostRecentRole = st.number_input("Jaren in Huidige Rol", min_value=0, max_value=30, value=2)

    col3, col4 = st.columns(2)
    with col3:
        YearsSinceLastPromotion = st.number_input("Jaren sinds Laatste Promotie", min_value=0, max_value=30, value=1)
    with col4:
        TrainingOpportunitiesWithinYear = st.number_input("Trainingen dit Jaar", min_value=0, max_value=5, value=2)

    col5, col6 = st.columns(2)
    with col5:
        SelfRating = st.number_input("Zelfbeoordeling (1-10)", min_value=1, max_value=10, value=5)
    with col6:
        ManagerRating = st.number_input("Beoordeling Manager (1-10)", min_value=1, max_value=10, value=5)

    # Gewichten voor elke factor invoeren
    st.write("### Stel de gewichten in voor elk criterium")
    w1 = st.slider("Gewicht: Jaren bij het Bedrijf", min_value=0.0, max_value=1.0, value=0.3)
    w2 = st.slider("Gewicht: Jaren in Huidige Rol", min_value=0.0, max_value=1.0, value=0.2)
    w3 = st.slider("Gewicht: Jaren sinds Laatste Promotie", min_value=0.0, max_value=1.0, value=0.1)
    w4 = st.slider("Gewicht: Trainingen dit Jaar", min_value=0.0, max_value=1.0, value=0.2)
    w5 = st.slider("Gewicht: Zelfbeoordeling", min_value=0.0, max_value=1.0, value=0.1)
    w6 = st.slider("Gewicht: Beoordeling Manager", min_value=0.0, max_value=1.0, value=0.1)

    # Knop voor berekening
    submit_button = st.form_submit_button("Bereken Promotiekans")

# Bereken en visualiseer promotiekans
if submit_button:
    promotion_chance = calculate_promotion_chance(
        YearsAtCompany, YearsInMostRecentRole, YearsSinceLastPromotion,
        TrainingOpportunitiesWithinYear, SelfRating, ManagerRating, 
        w1, w2, w3, w4, w5, w6)

    # Toon het resultaat
    st.write(f"Je promotiekans is: **{promotion_chance:.2f}%**")

    # Visualiseer promotiekans met een gauge chart
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=promotion_chance,
        title={'text': "Promotiekans"},
        gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "green"}}
    ))

    st.plotly_chart(fig)

    # AI-advies gebaseerd op promotiekans
    if promotion_chance < 40:
        st.write("Je promotiekans is relatief laag. Overweeg om je trainingskansen te verbeteren of aan je zelf-/managerbeoordelingen te werken.")
    elif promotion_chance < 70:
        st.write("Je promotiekans is matig. Consequente prestaties en enkele extra inspanningen kunnen je kansen verhogen.")
    else:
        st.write("Geweldig! Je promotiekans is hoog. Blijf je prestaties en communicatie met je manager op peil houden.")

    # Vergelijk de promotiekans met die van collega's
    avg_peer_promotion_chance = 65  # Stel dat dit het gemiddelde is van alle werknemers
    if promotion_chance > avg_peer_promotion_chance:
        st.write(f"Je promotiekans is **{promotion_chance:.2f}%**, wat hoger is dan het gemiddelde van collega's ({avg_peer_promotion_chance}%).")
    else:
        st.write(f"Je promotiekans is **{promotion_chance:.2f}%**, wat lager is dan het gemiddelde van collega's ({avg_peer_promotion_chance}%).")

# Knop om invoer te resetten
if st.button("Reset Invoer"):
    st.experimental_rerun()



# In[37]:




# In[700]:


employee = pd.read_csv('Employee.csv', delimiter = ',')
performancerate = pd.read_csv('PerformanceRating.csv', delimiter = ',')

#combineren van de datasets 
performancerate['ReviewDate'] = pd.to_datetime(performancerate['ReviewDate'])  # Zorg dat de aanstellingsdatum in datetime-formaat is
recent_performance = performancerate.loc[performancerate.groupby('EmployeeID')['ReviewDate'].idxmax()]
combined_dataset =  pd.merge(employee, recent_performance, on='EmployeeID', how='left')


# In[702]:


import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd


# Titel en beschrijving van de voorspelling
st.subheader("Voorspelling van werknemersattritie")
st.write("""
In deze sectie gebruiken we een Random Forest Classifier om te voorspellen of een werknemer het bedrijf zal verlaten 
(werknemersattritie) op basis van verschillende factoren zoals leeftijd, werktevredenheid, balans tussen werk en privé, enzovoort.
""")

# Select relevant features for prediction
features = ['JobSatisfaction', 'WorkLifeBalance', 'Age', 'YearsAtCompany', 
            'YearsSinceLastPromotion', 'DistanceFromHome (KM)', 'Salary']
X = combined_dataset[features]
y = combined_dataset['Attrition']

# Train-test split (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build and evaluate a Random Forest model
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# Random Forest metrics
rf_accuracy = accuracy_score(y_test, y_pred_rf)
rf_report = classification_report(y_test, y_pred_rf, output_dict=True)

# Weergeven van de accuracy score
st.subheader("Modelresultaten")
st.write(f"De nauwkeurigheid van het Random Forest-model is: **{rf_accuracy:.2f}**")

# Weergeven van het classification report
st.write("Hieronder vind je het classificatierapport, dat de prestaties van het model per categorie (verbleven of vertrokken) weergeeft:")

# Converteer het classification report naar een DataFrame voor overzichtelijke weergave
rf_report_df = pd.DataFrame(rf_report).transpose()
st.dataframe(rf_report_df)

# Optioneel: Extra uitleg van het classificatierapport
st.write("""
Het classificatierapport toont de prestaties van het model in termen van precision, recall en F1-score voor beide klassen (werknemers die blijven en werknemers die vertrekken). 
Een hogere F1-score betekent dat het model beter presteert in het voorspellen van die klasse.
""")



# In[ ]:





# In[ ]:





# In[ ]:




