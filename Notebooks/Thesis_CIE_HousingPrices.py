# region Libraries

import datetime
import json
import os
from typing import Type

import contextily as cx
import fiona
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import prince
import scipy.cluster.hierarchy as sch
import scipy.sparse as sp
import scipy.stats as stats
import seaborn as sns
import spopt
import statsmodels
import statsmodels.api as sm
from libpysal.weights import Queen
from pysal.lib import weights
from pysal.model import spreg
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise as skm
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.preprocessing import StandardScaler
from splot.esda import moran_scatterplot
from splot.libpysal import plot_spatial_weights
from spopt.region import MaxPHeuristic as MaxP
from statsmodels.graphics.gofplots import ProbPlot, qqplot
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import maybe_unwrap_results
from ydata_profiling import ProfileReport

# endregion


# region Data Loading Functions
def load_json(filename):
    """Loads JSON data from a file and normalizes it into a Pandas DataFrame."""
    with open(filename, encoding="utf-8", errors="ignore") as f:
        data = json.load(f)
    return pd.json_normalize(data)


def load_csv(filename, **kwargs):
    """Loads CSV data from a file into a Pandas DataFrame."""
    return pd.read_csv(filename, **kwargs)


# Dictionary to Store DataFrames
data_files = {
    "energy_certifications": ("Data/energy_certifications.json", load_json),
    "conditions": ("Data/conditions.json", load_json),
    "list_ids": ("Data/list_ids.json", load_json),
    "parishes": ("Data/parishes.json", load_json),
    "sources": ("Data/sources.json", load_json),
    "types": ("Data/types.json", load_json),
    "typologies": ("Data/typologies.json", load_json),
    "agueda": ("Data/agueda.json", load_json),
    "albergaria": ("Data/Albergaria-a-Velha.json", load_json),
    "anadia": ("Data/Anadia.json", load_json),
    "aveiro": ("Data/aveiro.json", load_json),
    "estarreja": ("Data/Estarreja.json", load_json),
    "ilhavo": ("Data/ilhavo.json", load_json),
    "murtosa": ("Data/Murtosa.json", load_json),
    "oliveira": ("Data/Oliveira do Bairro.json", load_json),
    "ovar": ("Data/Ovar.json", load_json),
    "sever": ("Data/Sever do Vouga.json", load_json),
    "vagos": ("Data/Vagos.json", load_json),
    "casasapo_2000_2009": (
        "Data/BD_JanelaDigital_2000to2009_withHeaders.txt",
        load_csv,
        {"sep": "\t"},
    ),
    "casasapo_transactions": (
        "Data/BDAVRILH_DONUT&PhdPB_VENDA_V20150624.csv",
        load_csv,
        {"encoding": "latin", "sep": ";"},
    ),
    "cluster_data_v1": (
        "Data/BGRI11_AVRILH_M1LABELS_LP_AdjMatGeoNetDist_v1.csv",
        load_csv,
        {"encoding": "latin", "sep": ","},
    ),
    "cluster_data_v2": (
        "Data/BGRI11_AVRILH_M1LABELS_LP_AdjMatGeoNetDist_v2.csv",
        load_csv,
        {"encoding": "latin", "sep": ","},
    ),
    "zones_csra": ("Data/zones_CSRA_gdf_descodifica.csv", load_csv, {"sep": ";"}),
    "al_establishments": ("Data/Estabelecimentos_de_AL.csv", load_csv, {"sep": ","}),
}

dataframes = {}
for name, (filename, load_func, *args) in data_files.items():
    if args:
        dataframes[name] = load_func(filename, **args[0])
    else:
        dataframes[name] = load_func(filename)
# endregion

# region Data Loading & Preparation - Prime Yeild
# Concatenate DataFrames for PY_data (Prime Yield data, post intervention)

PY_data = pd.concat(
    [
        dataframes["agueda"],
        dataframes["albergaria"],
        dataframes["anadia"],
        dataframes["aveiro"],
        dataframes["estarreja"],
        dataframes["ilhavo"],
        dataframes["murtosa"],
        dataframes["oliveira"],
        dataframes["ovar"],
        dataframes["sever"],
        dataframes["vagos"],
    ],
    ignore_index=True,
)

# remove data points without latitude and longitude
PY_data = PY_data[
    ~pd.isnull(PY_data["ADD_LONGITUDE"]) | ~pd.isnull(PY_data["ADD_LATITUDE"])
]

# Filter dwellings that are houses, apartments or luxury / drop unnecessary columns
PY_data = PY_data[
    (PY_data["TYPE_ID"] == 1) | (PY_data["TYPE_ID"] == 2) | (PY_data["TYPE_ID"] == 10)
]  # house+apartment+luxury

# Drop unnecessary columns
PY_data.drop(
    [
        "BUSINESS_TYPE",
        "SMALL_DESCRIPTION",
        "LONG_DESCRIPTION",
        "ADD_POSTAL_CODE",
        "ADD_ADDRESS",
        "SOURCE_CODE",
        "SOURCE_URL",
        "LAST_UPDATE",
        "SELL_DATE",
        "ENERGY_CERT_ID",
        "STATUS",
        "MAP_IMAGE",
        "MAP_COUNTRY_IMAGE",
        "REGISTRY",
        "BUILDING_N",
        "FRACTION",
        "FINANCES",
        "MATRIX_ARTICLE",
        "WCS",
        "MUNICIPALITY_ID",
        "PARISH_ID",
        "AREA_GROSS",
        "WIP_STATUS_PERCENT",
        "RENT_PRICE",
        "WIP_STATUS",
        "PATRIMONIAL_VALUE",
        "PATRIMONIAL_VALUE_DATE",
        "TASK_ID",
        "MAP_IMAGE3",
        "MAP_IMAGE2",
        "PROPERTY_DATA_FROM_API",
        "ADV_DATE",
    ],
    axis=1,
    inplace=True,
)

fig = px.scatter_mapbox(
    PY_data,
    lon="ADD_LONGITUDE",
    lat="ADD_LATITUDE",
    mapbox_style="open-street-map",
    center={"lat": 40.6, "lon": -8.5},
    zoom=8,
    height=800,
    width=600,
    hover_name="ID",
    hover_data=["CONSTRUCTION_YEAR", "CURRENT_PRICE"],
)
fig.show()
# endregion

# region Data Loading & Preparation - CasaSapo

# Load casasapo data
casasapo = dataframes["casasapo_transactions"]

# Transform Natureza in 1 and 2 (1=apartment, 2=house)
casasapo["Natureza"].replace("Apartamento", 1, inplace=True)
casasapo["Natureza"].replace("Andar de Moradia", 1, inplace=True)
casasapo["Natureza"].replace("Moradia", 2, inplace=True)
casasapo["Natureza"].replace("Moradia Geminada", 2, inplace=True)
casasapo["Natureza"].replace("Moradia Isolada", 2, inplace=True)
casasapo["Natureza"].replace("Moradia em Banda", 2, inplace=True)
casasapo["Natureza"] = casasapo["Natureza"].astype(int)

# Match "tipologia" with "classificação" from PrimeYield
# Scale from 0 to 7, where T0 is 0 and T6+ is 7
casasapo["Tipologia"].replace("T2", 3, inplace=True)
casasapo["Tipologia"].replace("T3", 4, inplace=True)
casasapo["Tipologia"].replace("T4", 5, inplace=True)
casasapo["Tipologia"].replace("T2 Duplex", 3, inplace=True)
casasapo["Tipologia"].replace("T1", 2, inplace=True)
casasapo["Tipologia"].replace("T3 Duplex", 4, inplace=True)
casasapo["Tipologia"].replace("T5", 6, inplace=True)
casasapo["Tipologia"].replace("T4+1", 5, inplace=True)
casasapo["Tipologia"].replace("T0", 1, inplace=True)
casasapo["Tipologia"].replace("T2+1", 3, inplace=True)
casasapo["Tipologia"].replace("T5+1", 6, inplace=True)
casasapo["Tipologia"].replace("T3+1", 4, inplace=True)
casasapo["Tipologia"].replace("T1+1", 2, inplace=True)
casasapo["Tipologia"].replace("T4 Duplex", 5, inplace=True)
casasapo["Tipologia"].replace("T1 Duplex", 2, inplace=True)
casasapo["Tipologia"].replace("T5 Duplex", 6, inplace=True)
casasapo["Tipologia"].replace("T2+1 Duplex", 3, inplace=True)
casasapo["Tipologia"].replace("T6", 7, inplace=True)
casasapo["Tipologia"].replace("T3+1 Duplex", 4, inplace=True)
casasapo["Tipologia"].replace("T1+1 Duplex", 2, inplace=True)
casasapo["Tipologia"].replace("T0 Duplex", 1, inplace=True)
casasapo["Tipologia"].replace("T2+2 Duplex", 3, inplace=True)
casasapo["Tipologia"].replace("T4+1 Duplex", 5, inplace=True)
casasapo["Tipologia"].replace("T2+2", 3, inplace=True)
casasapo["Tipologia"].replace("T4+2", 5, inplace=True)
casasapo["Tipologia"].replace("T4+2 Duplex", 5, inplace=True)
casasapo["Tipologia"].replace("T11", 7, inplace=True)
casasapo["Tipologia"].replace("T1+2", 2, inplace=True)
casasapo["Tipologia"].replace("T3+2", 4, inplace=True)
casasapo["Tipologia"].replace("T0+1", 1, inplace=True)
casasapo["Tipologia"].replace("T8", 7, inplace=True)
casasapo["Tipologia"] = casasapo["Tipologia"].astype(int)

# adapt typology to casasapo classification
PY_data["TYPOLOGY_ID"].replace(8, 7, inplace=True)
PY_data["TYPOLOGY_ID"].replace(9, 7, inplace=True)
PY_data["TYPOLOGY_ID"].replace(10, 7, inplace=True)
PY_data["TYPOLOGY_ID"].replace(11, 7, inplace=True)
PY_data["TYPOLOGY_ID"].replace(12, 7, inplace=True)
PY_data["TYPOLOGY_ID"].replace(13, 7, inplace=True)
PY_data["TYPOLOGY_ID"] = PY_data["TYPOLOGY_ID"].astype(int)

# update preservacao: 1 is new, 2 is used up to 10 years, 3 is used up to 25 years, 4 is used for more than 25 years, 5 is under construction
casasapo["Preservacao"].replace("Novo", 1, inplace=True)
casasapo["Preservacao"].replace("Usado até 10 anos", 2, inplace=True)
casasapo["Preservacao"].replace("Usado de 10 a 25 anos", 3, inplace=True)
casasapo["Preservacao"].replace("Usado com mais de 25 anos", 4, inplace=True)
casasapo["Preservacao"].replace("Em construcao/projecto", 5, inplace=True)
casasapo["Preservacao"] = casasapo["Preservacao"].astype(int)

# check no. of dwellings per year of announcement
casasapo["DTA_ano"] = casasapo["DTA_ano"].astype("Int64")
casasapo["DTA_ano"].describe()

# drop unnecessary columns
casasapo.drop(
    [
        "Estado",
        "Negocio",
        "Concelho",
        "Freguesia",
        "Zona",
        "ZonaFreg_CSREA",
        "VA1011_PrecoInicial_M2Area",
        "VA0511_TOM",
        "VA0512_LOGTOM",
        "VA0610_LOG_CodigoTipologia",
        "VA0111_LOG_Area",
        "VA0120_Area_Avaliacao",
        "VA02101_Apartamento",
        "VA02102_Moradia",
        "VA0410_ConstrucaoProjecto",
        "VA0420_0_Novo",
        "VA0431_Usado_Ate10",
        "VA0432_Usado_1025",
        "VA0433_Usado_mais25",
        "VA0430_0_Usados",
        "VA0420_1_NovosampRecuperados",
        "VA0430_1_Usados_Ate10ampRecuperados",
        "DTA_trimestre",
        "ZONAIII_FINAL",
        "VD02_AquecimentoCentral",
        "VD01_ArCondicionado",
        "VD03_Arrecadacao",
        "VD04_Arrumos",
        "VD05_Aspiracao",
        "VD06_Churrasqueira",
        "VD07_Climatizacao",
        "VD08_Despensa",
        "VD09_Domotica",
        "VD10_Estacionamento",
        "VD11_Garagem",
        "VD12_Hidromassagem",
        "VD13_Jacuzzi",
        "VD14_Jardim",
        "VD15_Kitchenette",
        "VD16_Lareira",
        "VD17_Lavandaria",
        "VD18_Logradouro",
        "VD19_Marquise",
        "VD20_Mobilado",
        "VD21_Patio",
        "VD22_Porteiro",
        "VD23_Recuperador",
        "VD24_Sauna",
        "VD25_Sotao",
        "VD26_Terraco",
        "VD27_Varanda",
        "AF_Basic_F1_LivingSpace",
        "AF_Basic_F2_Preservation_ConstructionNew",
        "AF_Basic_F3_Preservation_UsedTo10yrs",
        "AF_Basic_F4_Preservation_Used1025yrs",
        "AF_Basic_F5_Preservation_UsedMore25yrs",
        "AF_All_F1_LivingSpace",
        "AF_All_F2_Preservation_Used1025yrs",
        "AF_All_F3_Ads_Confort",
        "AF_All_F4_Ads_AditionalSpaceA",
        "AF_All_F5_Preservation_ConstructionNew",
        "AF_All_F6_Preservation_UsedTo10yrs",
        "AF_All_F7_Ads_Richness",
        "AF_All_F8_Ads_AditionalSpaceB",
        "AF_All_F9_Ads_Preservation_UsedMore25yrs",
        "AF_Ads_FAC1_3",
        "AF_Ads_FAC2_3",
        "AF_Ads_FAC3_3",
        "AF_Ads_FAC4_3",
        "AF_Ads_FAC5_3",
        "AF_Ads_FAC6_3",
        "FAC1_1",
        "FAC2_1",
        "FAC3_1",
        "FAC4_1",
        "FAC5_1",
        "C_11_Submercados",
        "C_12_Uts",
        "C14_Mixed",
        "C_11_SubF",
        "C12_MixedF",
        "C07_MixedFv3",
    ],
    axis=1,
    inplace=True,
)
# endregion

# region ETL

### 4.1 Casa_Sapo Data

casasapo_total = dataframes["casasapo_2000_2009"]

casasapo_total = casasapo_total[["ID", "Zone_ID"]]

# convert to Int64 for merging
casasapo_total = casasapo_total.astype({"Zone_ID": "Int64"})
# Bring Zine_ID info to our casasapo dataset
casasapo = casasapo.merge(casasapo_total, on="ID", how="left")
casasapo.shape
decodifier = dataframes["zones_csra"]
# keep only the zone_ID and cluster_ID columns
decodifier = decodifier[["Zone_ID", "ZONE_CS_newID_132Z_CATint"]]

# bring Cluster ID to our casasapo dataset (this will be Cluster_LP info.)
casasapo = casasapo.merge(decodifier, on="Zone_ID", how="left")

# rename Cluster columns - Cluster_new based on the new clusters calculated, cluster_old based on the criteria for PB PhD Thesis
casasapo.rename(
    columns={"ZONE_CS_newID_132Z_CATint": "Cluster_LP", "ZONA_D": "Cluster_old"},
    inplace=True,
)

# filter out dwellings with no Cluster ID
casasapo = casasapo[~casasapo["Cluster_LP"].isnull()]

# histogram  for Casasapo Price
sns.displot(casasapo["VA1010_PrecoInicial"], bins=30, kde=True)

# vamos aplicar o log (neste caso, o PB já o fez e vou usar os dados dele)
# histogram  for Casasapo Log Price per square meter
sns.displot(casasapo["VA1012_LOG_PrecoInicial_M2Area"], bins=30, kde=True)

#### 4.1.1 Fix problem with Cluster_LP (label propagation problems)


casasapo[casasapo.isnull().any(axis=1)]

casasapo["Cluster_LP"] = casasapo["Cluster_LP"].astype("int64")
# # This routine doesn't work properly, so it was necessary to manually change the values

# mapping = {
#     22: [22, 156],
#     26: [26, 157],
#     30: [30, 158, 151],
#     40: [40, 159],
#     41: [41, 160],
#     43: [43, 161],
#     60: [162, 152, 60],
#     63: [63, 163, 153],
#     64: [64, 164, 154, 150],
#     103: [103, 165],
#     107: [107, 166, 155],
# }

# casasapo['Cluster_LP'] = casasapo['Cluster_LP'].astype(int)

# np.random.seed(3)


# filtered_mapping = {key: value for key, value in mapping.items() if key in casasapo['Cluster_LP'].unique()}

# # Function to assign new Cluster_LP based on the filtered_mapping dictionary
# def assign_new_cluster_lp(row):
#     if row['Cluster_LP'] in filtered_mapping:
#         new_values = filtered_mapping[row['Cluster_LP']]
#         return np.random.choice(new_values)
#     return row['Cluster_LP']

# # Apply the function to the DataFrame to assign the new Cluster_LP values
# casasapo['New_Cluster_LP'] = casasapo.apply(assign_new_cluster_lp, axis=1)

# # Replace the original 'Cluster_LP' column with the new values
# casasapo['Cluster_LP'] = casasapo['New_Cluster_LP']

# # Drop the intermediate 'New_Cluster_LP' column
# casasapo.drop(columns=['New_Cluster_LP'], inplace=True)


# print(casasapo.shape)
# print(
#     len(casasapo[casasapo["Cluster_LP"] == 22]),
#     len(casasapo[casasapo["Cluster_LP"] == 156]),
# )
# print(
#     len(casasapo[casasapo["Cluster_LP"] == 26]),
#     len(casasapo[casasapo["Cluster_LP"] == 157]),
# )
# print(
#     len(casasapo[casasapo["Cluster_LP"] == 30]),
#     len(casasapo[casasapo["Cluster_LP"] == 151]),
#     len(casasapo[casasapo["Cluster_LP"] == 158]),
# )
# print(
#     len(casasapo[casasapo["Cluster_LP"] == 40]),
#     len(casasapo[casasapo["Cluster_LP"] == 159]),
# )
# print(
#     len(casasapo[casasapo["Cluster_LP"] == 41]),
#     len(casasapo[casasapo["Cluster_LP"] == 160]),
# )
# print(
#     len(casasapo[casasapo["Cluster_LP"] == 43]),
#     len(casasapo[casasapo["Cluster_LP"] == 161]),
# )
# print(
#     len(casasapo[casasapo["Cluster_LP"] == 60]),
#     len(casasapo[casasapo["Cluster_LP"] == 152]),
#     len(casasapo[casasapo["Cluster_LP"] == 162]),
# )
# print(
#     len(casasapo[casasapo["Cluster_LP"] == 63]),
#     len(casasapo[casasapo["Cluster_LP"] == 153]),
#     len(casasapo[casasapo["Cluster_LP"] == 163]),
# )
# print(
#     len(casasapo[casasapo["Cluster_LP"] == 64]),
#     len(casasapo[casasapo["Cluster_LP"] == 154]),
#     len(casasapo[casasapo["Cluster_LP"] == 164]),
#     len(casasapo[casasapo["Cluster_LP"] == 150]),
# )
# print(
#     len(casasapo[casasapo["Cluster_LP"] == 107]),
#     len(casasapo[casasapo["Cluster_LP"] == 166]),
#     len(casasapo[casasapo["Cluster_LP"] == 155]),
# )

# split dwelling from Cluster 22 between 22 (2) and 156 (1)
# casasapo[casasapo['Cluster_LP']==22].sample(n=1, random_state=1)
casasapo.at[4730, "Cluster_LP"] = 156

# split dwelling from Cluster 26 between 26 (39) and 157 (39)
# casasapo[casasapo['Cluster_LP']==26].sample(n=39, random_state=1)
casasapo.at[78, "Cluster_LP"] = 157
casasapo.at[370, "Cluster_LP"] = 157
casasapo.at[1087, "Cluster_LP"] = 157
casasapo.at[2106, "Cluster_LP"] = 157
casasapo.at[2572, "Cluster_LP"] = 157
casasapo.at[3153, "Cluster_LP"] = 157
casasapo.at[3344, "Cluster_LP"] = 157
casasapo.at[3794, "Cluster_LP"] = 157
casasapo.at[3992, "Cluster_LP"] = 157
casasapo.at[4182, "Cluster_LP"] = 157
casasapo.at[4369, "Cluster_LP"] = 157
casasapo.at[4408, "Cluster_LP"] = 157
casasapo.at[4871, "Cluster_LP"] = 157
casasapo.at[4888, "Cluster_LP"] = 157
casasapo.at[4893, "Cluster_LP"] = 157
casasapo.at[4906, "Cluster_LP"] = 157
casasapo.at[4933, "Cluster_LP"] = 157
casasapo.at[4963, "Cluster_LP"] = 157
casasapo.at[4965, "Cluster_LP"] = 157
casasapo.at[4969, "Cluster_LP"] = 157
casasapo.at[4975, "Cluster_LP"] = 157
casasapo.at[4979, "Cluster_LP"] = 157
casasapo.at[4983, "Cluster_LP"] = 157
casasapo.at[5055, "Cluster_LP"] = 157
casasapo.at[5192, "Cluster_LP"] = 157
casasapo.at[5391, "Cluster_LP"] = 157
casasapo.at[5393, "Cluster_LP"] = 157
casasapo.at[5401, "Cluster_LP"] = 157
casasapo.at[5403, "Cluster_LP"] = 157
casasapo.at[5589, "Cluster_LP"] = 157
casasapo.at[5592, "Cluster_LP"] = 157
casasapo.at[5597, "Cluster_LP"] = 157
casasapo.at[5708, "Cluster_LP"] = 157
casasapo.at[5722, "Cluster_LP"] = 157
casasapo.at[5729, "Cluster_LP"] = 157
casasapo.at[5737, "Cluster_LP"] = 157
casasapo.at[6018, "Cluster_LP"] = 157
casasapo.at[6532, "Cluster_LP"] = 157
casasapo.at[6810, "Cluster_LP"] = 157

# split dwelling from Cluster 30 between 30 (10), 151 (4), 158 (4)
# casasapo[casasapo['Cluster_LP']==30].sample(n=8, random_state=1)
casasapo.at[62, "Cluster_LP"] = 151
casasapo.at[699, "Cluster_LP"] = 158
casasapo.at[1927, "Cluster_LP"] = 151
casasapo.at[2148, "Cluster_LP"] = 158
casasapo.at[2707, "Cluster_LP"] = 151
casasapo.at[3808, "Cluster_LP"] = 158
casasapo.at[4648, "Cluster_LP"] = 151
casasapo.at[5862, "Cluster_LP"] = 158

# split dwelling from Cluster 40 between 40 (10) and 159 (10)
# casasapo[casasapo['Cluster_LP']==40].sample(n=10, random_state=1)
casasapo.at[3951, "Cluster_LP"] = 159
casasapo.at[3650, "Cluster_LP"] = 159
casasapo.at[4374, "Cluster_LP"] = 159
casasapo.at[5154, "Cluster_LP"] = 159
casasapo.at[6635, "Cluster_LP"] = 159
casasapo.at[4381, "Cluster_LP"] = 159
casasapo.at[7105, "Cluster_LP"] = 159
casasapo.at[4098, "Cluster_LP"] = 159
casasapo.at[7130, "Cluster_LP"] = 159
casasapo.at[7177, "Cluster_LP"] = 159

# split dwelling from Cluster 41 between 41 (75) and 160 (10)
# casasapo[casasapo['Cluster_LP']==41].sample(n=10, random_state=1)
casasapo.at[4455, "Cluster_LP"] = 160
casasapo.at[4663, "Cluster_LP"] = 160
casasapo.at[2947, "Cluster_LP"] = 160
casasapo.at[4534, "Cluster_LP"] = 160
casasapo.at[3677, "Cluster_LP"] = 160
casasapo.at[7017, "Cluster_LP"] = 160
casasapo.at[5338, "Cluster_LP"] = 160
casasapo.at[6213, "Cluster_LP"] = 160
casasapo.at[7280, "Cluster_LP"] = 160
casasapo.at[5225, "Cluster_LP"] = 160

# split dwelling from Cluster 43 between 43 (15) and 161 (14)
# casasapo[casasapo['Cluster_LP']==43].sample(n=14, random_state=1)
casasapo.at[573, "Cluster_LP"] = 161
casasapo.at[5227, "Cluster_LP"] = 161
casasapo.at[3307, "Cluster_LP"] = 161
casasapo.at[4801, "Cluster_LP"] = 161
casasapo.at[5684, "Cluster_LP"] = 161
casasapo.at[3347, "Cluster_LP"] = 161
casasapo.at[423, "Cluster_LP"] = 161
casasapo.at[436, "Cluster_LP"] = 161
casasapo.at[5677, "Cluster_LP"] = 161
casasapo.at[5345, "Cluster_LP"] = 161
casasapo.at[425, "Cluster_LP"] = 161
casasapo.at[222, "Cluster_LP"] = 161
casasapo.at[5683, "Cluster_LP"] = 161
casasapo.at[431, "Cluster_LP"] = 161

# split dwelling from Cluster 60 between 60 (15), 152 (1), 162 (1)
# casasapo[casasapo['Cluster_LP']==60].sample(n=2, random_state=1)
casasapo.at[1682, "Cluster_LP"] = 152
casasapo.at[1540, "Cluster_LP"] = 162

# split dwelling from Cluster 63 between 63 (15), 153 (1), 163 (2)
# casasapo[casasapo['Cluster_LP']==63].sample(n=3, random_state=1)
casasapo.at[6946, "Cluster_LP"] = 153
casasapo.at[6752, "Cluster_LP"] = 163
casasapo.at[7264, "Cluster_LP"] = 163

# split dwelling from Cluster 64 between 64 (1), 154 (2), 164 (2), 150 (2)
# casasapo[casasapo['Cluster_LP']==64].sample(n=6, random_state=1)
casasapo.at[7121, "Cluster_LP"] = 154
casasapo.at[2008, "Cluster_LP"] = 164
casasapo.at[1004, "Cluster_LP"] = 150
casasapo.at[970, "Cluster_LP"] = 154
casasapo.at[6514, "Cluster_LP"] = 164
casasapo.at[6028, "Cluster_LP"] = 150

# split dwelling from Cluster 107 between 107 (0), 166 (1), 155 (1)
# casasapo[casasapo['Cluster_LP']==107].sample(n=2, random_state=1)
casasapo.at[6408, "Cluster_LP"] = 166
casasapo.at[6695, "Cluster_LP"] = 155

# print(casasapo.shape)
# print(
#     len(casasapo[casasapo["Cluster_LP"] == 22]),
#     len(casasapo[casasapo["Cluster_LP"] == 156]),
# )
# print(
#     len(casasapo[casasapo["Cluster_LP"] == 26]),
#     len(casasapo[casasapo["Cluster_LP"] == 157]),
# )
# print(
#     len(casasapo[casasapo["Cluster_LP"] == 30]),
#     len(casasapo[casasapo["Cluster_LP"] == 151]),
#     len(casasapo[casasapo["Cluster_LP"] == 158]),
# )
# print(
#     len(casasapo[casasapo["Cluster_LP"] == 40]),
#     len(casasapo[casasapo["Cluster_LP"] == 159]),
# )
# print(
#     len(casasapo[casasapo["Cluster_LP"] == 41]),
#     len(casasapo[casasapo["Cluster_LP"] == 160]),
# )
# print(
#     len(casasapo[casasapo["Cluster_LP"] == 43]),
#     len(casasapo[casasapo["Cluster_LP"] == 161]),
# )
# print(
#     len(casasapo[casasapo["Cluster_LP"] == 60]),
#     len(casasapo[casasapo["Cluster_LP"] == 152]),
#     len(casasapo[casasapo["Cluster_LP"] == 162]),
# )
# print(
#     len(casasapo[casasapo["Cluster_LP"] == 63]),
#     len(casasapo[casasapo["Cluster_LP"] == 153]),
#     len(casasapo[casasapo["Cluster_LP"] == 163]),
# )
# print(
#     len(casasapo[casasapo["Cluster_LP"] == 64]),
#     len(casasapo[casasapo["Cluster_LP"] == 154]),
#     len(casasapo[casasapo["Cluster_LP"] == 164]),
#     len(casasapo[casasapo["Cluster_LP"] == 150]),
# )
# print(
#     len(casasapo[casasapo["Cluster_LP"] == 107]),
#     len(casasapo[casasapo["Cluster_LP"] == 166]),
#     len(casasapo[casasapo["Cluster_LP"] == 155]),
# )
casasapo[casasapo.isnull().any(axis=1)]

### 4.1.2 Cluster Data (based on Casa Sapo Data)

#### 4.1.2a Initial Clusters


# df de relação entre subsecção e cluster
u = dataframes["cluster_data_v1"]
u = u.astype({"BGRI11_M1LABELS_LP_AdjMatGeoNetDist": "Int64"})
# NaN values for areas that belong to São Jacinto area - no impact on the analysis
u[u["BGRI11_M1LABELS_LP_AdjMatGeoNetDist"].isna()]

u_updated = dataframes["cluster_data_v2"]

# load admnistrative limits for portuguese territory (subsection level)
PT_SUBSEC = gpd.read_file("Data/BGRI_2011/CONTINENTE/BGRI11_CONT.shp")

# convert to EPSG:3763
PT_SUBSEC = PT_SUBSEC.to_crs("EPSG:3763")

# filter for Aveiro LUG
filtro = ["014757"]
# apply filter
AVR = PT_SUBSEC[PT_SUBSEC["LUG11"].isin(filtro)]

# filter out residuals
AVR = AVR[AVR["LUG11"] != "999999"]

CLUSTER_FR = AVR.dissolve(by="FR11")
CLUSTER_FR.reset_index(inplace=True)
CLUSTER_LUG = AVR.dissolve(by="LUG11")
CLUSTER_LUG.reset_index(inplace=True)

# convert for merging
AVR["BGRI11"] = AVR["BGRI11"].astype("Int64")
# rename columns - this clusters are related to the Label Propagation algorithm applied by PB
u.rename(columns={"BGRI11_M1LABELS_LP_AdjMatGeoNetDist": "Cluster_LP"}, inplace=True)

u_updated.drop(["area"], axis=1, inplace=True)

# create a new dataframe with the cluster info for Aveiro and Ílhavo subsections
CLUSTER_LP = AVR.merge(u_updated, on="BGRI11", how="left")
# drop unnecessary columns
CLUSTER_LP.drop(
    ["OBJECTID", "DTMN11", "SEC11", "FR11", "LUG11", "SS11", "BGRI11", "LUG11DESIG"],
    axis=1,
    inplace=True,
)

##### 4.1.2a.1 Correction of the Clusters

# This section is important to generate the variable u2 (that is used as u_updated in the section above).

# This was necessary to be performed because some clusters from the Label propagation process (not done by me) are multypolygons. These multypolygons are "exploded" and labeled as new polygons (its components), generating a new u2 variable with better results than the variable u that was supplied by my thesis coordinator.

# commented because this was used to generate the u2 file, used above as u_updated

# CLUSTER_LP = CLUSTER_LP.dissolve(by='Cluster_LP').explode()
# CLUSTER_LP.reset_index(inplace=True)

# CLUSTER_LP0=CLUSTER_LP[CLUSTER_LP['level_1']==0]
# CLUSTER_LP1=CLUSTER_LP[CLUSTER_LP['level_1']==1]
# CLUSTER_LP2=CLUSTER_LP[CLUSTER_LP['level_1']==2]
# CLUSTER_LP3=CLUSTER_LP[CLUSTER_LP['level_1']==3]
# CLUSTER_LP0.reset_index(inplace=True)
# CLUSTER_LP1.reset_index(inplace=True)
# CLUSTER_LP2.reset_index(inplace=True)
# CLUSTER_LP3.reset_index(inplace=True)
# commented because this was used to generate the u2 file, used above as u_updated
# CLUSTER_LP0.Cluster_LP.unique()
# commented because this was used to generate the u2 file, used above as u_updated
# CLUSTER_LP0['Cluster_LP'].unique()
# commented because this was used to generate the u2 file, used above as u_updated
# CLUSTER_LP1['Cluster_LP'].unique()
# commented because this was used to generate the u2 file, used above as u_updated
# CLUSTER_LP2['Cluster_LP'].unique()
# commented because this was used to generate the u2 file, used above as u_updated
# CLUSTER_LP3['Cluster_LP'].unique()
# commented because this was used to generate the u2 file, used above as u_updated

# CLUSTER_LP3['Cluster_LP'] = np.where(CLUSTER_LP3['Cluster_LP']==64, 150, CLUSTER_LP3['Cluster_LP'])
# CLUSTER_LP2['Cluster_LP'] = np.where(CLUSTER_LP2['Cluster_LP']==30, 151, CLUSTER_LP2['Cluster_LP'])
# CLUSTER_LP2['Cluster_LP'] = np.where(CLUSTER_LP2['Cluster_LP']==60, 152, CLUSTER_LP2['Cluster_LP'])
# CLUSTER_LP2['Cluster_LP'] = np.where(CLUSTER_LP2['Cluster_LP']==63, 153, CLUSTER_LP2['Cluster_LP'])
# CLUSTER_LP2['Cluster_LP'] = np.where(CLUSTER_LP2['Cluster_LP']==64, 154, CLUSTER_LP2['Cluster_LP'])
# CLUSTER_LP2['Cluster_LP'] = np.where(CLUSTER_LP2['Cluster_LP']==107, 155, CLUSTER_LP2['Cluster_LP'])
# CLUSTER_LP1['Cluster_LP'] = np.where(CLUSTER_LP1['Cluster_LP']==22, 156, CLUSTER_LP1['Cluster_LP'])
# CLUSTER_LP1['Cluster_LP'] = np.where(CLUSTER_LP1['Cluster_LP']==26, 157, CLUSTER_LP1['Cluster_LP'])
# CLUSTER_LP1['Cluster_LP'] = np.where(CLUSTER_LP1['Cluster_LP']==30, 158, CLUSTER_LP1['Cluster_LP'])
# CLUSTER_LP1['Cluster_LP'] = np.where(CLUSTER_LP1['Cluster_LP']==40, 159, CLUSTER_LP1['Cluster_LP'])
# CLUSTER_LP1['Cluster_LP'] = np.where(CLUSTER_LP1['Cluster_LP']==41, 160, CLUSTER_LP1['Cluster_LP'])
# CLUSTER_LP1['Cluster_LP'] = np.where(CLUSTER_LP1['Cluster_LP']==43, 161, CLUSTER_LP1['Cluster_LP'])
# CLUSTER_LP1['Cluster_LP'] = np.where(CLUSTER_LP1['Cluster_LP']==60, 162, CLUSTER_LP1['Cluster_LP'])
# CLUSTER_LP1['Cluster_LP'] = np.where(CLUSTER_LP1['Cluster_LP']==63, 163, CLUSTER_LP1['Cluster_LP'])
# CLUSTER_LP1['Cluster_LP'] = np.where(CLUSTER_LP1['Cluster_LP']==64, 164, CLUSTER_LP1['Cluster_LP'])
# CLUSTER_LP1['Cluster_LP'] = np.where(CLUSTER_LP1['Cluster_LP']==103, 165, CLUSTER_LP1['Cluster_LP'])
# CLUSTER_LP1['Cluster_LP'] = np.where(CLUSTER_LP1['Cluster_LP']==107, 166, CLUSTER_LP1['Cluster_LP'])

# CLUSTER_LP=pd.concat([CLUSTER_LP0,CLUSTER_LP1,CLUSTER_LP2,CLUSTER_LP3],axis=0)

# drop unnecessary columns
CLUSTER_LUG.drop(
    ["OBJECTID", "DTMN11", "SEC11", "FR11", "SS11", "BGRI11", "LUG11DESIG"],
    axis=1,
    inplace=True,
)
CLUSTER_FR.drop(
    ["OBJECTID", "DTMN11", "SEC11", "LUG11", "SS11", "BGRI11", "LUG11DESIG"],
    axis=1,
    inplace=True,
)

# Plot Clusters LP, LUG and FR
CLUSTER_LP = CLUSTER_LP.dissolve(by="Cluster_LP").reset_index()
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle("Clustering by Label Propagation, LUG11 and FR11")
CLUSTER_LP.plot(ax=ax1)
CLUSTER_LUG.plot(ax=ax2)
CLUSTER_FR.plot(ax=ax3)

# print the result from the clustering with Label Propagation
ax = CLUSTER_LP.plot(
    figsize=(10, 10),
    column="Cluster_LP",
    edgecolor="b",
    legend=False,
    linewidth=0.2,
    cmap="tab20",
)
cx.add_basemap(ax, crs=CLUSTER_LP.crs, source=cx.providers.OpenStreetMap.Mapnik)
plt.title("Representação das 74 zonas base geradas")
plt.show()

# AVR_C=AVR.copy()
# create a centroid for the geometries
# AVR_C['centroid'] = AVR_C.centroid
# rename columns so centroid is the geometry for the geodataframe AVR_C (necessary for the spatial join)
# AVR_C.rename(columns={"geometry":"geometry2","centroid": "geometry"}, inplace=True)
# AVR_C.crs==CLUSTER_LP.crs
# commented because this was used to generate the u2 file, used above as u_updated
# u2=gpd.sjoin(AVR_C[['BGRI11','geometry']],CLUSTER_LP, how='left', predicate='intersects')
# commented because this was used to generate the u2 file, used above as u_updated
# u2['Cluster_LP'].unique()
# commented because this was used to generate the u2 file, used above as u_updated
# u2['Cluster_LP'].nunique()
# commented because this was used to generate the u2 file, used above as u_updated
# u2.drop(columns=['geometry','index_right'], inplace=True)
# commented because this was used to generate the u2 file, used above as u_updated
# u2.reset_index(drop=True, inplace=True)
# commented because this was used to generate the u2 file, used above as u_updated
# u2
# commented because this was used to generate the u2 file, used above as u_updated
# u2.to_csv('../Data/BGRI11_AVRILH_M1LABELS_LP_AdjMatGeoNetDist_v2.csv', index=False)
#### 4.1.2b Merge Cluster Data with CAOP2011

AVR_C = AVR.copy()
# create a centroid for the geometries
AVR_C["centroid"] = AVR_C.centroid
# rename columns so centroid is the geometry for the geodataframe AVR_C (necessary for the spatial join)
AVR_C.rename(columns={"geometry": "geometry2", "centroid": "geometry"}, inplace=True)
AVR_C.head()
AVR_C.shape
AVR_C.crs
CLUSTER_LP.shape
AVR_C.shape
CLUSTER_LP["Cluster_LP"].unique()
CLUSTER_LP.plot()
AVR.plot()

# bring Cluster Coding to the temp dataframe (with DICOFRESUBSEC info)
AVR = gpd.sjoin(
    AVR_C, CLUSTER_LP[["Cluster_LP", "geometry"]], how="left", predicate="intersects"
)
# drop unnecessary columns and rename back geometry 2 to geometry
AVR.drop(["geometry", "index_right"], axis=1, inplace=True)
AVR.rename(columns={"geometry2": "geometry"}, inplace=True)
AVR.head()
AVR.shape
### 4.1.3 PY Data

PY_data.shape
PY_data2 = PY_data.copy()
PY_data2 = PY_data2[["ID", "CONSTRUCTION_YEAR", "ADD_LONGITUDE", "ADD_LATITUDE"]]
# create a geodataframe for the PY data
gdf_PY = gpd.GeoDataFrame(
    PY_data, geometry=gpd.points_from_xy(PY_data.ADD_LONGITUDE, PY_data.ADD_LATITUDE)
)
# drop unnecessary columns
gdf_PY = gdf_PY.drop(["ADD_LATITUDE", "ADD_LONGITUDE"], axis=1)
# set crs to EPSG:4326
gdf_PY = gdf_PY.set_crs("epsg:4326")
# convert to EPSG:3763
gdf_PY = gdf_PY.to_crs("epsg:3763")
gdf_PY.plot()
gdf_PY.shape
AVR.crs == gdf_PY.crs
gdf_PY.head()
gdf_PY["CREATION_DATE"]
# steps to extract the year from the date
a = gdf_PY["CREATION_DATE"].str.split("-", n=1, expand=True)
# steps to extract the year from the date
a.rename(columns={0: "year", 1: "other"}, inplace=True)
a["year"].value_counts()
# transform original date to year
gdf_PY["Year"] = a["year"]
# drop unnecessary column
gdf_PY.drop(["CREATION_DATE"], axis=1, inplace=True)
gdf_PY = gdf_PY[~gdf_PY["AREA"].isnull()]  # filtrar elementos sem informação de área
gdf_PY = gdf_PY[gdf_PY["AREA"] > 0]  # filtrar elementos com área igual a 0
gdf_PY = gdf_PY[
    gdf_PY["AREA"] < 10000
]  # filtrar elementos com área igual a 0 - isto é feito mais à frente, mas trouxe para cá na elaboração do guia metodológico
gdf_PY.shape
gdf_PY["CONSTRUCTION_YEAR"] = gdf_PY["CONSTRUCTION_YEAR"].astype(
    "Int64"
)  # converter variável ano de construção para inteiro
gdf_PY = gdf_PY[~gdf_PY["CONSTRUCTION_YEAR"].isnull()]  # remover nan
gdf_PY = gdf_PY[
    gdf_PY["CONSTRUCTION_YEAR"] > 1600
]  # filtrar elementos com ano de construção errado (inferior a 1600)
gdf_PY = gdf_PY[
    gdf_PY["CONSTRUCTION_YEAR"] <= (datetime.date.today().year)
]  # filtrar elementos com ano de construção errado (superior ao ano atual)
gdf_PY.shape
# Preparação de dados para posterior lin reg
gdf_PY["Price_Area"] = (
    gdf_PY["CURRENT_PRICE"] / gdf_PY["AREA"]
)  # criar variável por preço por m2
gdf_PY["Log_Price_Area"] = np.log(
    gdf_PY["Price_Area"]
)  # criar variável por preço por m2
# drop unnecessary column
gdf_PY.drop(["Price_Area"], axis=1, inplace=True)
gdf_PY.shape
# calculate Preservação for PY data
today = datetime.datetime.now()

gdf_PY["Preservacao"] = (
    today.year - gdf_PY["CONSTRUCTION_YEAR"]
)  # criar variável por idade do imóvel
gdf_PY["Preservacao"].describe()
# match preservação to the categories defined for casasapo
for i, row in gdf_PY.iterrows():
    if (
        row["CONDITION_ID"] in [3, 7] or row["Preservacao"] < 0
    ):  # Under Construction, Under Project
        gdf_PY.at[i, "Preservacao"] = 5
    elif row["Preservacao"] >= 0 and row["Preservacao"] < 2:
        gdf_PY.at[i, "Preservacao"] = 1
    elif row["Preservacao"] >= 2 and row["Preservacao"] <= 10:
        gdf_PY.at[i, "Preservacao"] = 2
    elif row["Preservacao"] > 10 and row["Preservacao"] <= 25:
        gdf_PY.at[i, "Preservacao"] = 3
    elif row["Preservacao"] > 25:
        gdf_PY.at[i, "Preservacao"] = 4
gdf_PY["Preservacao"].value_counts()
gdf_PY.shape
# convert to integer
gdf_PY["ID"] = gdf_PY["ID"].astype("Int64")
gdf_PY["CONDITION_ID"] = gdf_PY["CONDITION_ID"].astype("Int64")
gdf_PY["TYPE_ID"] = gdf_PY["TYPE_ID"].astype("Int64")
gdf_PY["TYPOLOGY_ID"] = gdf_PY["TYPOLOGY_ID"].astype("Int64")
# Análise da Variável BUSINESS_TYPE
gdf_PY["CONSTRUCTION_YEAR"].unique()
# drop unnecessary columns
gdf_PY.drop(["CONSTRUCTION_YEAR"], axis=1, inplace=True)
gdf_PY.shape
gdf_PY.nsmallest(
    5, ["CURRENT_PRICE"]
)  # há um valor que não faz sentido (400€) - será filtrado
gdf_PY = gdf_PY[
    gdf_PY["CURRENT_PRICE"] > 1000
]  # filtrar elementos com preço inferior a 1000€
gdf_PY.shape
gdf_PY.nsmallest(5, ["CURRENT_PRICE"])  # OK
gdf_PY.nsmallest(5, ["AREA"])
gdf_PY = gdf_PY[gdf_PY["AREA"] > 20]  # filtrar elementos com area inferior a 20m2
gdf_PY.shape
gdf_PY.nlargest(5, ["AREA"])
gdf_PY.shape
# drop unnecessary column
gdf_PY.drop(["CONDITION_ID"], axis=1, inplace=True)
gdf_PY["TYPE_ID"].unique()  # OK, será variável dummy
gdf_PY["TYPOLOGY_ID"].unique()  # OK
gdf_PY.shape
# spatial join between the geodataframe with the PY data and the geodataframe with the cluster info
PY = gpd.sjoin(
    gdf_PY, CLUSTER_LP[["Cluster_LP", "geometry"]], how="left", predicate="intersects"
)
PY.shape
PY["Cluster_LP"] = PY["Cluster_LP"].astype("Int64")
PY["Cluster_LP"].value_counts()
# histogram  for Log_Price_Area
sns.displot(PY["Log_Price_Area"], bins=30, kde=True)


ax = PY.plot()
cx.add_basemap(ax, crs=PY.crs, source=cx.providers.OpenStreetMap.Mapnik)
# filter out elements outside the study area
PY = PY[~PY["Cluster_LP"].isnull()]
# filtrar elementos fora da região em estudo
PY.shape
PY.head()
PY.shape
ax = PY.plot()
cx.add_basemap(ax, crs=PY.crs, source=cx.providers.OpenStreetMap.Mapnik)
### 4.1.4 Aggregation of Socioeconomic Indicators to the Datasets


# load BGRI 2011 indicators (dataframe)
PT_BGRI = pd.read_table(
    "Data/BGRI_2011/BGRI2011_PT_corrigido.csv", sep=",", encoding="latin1"
)
PT_BGRI.shape
# subset of BGRI11, with only subsec entries
PT_BGRI = PT_BGRI.loc[PT_BGRI["NIVEL"] == 8]
PT_BGRI.shape
PT_BGRI["GEO_COD"]
list(PT_BGRI.columns)
AVR.shape
# convert Cluster variable to integer
AVR["Cluster_LP"] = AVR["Cluster_LP"].astype("Int64")
AVR["Cluster_LP"].unique()
AVR.head()
AVR.reset_index(drop=True, inplace=True)
PT_BGRI.reset_index(drop=True, inplace=True)
PT_BGRI.shape
# drop unnecessary columns
PT_BGRI.drop(["Unnamed: 0", "GEO_COD_DSG", "NIVEL", "NIVEL_DSG"], axis=1, inplace=True)
PT_BGRI.shape
PT_BGRI.head()
# convert variables to integer prior to merge
AVR["BGRI11"] = AVR["BGRI11"].astype(int)
PT_BGRI["GEO_COD"] = PT_BGRI["GEO_COD"].astype(int)
# include the BGRI socioeconomic indicators in the geodataframe
BGRI_CLUSTER = PT_BGRI.merge(AVR, left_on="GEO_COD", right_on="BGRI11")
BGRI_CLUSTER.shape
BGRI_CLUSTER.head()
# drop unnecessary columns
BGRI_CLUSTER.drop(["OBJECTID", "DTMN11", "SEC11", "SS11"], axis=1, inplace=True)
BGRI_CLUSTER
BGRI_CLUSTER.shape
BGRI_CLUSTER.reset_index(drop=True, inplace=True)
# rename columns
BGRI_CLUSTER.rename(
    columns={
        "N_EDIFICIOS_CLASSICOS_ISOLADOS ": "N_EDIFICIOS_CLASSICOS_ISOLADOS",
        ".N_EDIFICIOS_5OU_MAIS_PISOS": "N_EDIFICIOS_5OU_MAIS_PISOS",
        "N_IND_RESID_TRAB_MUN_RESID,": "N_IND_RESID_TRAB_MUN_RESID",
    },
    inplace=True,
    errors="raise",
)
# fix variable (remove last character)
BGRI_CLUSTER["N_IND_RESID_TRAB_MUN_RESID"] = BGRI_CLUSTER[
    "N_IND_RESID_TRAB_MUN_RESID"
].str[:-1]
# convert it to integer
BGRI_CLUSTER["N_IND_RESID_TRAB_MUN_RESID"] = BGRI_CLUSTER[
    "N_IND_RESID_TRAB_MUN_RESID"
].astype("Int64")
# this dataframe BGRI_CLUSTER has the BGRI11 and the cluster coding, so now we can bring the inddicators to the datasets with housing prices and transactions
BGRI_CLUSTER.head()
len(BGRI_CLUSTER["GEO_COD"].unique())
BGRI_CLUSTER.shape
### 4.1.5 Aggregation of Tourism Indicators to the Datasets

#### 4.1.5a Explore AL Data and prepare for merging

al = dataframes["al_establishments"]
# filter Alojamento Local data for Municipalities of Aveiro and Ílhavo
aveiro = al[al["Concelho"] == "Aveiro"]
ilhavo = al[al["Concelho"] == "Ílhavo"]

# create a new df with both municipalities
al_avrilh = pd.concat([aveiro, ilhavo], axis=0)
al_avrilh.shape
al_avrilh.head()
# get year from date
al_avrilh["DataAberturaPublico"] = al_avrilh["DataAberturaPublico"].str[:4]
al_avrilh.reset_index(drop=True, inplace=True)
# filter out unnecessary columns
al_avrilh = al_avrilh[["X", "Y", "DataAberturaPublico", "NrUtentes"]]
al_avrilh.head()
# transform to geodataframe
gdf_al = gpd.GeoDataFrame(
    al_avrilh, geometry=gpd.points_from_xy(al_avrilh.X, al_avrilh.Y)
)
# set crs
gdf_al = gdf_al.set_crs("epsg:4326")
# convert to portuguese crs
gdf_al = gdf_al.to_crs("epsg:3763")
# drop unnecessary columns
gdf_al.drop(["X", "Y"], axis=1, inplace=True)
gdf_al.plot()
#### 4.1.5b Merge Info into the Datasets

gdf_al.crs == CLUSTER_LP.crs
# intersect the geodataframes to bring Cluster info to the AL dataset
gdf_al = gpd.sjoin(
    gdf_al, CLUSTER_LP[["Cluster_LP", "geometry"]], how="left", predicate="intersects"
)
gdf_al.shape
# filter out AL that are not inside a cluster
gdf_al = gdf_al[gdf_al["Cluster_LP"].notnull()]
gdf_al.shape
BGRI_CLUSTER["N_INDIVIDUOS_RESIDENT"].describe()
# get a sum of the number of residents in 2011
a = BGRI_CLUSTER["N_INDIVIDUOS_RESIDENT"].sum()
a
fiona.listlayers("Data/BGRI21_CONT/BGRI21_CONT.gpkg")
# load BGRI 2021 data
BGRI_2021 = gpd.read_file("Data/BGRI21_CONT/BGRI21_CONT.gpkg", layer="BGRI21_CONT")
BGRI_2021.columns
BGRI_2021.head()
BGRI_2021_AVRILH = BGRI_2021.clip(AVR)
BGRI_2021_AVRILH.plot()
BGRI_2021_AVRILH["N_INDIVIDUOS"].describe()
# get a sum of the number of residents in 2021
a = BGRI_2021_AVRILH["N_INDIVIDUOS"].sum()
a
BGRI_2021_AVRILH_C = BGRI_2021_AVRILH.copy()
# calculate centroids
BGRI_2021_AVRILH_C["centroid"] = BGRI_2021_AVRILH_C.centroid
# use centroids as geometry
BGRI_2021_AVRILH_C.rename(
    columns={"geometry": "geometry2", "centroid": "geometry"}, inplace=True
)
# intersect the geodataframes to bring Cluster info to the BGRI 2021 dataset
AVRILH_CLUSTERS_2 = gpd.sjoin(
    BGRI_2021_AVRILH_C,
    CLUSTER_LP[["Cluster_LP", "geometry"]],
    how="left",
    predicate="intersects",
)
AVRILH_CLUSTERS_2.head()
AVRILH_CLUSTERS_2.shape
# drop unnecessary columns
AVRILH_CLUSTERS_2.drop(["geometry", "index_right"], axis=1, inplace=True)
# make geometry2, geometry again
AVRILH_CLUSTERS_2.rename(columns={"geometry2": "geometry"}, inplace=True)
AVRILH_CLUSTERS_2.shape
# remove data points not inside a cluster
AVRILH_CLUSTERS_2 = AVRILH_CLUSTERS_2[AVRILH_CLUSTERS_2["Cluster_LP"].notnull()]
AVRILH_CLUSTERS_2.shape
AVRILH_CLUSTERS_2.head()
len(AVRILH_CLUSTERS_2["BGRI2021"].unique())
# convert to int the date
gdf_al["DataAberturaPublico"] = gdf_al["DataAberturaPublico"].astype(int)
# create a new df with the AL existing before 2011
gdf_al_cs = gdf_al[gdf_al["DataAberturaPublico"] <= 2010]
gdf_al_cs["DataAberturaPublico"].unique()
BGRI_CLUSTER.head()
# check residents per cluster in 2021
df1 = BGRI_CLUSTER.groupby(["Cluster_LP"])["N_INDIVIDUOS_RESIDENT"].sum().reset_index()
df1
# check Utentes AL per cluster in 2011
df2 = gdf_al_cs.groupby(["Cluster_LP"])["NrUtentes"].sum().reset_index()
df3 = gdf_al_cs.groupby(["Cluster_LP"])["DataAberturaPublico"].count().reset_index()
df3
df2
df1.dtypes
df2["Cluster_LP"] = df2["Cluster_LP"].astype("int64")
df2 = df2.merge(df1, on="Cluster_LP", how="left")
# calculate AL_pc, Alojamento Local Per Capita, as the number of utentes AL divided by residents, per cluster
df2["AL_pc"] = df2["NrUtentes"] / df2["N_INDIVIDUOS_RESIDENT"]
df2
df3 = df3.merge(df2, on="Cluster_LP", how="left")
df3
# merge
gdf_al_cs = gdf_al_cs.merge(df3, on="Cluster_LP", how="left")
gdf_al_cs.drop(
    columns=[
        "DataAberturaPublico_x",
        "NrUtentes_x",
        "geometry",
        "index_right",
        "NrUtentes_y",
        "N_INDIVIDUOS_RESIDENT",
    ],
    inplace=True,
)
gdf_al_cs.rename(columns={"DataAberturaPublico_y": "Tot_AL"}, inplace=True)
gdf_al_cs
# repeat process, but now for AL existing after 2011
dfa = AVRILH_CLUSTERS_2.groupby(["Cluster_LP"])["N_INDIVIDUOS"].sum().reset_index()
# repeat process, but now for AL existing after 2011
dfb = gdf_al.groupby(["Cluster_LP"])["NrUtentes"].sum().reset_index()
# repeat process, but now for AL existing after 2011
dfc = gdf_al.groupby(["Cluster_LP"])["DataAberturaPublico"].count().reset_index()
# repeat process, but now for AL existing after 2011
dfb = dfb.merge(dfa, on="Cluster_LP", how="left")
# repeat process, but now for AL existing after 2011
dfb["AL_pc"] = dfb["NrUtentes"] / dfb["N_INDIVIDUOS"]
dfb.drop(columns=["NrUtentes", "N_INDIVIDUOS"], inplace=True)
# repeat process, but now for AL existing after 2011
dfb = dfb.merge(dfc, on="Cluster_LP", how="left")
dfb.dtypes
dfb.rename(columns={"DataAberturaPublico": "Tot_AL"}, inplace=True)
dfb.head()
# bring AL_pc to the casasapo dataset
casasapo = casasapo.merge(gdf_al_cs, on="Cluster_LP", how="left")
# fill na with 0
casasapo["AL_pc"] = casasapo["AL_pc"].fillna(0)
casasapo["Tot_AL"] = casasapo["Tot_AL"].fillna(0)
dfb["Cluster_LP"] = dfb["Cluster_LP"].astype("int64")
# bring AL_pc to the PY dataset
PY = PY.merge(dfb, on="Cluster_LP", how="left")
# fill na with 0
PY["AL_pc"] = PY["AL_pc"].fillna(0)
PY["Tot_AL"] = PY["Tot_AL"].fillna(0)
casasapo = casasapo.drop_duplicates()
PY = PY.drop_duplicates()
casasapo.reset_index(drop=True, inplace=True)
PY.reset_index(drop=True, inplace=True)
### 4.1.6 Export Data to Pickle Files

#### 4.1.6a Export Casasapo Data to a Pickle File


casasapo.to_pickle("Data/piclo_casasapo.piclo")

#### 4.1.6b Export PY Data to a Pickle File

# save PY dataset to a pickle
PY.to_pickle("Data/piclo_py.piclo")
#### 4.1.6c Export All Data (Casasapo + PY) Data to a Pickle File

# rename columns
casasapo.rename(
    columns={
        "Tipologia": "Typology",
        "Natureza": "Nature",
        "Preservacao": "Status",
        "VA1010_PrecoInicial": "Price",
        "VA1012_LOG_PrecoInicial_M2Area": "Log_P_A",
        "VA0110_Area": "A",
        "DTA_ano": "Year",
    },
    inplace=True,
)

# rename columns
PY.rename(
    columns={
        "AREA": "A",
        "Preservacao": "Status",
        "TYPE_ID": "Nature",
        "TYPOLOGY_ID": "Typology",
        "Log_Price_Area": "Log_P_A",
        "Log_VA0110_Area": "Log_A",
        "ZONECorr": "Cluster_50",
        "CURRENT_PRICE": "Price",
        "Log_Price": "Log_P",
        "Log_Area": "Log_A",
    },
    inplace=True,
)
# define T=1 for py and T=0 for cs, post intervention and pre intervention
PY["T"] = 1
casasapo["T"] = 0
PY.reset_index(drop=True, inplace=True)
# drop unnecessary columns
PY.drop(columns=["geometry", "index_right"], axis=1, inplace=True)
# drop unnecessary columns
casasapo.drop(columns=["Zone_ID", "Cluster_old"], axis=1, inplace=True)

# concat data
all_data = pd.concat([casasapo, PY], axis=0, ignore_index=True)
all_data["Year"].unique()

all_data = all_data[all_data["Status"] != 5]  # remove elements under construction


all_data.loc[all_data["T"] == 0, "SOURCE_ID"] = 4.0
all_data2 = all_data.merge(PY_data2, left_on="ID", right_on="ID", how="left")

all_data2.rename(columns={"geometry": "Coordinates"}, inplace=True)
all_data2.rename(
    columns={
        "ADD_LONGITUDE": "Longitude",
        "ADD_LATITUDE": "Latitude",
        "CONSTRUCTION_YEAR": "Construc_Year",
    },
    inplace=True,
)
all_data2.fillna(0, inplace=True)

all_data2["Construc_Year"] = all_data2["Construc_Year"].astype("Int64")
all_data2.drop(columns=["Log_P_A"], axis=1, inplace=True)

# export CSV with all data
all_data2.to_csv("Data/all_data.csv", index=False)
# save df with bluster info to a pickle
all_data.to_pickle("Data/all_data.piclo")
#### 4.1.6d Export BGRI_Cluster data to a Pickle File


BGRI_CLUSTER.Cluster_LP.unique()
BGRI_CLUSTER.drop(
    columns=["area", "geometry", "GEO_COD", "BGRI11", "LUG11DESIG"], inplace=True
)

# save df with socioeconomic indicators to a pickle
BGRI_CLUSTER.to_pickle("Data/piclo_bgri.piclo")

# save df with bluster info to a pickle
CLUSTER_LP.to_pickle("Data/piclo_clusters_lp.piclo")
CLUSTER_LUG.to_pickle("Data/piclo_clusters_lug.piclo")
CLUSTER_FR.to_pickle("Data/piclo_clusters_fr.piclo")
#endregion

# Clustering (taking into account the pre-calculated Clusters)

## 1. Import Libraries

# Import Libraries


os.environ["USE_PYGEOS"] = "0"
## 2. Loading Data, Transforming Variables

# read data
bgri_cluster = pd.read_pickle("Data/piclo_bgri.piclo")
all_data = pd.read_pickle("Data/all_data.piclo")

# read data
clusters_fr = pd.read_pickle("Data/piclo_clusters_fr.piclo")
clusters_lug = pd.read_pickle("Data/piclo_clusters_lug.piclo")
clusters_lp = pd.read_pickle("Data/piclo_clusters_lp.piclo")

bgri_cluster[bgri_cluster["Cluster_LP"].isnull()]
bgri_cluster = bgri_cluster[~bgri_cluster["Cluster_LP"].isnull()]

# group by cluster and sum values, getting socioeconomic indicators for each cluster (from the subsection information)
bgri_cluster_LP = (
    bgri_cluster.groupby(["Cluster_LP"]).sum().reset_index(level="Cluster_LP")
)
bgri_cluster_LUG = bgri_cluster.groupby(["LUG11"]).sum().reset_index(level="LUG11")
bgri_cluster_FR = bgri_cluster.groupby(["FR11"]).sum().reset_index(level="FR11")

# rotinas utilizadas para verificar a presença de NaNs - linhas dropadas

# set de pontos (X) com NaNs depois do standardscaler()
# este output só aparece se o bloco "bgri_cluster.drop([24, 32, 47, 94, 101], axis=0, inplace=True)" não estiver implementado


# {10, 14, 49, 56, 59, 61, 62}


bgri_cluster_LP.isnull().values.any()
list(bgri_cluster_LP.columns)
## code below was used to check for NaNs in the rows identified with NaN values
# bgri_cluster_LP.loc[[10]].transpose()[0:60]
# bgri_cluster_LP.loc[[14]].transpose()[0:60]
# bgri_cluster_LP.loc[[49]].transpose()[0:60]
# bgri_cluster_LP.loc[[56]].transpose()[0:60]
# bgri_cluster_LP.loc[[61]].transpose()[0:60]
# bgri_cluster_LP.loc[[62]].transpose()[0:60]
## after analyzing all the columns, we decided to drop the following columns:
## - N_RES_HABITUAL_1_2_DIV
## - N_RES_HABITUAL_3_4_DIV
## - N_RES_HABITUAL_ESTAC_1
## - N_RES_HABITUAL_ESTAC_2
## - N_RES_HABITUAL_ESTAC_3
bgri_cluster_LP.drop(
    columns=[
        "N_RES_HABITUAL_1_2_DIV",
        "N_RES_HABITUAL_3_4_DIV",
        "N_RES_HABITUAL_ESTAC_1",
        "N_RES_HABITUAL_ESTAC_2",
        "N_RES_HABITUAL_ESTAC_3",
    ],
    inplace=True,
)
## drop row with ONLY NaNs - other 6 rows with NaNs were not dropped; columns with NaN values were dropped instead
# bgri_cluster_LP.drop([61], axis=0, inplace=True)
bgri_cluster_LP.head()
### 2.1 New variables, enhancing the information available in the dataset

# here we prove that N_EDIFICIOS_CLASSICOS_ISOLADOS+N_EDIFICIOS_CLASSICOS_GEMIN+N_EDIFICIOS_CLASSICOS_EMBANDA=N_EDIFICIOS_CLASSICOS_1OU2
bgri_cluster_LP["N_EDIFICIOS_CLASSICOS_1OU2"].sum() == (
    bgri_cluster_LP["N_EDIFICIOS_CLASSICOS_ISOLADOS"]
    + bgri_cluster_LP["N_EDIFICIOS_CLASSICOS_GEMIN"]
    + bgri_cluster_LP["N_EDIFICIOS_CLASSICOS_EMBANDA"]
).sum()
# here we prove that N_EDIFICIOS_CLASSICOS_ISOLADOS+N_EDIFICIOS_CLASSICOS_GEMIN+N_EDIFICIOS_CLASSICOS_EMBANDA=N_EDIFICIOS_CLASSICOS_1OU2
bgri_cluster_LUG["N_EDIFICIOS_CLASSICOS_1OU2"].sum() == (
    bgri_cluster_LUG["N_EDIFICIOS_CLASSICOS_ISOLADOS"]
    + bgri_cluster_LUG["N_EDIFICIOS_CLASSICOS_GEMIN"]
    + bgri_cluster_LUG["N_EDIFICIOS_CLASSICOS_EMBANDA"]
).sum()
# here we prove that N_EDIFICIOS_CLASSICOS_ISOLADOS+N_EDIFICIOS_CLASSICOS_GEMIN+N_EDIFICIOS_CLASSICOS_EMBANDA=N_EDIFICIOS_CLASSICOS_1OU2
bgri_cluster_FR["N_EDIFICIOS_CLASSICOS_1OU2"].sum() == (
    bgri_cluster_FR["N_EDIFICIOS_CLASSICOS_ISOLADOS"]
    + bgri_cluster_FR["N_EDIFICIOS_CLASSICOS_GEMIN"]
    + bgri_cluster_FR["N_EDIFICIOS_CLASSICOS_EMBANDA"]
).sum()
bgri_cluster_LP.head()
bgri_cluster_LP.shape
# sabemos que N_EDIFICIOS_CLASSICOS = N_EDIFICIOS_CLASSICOS_ISOLADOS + N_EDIFICIOS_CLASSICOS_GEMIN + N_EDIFICIOS_CLASSICOS_EMBANDA +
# + N_EDIFICIOS_CLASSICOS_3OUMAIS + N_EDIFICIOS_CLASSICOS_OUTROS

# drop possivel - N_EDIFICIOS_CLASSICOS_OUTROS (para não dar 100%)

bgri_cluster_LP["PER_EDIFICIOS_CLASSICOS_ISOLADOS"] = (
    bgri_cluster_LP["N_EDIFICIOS_CLASSICOS_ISOLADOS"]
    / bgri_cluster_LP["N_EDIFICIOS_CLASSICOS"]
)
bgri_cluster_LP["PER_EDIFICIOS_CLASSICOS_GEMIN"] = (
    bgri_cluster_LP["N_EDIFICIOS_CLASSICOS_GEMIN"]
    / bgri_cluster_LP["N_EDIFICIOS_CLASSICOS"]
)
bgri_cluster_LP["PER_EDIFICIOS_CLASSICOS_EMBANDA"] = (
    bgri_cluster_LP["N_EDIFICIOS_CLASSICOS_EMBANDA"]
    / bgri_cluster_LP["N_EDIFICIOS_CLASSICOS"]
)
bgri_cluster_LP["PER_EDIFICIOS_CLASSICOS_1OU2"] = (
    bgri_cluster_LP["N_EDIFICIOS_CLASSICOS_1OU2"]
    / bgri_cluster_LP["N_EDIFICIOS_CLASSICOS"]
)
bgri_cluster_LP["PER_EDIFICIOS_CLASSICOS_3OUMAIS"] = (
    bgri_cluster_LP["N_EDIFICIOS_CLASSICOS_3OUMAIS"]
    / bgri_cluster_LP["N_EDIFICIOS_CLASSICOS"]
)
bgri_cluster_LP["PER_EDIFICIOS_CLASSICOS_OUTROS"] = (
    bgri_cluster_LP["N_EDIFICIOS_CLASSICOS_OUTROS"]
    / bgri_cluster_LP["N_EDIFICIOS_CLASSICOS"]
)

bgri_cluster_LP.drop(
    [
        "N_EDIFICIOS_CLASSICOS",
        "N_EDIFICIOS_CLASSICOS_1OU2",
        "N_EDIFICIOS_CLASSICOS_ISOLADOS",
        "N_EDIFICIOS_CLASSICOS_GEMIN",
        "N_EDIFICIOS_CLASSICOS_EMBANDA",
        "N_EDIFICIOS_CLASSICOS_3OUMAIS",
        "N_EDIFICIOS_CLASSICOS_OUTROS",
    ],
    axis=1,
    inplace=True,
)
# sabemos que N_EDIFICIOS_EXCLUSIV_RESID + N_EDIFICIOS_PRINCIPAL_RESID + N_EDIFICIOS_PRINCIP_NAO_RESID =
# = N_EDIFICIOS_1OU2_PISOS + N_EDIFICIOS_3OU4_PISOS + N_EDIFICIOS_5OU_MAIS_PISOS

# drop possivel - PER_EDIFICIOS_5OU_MAIS_PISOS (para não dar 100%)

bgri_cluster_LP["total_temp"] = (
    bgri_cluster_LP["N_EDIFICIOS_EXCLUSIV_RESID"]
    + bgri_cluster_LP["N_EDIFICIOS_PRINCIPAL_RESID"]
    + bgri_cluster_LP["N_EDIFICIOS_PRINCIP_NAO_RESID"]
)

bgri_cluster_LP["PER_EDIFICIOS_EXCLUSIV_RESID"] = (
    bgri_cluster_LP["N_EDIFICIOS_EXCLUSIV_RESID"] / bgri_cluster_LP["total_temp"]
)
bgri_cluster_LP["PER_EDIFICIOS_PRINCIPAL_RESID"] = (
    bgri_cluster_LP["N_EDIFICIOS_PRINCIPAL_RESID"] / bgri_cluster_LP["total_temp"]
)
bgri_cluster_LP["PER_EDIFICIOS_PRINCIP_NAO_RESID"] = (
    bgri_cluster_LP["N_EDIFICIOS_PRINCIP_NAO_RESID"] / bgri_cluster_LP["total_temp"]
)

bgri_cluster_LP["PER_EDIFICIOS_1OU2_PISOS"] = (
    bgri_cluster_LP["N_EDIFICIOS_1OU2_PISOS"] / bgri_cluster_LP["total_temp"]
)
bgri_cluster_LP["PER_EDIFICIOS_3OU4_PISOS"] = (
    bgri_cluster_LP["N_EDIFICIOS_3OU4_PISOS"] / bgri_cluster_LP["total_temp"]
)
bgri_cluster_LP["PER_EDIFICIOS_5OU_MAIS_PISOS"] = (
    bgri_cluster_LP["N_EDIFICIOS_5OU_MAIS_PISOS"] / bgri_cluster_LP["total_temp"]
)

bgri_cluster_LP.drop(
    [
        "total_temp",
        "N_EDIFICIOS_EXCLUSIV_RESID",
        "N_EDIFICIOS_PRINCIPAL_RESID",
        "N_EDIFICIOS_PRINCIP_NAO_RESID",
        "N_EDIFICIOS_1OU2_PISOS",
        "N_EDIFICIOS_3OU4_PISOS",
        "N_EDIFICIOS_5OU_MAIS_PISOS",
    ],
    axis=1,
    inplace=True,
)
# conversão dos indicadores do ano de construção em percentagens do total de casas

# drop possivel - N_EDIFICIOS_CONSTR_2006A2011 (para não dar 100%)

bgri_cluster_LP["total_temp"] = (
    bgri_cluster_LP["N_EDIFICIOS_CONSTR_ANTES_1919"]
    + bgri_cluster_LP["N_EDIFICIOS_CONSTR_1919A1945"]
    + bgri_cluster_LP["N_EDIFICIOS_CONSTR_1946A1960"]
    + bgri_cluster_LP["N_EDIFICIOS_CONSTR_1961A1970"]
    + bgri_cluster_LP["N_EDIFICIOS_CONSTR_1971A1980"]
    + bgri_cluster_LP["N_EDIFICIOS_CONSTR_1981A1990"]
    + bgri_cluster_LP["N_EDIFICIOS_CONSTR_1991A1995"]
    + bgri_cluster_LP["N_EDIFICIOS_CONSTR_1996A2000"]
    + bgri_cluster_LP["N_EDIFICIOS_CONSTR_2001A2005"]
    + bgri_cluster_LP["N_EDIFICIOS_CONSTR_2006A2011"]
)

bgri_cluster_LP["PER_EDIFICIOS_CONSTR_ANTES_1919"] = (
    bgri_cluster_LP["N_EDIFICIOS_CONSTR_ANTES_1919"] / bgri_cluster_LP["total_temp"]
)
bgri_cluster_LP["PER_EDIFICIOS_CONSTR_1919A1945"] = (
    bgri_cluster_LP["N_EDIFICIOS_CONSTR_1919A1945"] / bgri_cluster_LP["total_temp"]
)
bgri_cluster_LP["PER_EDIFICIOS_CONSTR_1946A1960"] = (
    bgri_cluster_LP["N_EDIFICIOS_CONSTR_1946A1960"] / bgri_cluster_LP["total_temp"]
)
bgri_cluster_LP["PER_EDIFICIOS_CONSTR_1961A1970"] = (
    bgri_cluster_LP["N_EDIFICIOS_CONSTR_1961A1970"] / bgri_cluster_LP["total_temp"]
)
bgri_cluster_LP["PER_EDIFICIOS_CONSTR_1971A1980"] = (
    bgri_cluster_LP["N_EDIFICIOS_CONSTR_1971A1980"] / bgri_cluster_LP["total_temp"]
)
bgri_cluster_LP["PER_EDIFICIOS_CONSTR_1981A1990"] = (
    bgri_cluster_LP["N_EDIFICIOS_CONSTR_1981A1990"] / bgri_cluster_LP["total_temp"]
)
bgri_cluster_LP["PER_EDIFICIOS_CONSTR_1991A1995"] = (
    bgri_cluster_LP["N_EDIFICIOS_CONSTR_1991A1995"] / bgri_cluster_LP["total_temp"]
)
bgri_cluster_LP["PER_EDIFICIOS_CONSTR_1996A2000"] = (
    bgri_cluster_LP["N_EDIFICIOS_CONSTR_1996A2000"] / bgri_cluster_LP["total_temp"]
)
bgri_cluster_LP["PER_EDIFICIOS_CONSTR_2001A2005"] = (
    bgri_cluster_LP["N_EDIFICIOS_CONSTR_2001A2005"] / bgri_cluster_LP["total_temp"]
)
bgri_cluster_LP["PER_EDIFICIOS_CONSTR_2006A2011"] = (
    bgri_cluster_LP["N_EDIFICIOS_CONSTR_2006A2011"] / bgri_cluster_LP["total_temp"]
)

bgri_cluster_LP.drop(
    [
        "total_temp",
        "N_EDIFICIOS_CONSTR_ANTES_1919",
        "N_EDIFICIOS_CONSTR_1919A1945",
        "N_EDIFICIOS_CONSTR_1946A1960",
        "N_EDIFICIOS_CONSTR_1961A1970",
        "N_EDIFICIOS_CONSTR_1971A1980",
        "N_EDIFICIOS_CONSTR_1981A1990",
        "N_EDIFICIOS_CONSTR_1991A1995",
        "N_EDIFICIOS_CONSTR_1996A2000",
        "N_EDIFICIOS_CONSTR_2001A2005",
        "N_EDIFICIOS_CONSTR_2006A2011",
    ],
    axis=1,
    inplace=True,
)
# vamos agora droppar variáveis que decidimos não usar, por acreditarmos que não "informam" o modelo

bgri_cluster_LP.drop(
    [
        "N_EDIFICIOS_ESTRUT_BETAO",
        "N_EDIFICIOS_ESTRUT_COM_PLACA",
        "N_EDIFICIOS_ESTRUT_SEM_PLACA",
        "N_EDIFICIOS_ESTRUT_ADOBE_PEDRA",
        "N_EDIFICIOS_ESTRUT_OUTRA",
    ],
    axis=1,
    inplace=True,
)
# "Cluster" de Variáveis em análise: N_ALOJAMENTOS, N_ALOJAMENTOS_FAM_CLASSICOS, N_ALOJAMENTOS_FAM_N_CLASSICOS, N_ALOJAMENTOS_COLECTIVOS, N_CLASSICOS_RES_HABITUAL, N_ALOJAMENTOS_RES_HABITUAL, N_ALOJAMENTOS_VAGOS
# Após análise (à parte), definiu-se que toda a informação destas variáveis está contida nos seguintes percentuais:

bgri_cluster_LP["PER_ALOJAMENTOS_FAM_CLASSICOS"] = (
    bgri_cluster_LP["N_ALOJAMENTOS_FAM_CLASSICOS"] / bgri_cluster_LP["N_ALOJAMENTOS"]
)
bgri_cluster_LP["PER_ALOJAMENTOS_FAM_N_CLASSICOS"] = (
    bgri_cluster_LP["N_ALOJAMENTOS_FAM_N_CLASSICOS"] / bgri_cluster_LP["N_ALOJAMENTOS"]
)
bgri_cluster_LP["PER_ALOJAMENTOS_COLECTIVOS"] = (
    bgri_cluster_LP["N_ALOJAMENTOS_COLECTIVOS"] / bgri_cluster_LP["N_ALOJAMENTOS"]
)
bgri_cluster_LP["PER_CLASSICOS_RES_HABITUAL"] = (
    bgri_cluster_LP["N_CLASSICOS_RES_HABITUAL"] / bgri_cluster_LP["N_ALOJAMENTOS"]
)
bgri_cluster_LP["PER_ALOJAMENTOS_RES_HABITUAL"] = (
    bgri_cluster_LP["N_ALOJAMENTOS_RES_HABITUAL"] / bgri_cluster_LP["N_ALOJAMENTOS"]
)
bgri_cluster_LP["PER_ALOJAMENTOS_VAGOS"] = (
    bgri_cluster_LP["N_ALOJAMENTOS_VAGOS"] / bgri_cluster_LP["N_ALOJAMENTOS"]
)
bgri_cluster_LP["PER_ALOJAMENTOS_FAMILIARES"] = (
    bgri_cluster_LP["N_ALOJAMENTOS_FAMILIARES"] / bgri_cluster_LP["N_ALOJAMENTOS"]
)

bgri_cluster_LP.drop(
    [
        "N_ALOJAMENTOS",
        "N_ALOJAMENTOS_FAM_CLASSICOS",
        "N_ALOJAMENTOS_FAM_N_CLASSICOS",
        "N_ALOJAMENTOS_COLECTIVOS",
        "N_CLASSICOS_RES_HABITUAL",
        "N_ALOJAMENTOS_RES_HABITUAL",
        "N_ALOJAMENTOS_VAGOS",
        "N_ALOJAMENTOS_FAMILIARES",
    ],
    axis=1,
    inplace=True,
)
# vamos droppar as seguintes variáveis 'N_RES_HABITUAL_COM_AGUA','N_RES_HABITUAL_COM_RETRETE','N_RES_HABITUAL_COM_ESGOTOS','N_RES_HABITUAL_COM_BANHO'

bgri_cluster_LP.drop(
    [
        "N_RES_HABITUAL_COM_AGUA",
        "N_RES_HABITUAL_COM_RETRETE",
        "N_RES_HABITUAL_COM_ESGOTOS",
        "N_RES_HABITUAL_COM_BANHO",
    ],
    axis=1,
    inplace=True,
)
# Próximo Cluster de 'N_RES_HABITUAL_AREA_50', 'N_RES_HABITUAL_AREA_50_100', 'N_RES_HABITUAL_AREA_100_200', 'N_RES_HABITUAL_AREA_200'

bgri_cluster_LP["temp_total"] = (
    bgri_cluster_LP["N_RES_HABITUAL_AREA_50"]
    + bgri_cluster_LP["N_RES_HABITUAL_AREA_50_100"]
    + bgri_cluster_LP["N_RES_HABITUAL_AREA_100_200"]
    + bgri_cluster_LP["N_RES_HABITUAL_AREA_200"]
)

bgri_cluster_LP["PER_RES_HABITUAL_AREA_50"] = (
    bgri_cluster_LP["N_RES_HABITUAL_AREA_50"] / bgri_cluster_LP["temp_total"]
)
bgri_cluster_LP["PER_RES_HABITUAL_AREA_50_100"] = (
    bgri_cluster_LP["N_RES_HABITUAL_AREA_50_100"] / bgri_cluster_LP["temp_total"]
)
bgri_cluster_LP["PER_RES_HABITUAL_AREA_100_200"] = (
    bgri_cluster_LP["N_RES_HABITUAL_AREA_100_200"] / bgri_cluster_LP["temp_total"]
)
bgri_cluster_LP["PER_RES_HABITUAL_AREA_200"] = (
    bgri_cluster_LP["N_RES_HABITUAL_AREA_200"] / bgri_cluster_LP["temp_total"]
)

bgri_cluster_LP.drop(
    [
        "temp_total",
        "N_RES_HABITUAL_AREA_50",
        "N_RES_HABITUAL_AREA_50_100",
        "N_RES_HABITUAL_AREA_100_200",
        "N_RES_HABITUAL_AREA_200",
    ],
    axis=1,
    inplace=True,
)
# Próximo Cluster de 'N_RES_HABITUAL_PROP_OCUP','N_RES_HABITUAL_ARREND'
# Não fui capaz de encontrar relação entre variáveis (os totais não batem certo), pelo que será criado subtotatal e as variáveis serão percentagens desses subtotais

bgri_cluster_LP["temp_total3"] = (
    bgri_cluster_LP["N_RES_HABITUAL_PROP_OCUP"]
    + bgri_cluster_LP["N_RES_HABITUAL_ARREND"]
)

bgri_cluster_LP["PER_RES_HABITUAL_PROP_OCUP"] = (
    bgri_cluster_LP["N_RES_HABITUAL_PROP_OCUP"] / bgri_cluster_LP["temp_total3"]
)
bgri_cluster_LP["PER_RES_HABITUAL_ARREND"] = (
    bgri_cluster_LP["N_RES_HABITUAL_ARREND"] / bgri_cluster_LP["temp_total3"]
)


bgri_cluster_LP.drop(
    ["temp_total3", "N_RES_HABITUAL_PROP_OCUP", "N_RES_HABITUAL_ARREND"],
    axis=1,
    inplace=True,
)
# Próximo Cluster de 'N_FAMILIAS_CLASSICAS','N_FAMILIAS_INSTITUCIONAIS','N_FAMILIAS_CLASSICAS_1OU2_PESS','N_FAMILIAS_CLASSICAS_3OU4_PESS','N_FAMILIAS_CLASSICAS_NPES65',
#'N_FAMILIAS_CLASSICAS_NPES14','N_FAMILIAS_CLASSIC_SEM_DESEMP','N_FAMILIAS_CLASSIC_1DESEMPREG','N_FAMILIAS_CLASS_2MAIS_DESEMP'

# Não fui capaz de encontrar relação entre todas as variáveis (os totais não batem todos certo), pelo que serão as variáveis serão percentagens do N_FAMILIAS_CLASSICAS

bgri_cluster_LP["PER_FAMILIAS_INSTITUCIONAIS"] = (
    bgri_cluster_LP["N_FAMILIAS_INSTITUCIONAIS"]
    / bgri_cluster_LP["N_FAMILIAS_CLASSICAS"]
)
bgri_cluster_LP["PER_FAMILIAS_CLASSICAS_1OU2_PESS"] = (
    bgri_cluster_LP["N_FAMILIAS_CLASSICAS_1OU2_PESS"]
    / bgri_cluster_LP["N_FAMILIAS_CLASSICAS"]
)
bgri_cluster_LP["PER_FAMILIAS_CLASSICAS_3OU4_PESS"] = (
    bgri_cluster_LP["N_FAMILIAS_CLASSICAS_3OU4_PESS"]
    / bgri_cluster_LP["N_FAMILIAS_CLASSICAS"]
)
bgri_cluster_LP["PER_FAMILIAS_CLASSICAS_NPES65"] = (
    bgri_cluster_LP["N_FAMILIAS_CLASSICAS_NPES65"]
    / bgri_cluster_LP["N_FAMILIAS_CLASSICAS"]
)
bgri_cluster_LP["PER_FAMILIAS_CLASSICAS_NPES14"] = (
    bgri_cluster_LP["N_FAMILIAS_CLASSICAS_NPES14"]
    / bgri_cluster_LP["N_FAMILIAS_CLASSICAS"]
)
bgri_cluster_LP["PER_FAMILIAS_CLASSIC_SEM_DESEMP"] = (
    bgri_cluster_LP["N_FAMILIAS_CLASSIC_SEM_DESEMP"]
    / bgri_cluster_LP["N_FAMILIAS_CLASSICAS"]
)
bgri_cluster_LP["PER_FAMILIAS_CLASSIC_1DESEMPREG"] = (
    bgri_cluster_LP["N_FAMILIAS_CLASSIC_1DESEMPREG"]
    / bgri_cluster_LP["N_FAMILIAS_CLASSICAS"]
)
bgri_cluster_LP["PER_FAMILIAS_CLASS_2MAIS_DESEMP"] = (
    bgri_cluster_LP["N_FAMILIAS_CLASS_2MAIS_DESEMP"]
    / bgri_cluster_LP["N_FAMILIAS_CLASSICAS"]
)

bgri_cluster_LP.drop(
    [
        "N_FAMILIAS_CLASSICAS",
        "N_FAMILIAS_INSTITUCIONAIS",
        "N_FAMILIAS_CLASSICAS_1OU2_PESS",
        "N_FAMILIAS_CLASSICAS_3OU4_PESS",
        "N_FAMILIAS_CLASSICAS_NPES65",
        "N_FAMILIAS_CLASSICAS_NPES14",
        "N_FAMILIAS_CLASSIC_SEM_DESEMP",
        "N_FAMILIAS_CLASSIC_1DESEMPREG",
        "N_FAMILIAS_CLASS_2MAIS_DESEMP",
    ],
    axis=1,
    inplace=True,
)
# Próximo Cluster de 'N_NUCLEOS_FAMILIARES','N_NUCLEOS_1FILH_NAO_CASADO','N_NUCLEOS_2FILH_NAO_CASADO','N_NUCLEOS_FILH_INF_6ANOS','N_NUCLEOS_FILH_INF_15ANOS','N_NUCLEOS_FILH_MAIS_15ANOS'

# Não fui capaz de encontrar relação entre todas as variáveis (os totais não batem todos certo), pelo que serão as variáveis serão percentagens do N_NUCLEOS_FAMILIARES


bgri_cluster_LP["PER_NUCLEOS_1FILH_NAO_CASADO"] = (
    bgri_cluster_LP["N_NUCLEOS_1FILH_NAO_CASADO"]
    / bgri_cluster_LP["N_NUCLEOS_FAMILIARES"]
)
bgri_cluster_LP["PER_NUCLEOS_2FILH_NAO_CASADO"] = (
    bgri_cluster_LP["N_NUCLEOS_2FILH_NAO_CASADO"]
    / bgri_cluster_LP["N_NUCLEOS_FAMILIARES"]
)
bgri_cluster_LP["PER_NUCLEOS_FILH_INF_6ANOS"] = (
    bgri_cluster_LP["N_NUCLEOS_FILH_INF_6ANOS"]
    / bgri_cluster_LP["N_NUCLEOS_FAMILIARES"]
)
bgri_cluster_LP["PER_NUCLEOS_FILH_INF_15ANOS"] = (
    bgri_cluster_LP["N_NUCLEOS_FILH_INF_15ANOS"]
    / bgri_cluster_LP["N_NUCLEOS_FAMILIARES"]
)
bgri_cluster_LP["PER_NUCLEOS_FILH_MAIS_15ANOS"] = (
    bgri_cluster_LP["N_NUCLEOS_FILH_MAIS_15ANOS"]
    / bgri_cluster_LP["N_NUCLEOS_FAMILIARES"]
)

bgri_cluster_LP.drop(
    [
        "N_NUCLEOS_FAMILIARES",
        "N_NUCLEOS_1FILH_NAO_CASADO",
        "N_NUCLEOS_2FILH_NAO_CASADO",
        "N_NUCLEOS_FILH_INF_6ANOS",
        "N_NUCLEOS_FILH_INF_15ANOS",
        "N_NUCLEOS_FILH_MAIS_15ANOS",
    ],
    axis=1,
    inplace=True,
)
# Antes de passarmos ao próximo bloco de indicadores a tratar, vamos eliminar todas as variáveis relativas ao sexo dos residentes, mantendo aoenas informação relativa à idade
bgri_cluster_LP.drop(
    [
        "N_INDIVIDUOS_PRESENT_H",
        "N_INDIVIDUOS_PRESENT_M",
        "N_INDIVIDUOS_RESIDENT_H",
        "N_INDIVIDUOS_RESIDENT_M",
        "N_INDIVIDUOS_RESIDENT_H_0A4",
        "N_INDIVIDUOS_RESIDENT_H_5A9",
        "N_INDIVIDUOS_RESIDENT_H_10A13",
        "N_INDIVIDUOS_RESIDENT_H_14A19",
        "N_INDIVIDUOS_RESIDENT_H_15A19",
        "N_INDIVIDUOS_RESIDENT_H_20A24",
        "N_INDIVIDUOS_RESIDENT_H_20A64",
        "N_INDIVIDUOS_RESIDENT_H_25A64",
        "N_INDIVIDUOS_RESIDENT_H_65",
        "N_INDIVIDUOS_RESIDENT_M_0A4",
        "N_INDIVIDUOS_RESIDENT_M_5A9",
        "N_INDIVIDUOS_RESIDENT_M_10A13",
        "N_INDIVIDUOS_RESIDENT_M_14A19",
        "N_INDIVIDUOS_RESIDENT_M_15A19",
        "N_INDIVIDUOS_RESIDENT_M_20A24",
        "N_INDIVIDUOS_RESIDENT_M_20A64",
        "N_INDIVIDUOS_RESIDENT_M_25A64",
        "N_INDIVIDUOS_RESIDENT_M_65",
    ],
    axis=1,
    inplace=True,
)
# Próximo Cluster de 'N_INDIVIDUOS_PRESENT','N_INDIVIDUOS_RESIDENT','N_INDIVIDUOS_RESIDENT_0A4','N_INDIVIDUOS_RESIDENT_5A9','N_INDIVIDUOS_RESIDENT_10A13','N_INDIVIDUOS_RESIDENT_14A19',
#                       'N_INDIVIDUOS_RESIDENT_15A19','N_INDIVIDUOS_RESIDENT_20A24','N_INDIVIDUOS_RESIDENT_20A64','N_INDIVIDUOS_RESIDENT_25A64','N_INDIVIDUOS_RESIDENT_65'

# Todos os indicadores acima serão calculados em percentagem da variável N_INDIVIDUOS_RESIDENT
# NOTA: esta variável N_INDIVIDUOS_RESIDENT não será dropada no final visto que ainda será necessária para o próximo batch de indicadores a tratar

bgri_cluster_LP["PER_INDIVIDUOS_PRESENT"] = (
    bgri_cluster_LP["N_INDIVIDUOS_PRESENT"] / bgri_cluster_LP["N_INDIVIDUOS_RESIDENT"]
)
bgri_cluster_LP["PER_INDIVIDUOS_RESIDENT_0A4"] = (
    bgri_cluster_LP["N_INDIVIDUOS_RESIDENT_0A4"]
    / bgri_cluster_LP["N_INDIVIDUOS_RESIDENT"]
)
bgri_cluster_LP["PER_INDIVIDUOS_RESIDENT_5A9"] = (
    bgri_cluster_LP["N_INDIVIDUOS_RESIDENT_5A9"]
    / bgri_cluster_LP["N_INDIVIDUOS_RESIDENT"]
)
bgri_cluster_LP["PER_INDIVIDUOS_RESIDENT_10A13"] = (
    bgri_cluster_LP["N_INDIVIDUOS_RESIDENT_10A13"]
    / bgri_cluster_LP["N_INDIVIDUOS_RESIDENT"]
)
bgri_cluster_LP["PER_INDIVIDUOS_RESIDENT_14A19"] = (
    bgri_cluster_LP["N_INDIVIDUOS_RESIDENT_14A19"]
    / bgri_cluster_LP["N_INDIVIDUOS_RESIDENT"]
)
bgri_cluster_LP["PER_INDIVIDUOS_RESIDENT_20A24"] = (
    bgri_cluster_LP["N_INDIVIDUOS_RESIDENT_20A24"]
    / bgri_cluster_LP["N_INDIVIDUOS_RESIDENT"]
)
bgri_cluster_LP["PER_INDIVIDUOS_RESIDENT_25A64"] = (
    bgri_cluster_LP["N_INDIVIDUOS_RESIDENT_25A64"]
    / bgri_cluster_LP["N_INDIVIDUOS_RESIDENT"]
)
bgri_cluster_LP["PER_INDIVIDUOS_RESIDENT_65"] = (
    bgri_cluster_LP["N_INDIVIDUOS_RESIDENT_65"]
    / bgri_cluster_LP["N_INDIVIDUOS_RESIDENT"]
)

bgri_cluster_LP.drop(
    [
        "N_INDIVIDUOS_PRESENT",
        "N_INDIVIDUOS_RESIDENT_0A4",
        "N_INDIVIDUOS_RESIDENT_5A9",
        "N_INDIVIDUOS_RESIDENT_10A13",
        "N_INDIVIDUOS_RESIDENT_14A19",
        "N_INDIVIDUOS_RESIDENT_15A19",
        "N_INDIVIDUOS_RESIDENT_20A24",
        "N_INDIVIDUOS_RESIDENT_20A64",
        "N_INDIVIDUOS_RESIDENT_25A64",
        "N_INDIVIDUOS_RESIDENT_65",
    ],
    axis=1,
    inplace=True,
)
# Próximo Cluster de 'N_INDIVIDUOS_RESIDENT','N_INDIV_RESIDENT_N_LER_ESCRV','N_IND_RESIDENT_FENSINO_1BAS','N_IND_RESIDENT_FENSINO_2BAS','N_IND_RESIDENT_FENSINO_3BAS','N_IND_RESIDENT_FENSINO_SEC','N_IND_RESIDENT_FENSINO_POSSEC',
#'N_IND_RESIDENT_FENSINO_SUP','N_IND_RESIDENT_ENSINCOMP_1BAS','N_IND_RESIDENT_ENSINCOMP_2BAS','N_IND_RESIDENT_ENSINCOMP_3BAS','N_IND_RESIDENT_ENSINCOMP_SEC','N_IND_RESIDENT_ENSINCOMP_POSEC','N_IND_RESIDENT_ENSINCOMP_SUP',
#'N_IND_RESID_DESEMP_PROC_1EMPRG','N_IND_RESID_DESEMP_PROC_EMPRG','N_IND_RESID_EMPREGADOS','N_IND_RESID_PENS_REFORM','N_IND_RESID_SEM_ACT_ECON','N_IND_RESID_EMPREG_SECT_PRIM','N_IND_RESID_EMPREG_SECT_SEQ',
#'N_IND_RESID_EMPREG_SECT_TERC','N_IND_RESID_ESTUD_MUN_RESID','N_IND_RESID_TRAB_MUN_RESID'

# Todos os indicadores acima serão calculados em percentagem da variável N_INDIVIDUOS_RESIDENT (vamos ignorar indicadores relativos a emprego/desemprego porque são indicadores muito "conjunturais")

bgri_cluster_LP["PER_INDIV_RESIDENT_N_LER_ESCRV"] = (
    bgri_cluster_LP["N_INDIV_RESIDENT_N_LER_ESCRV"]
    / bgri_cluster_LP["N_INDIVIDUOS_RESIDENT"]
)
bgri_cluster_LP["PER_IND_RESIDENT_FENSINO_1BAS"] = (
    bgri_cluster_LP["N_IND_RESIDENT_FENSINO_1BAS"]
    / bgri_cluster_LP["N_INDIVIDUOS_RESIDENT"]
)
bgri_cluster_LP["PER_IND_RESIDENT_FENSINO_2BAS"] = (
    bgri_cluster_LP["N_IND_RESIDENT_FENSINO_2BAS"]
    / bgri_cluster_LP["N_INDIVIDUOS_RESIDENT"]
)
bgri_cluster_LP["PER_IND_RESIDENT_FENSINO_3BAS"] = (
    bgri_cluster_LP["N_IND_RESIDENT_FENSINO_3BAS"]
    / bgri_cluster_LP["N_INDIVIDUOS_RESIDENT"]
)
bgri_cluster_LP["PER_IND_RESIDENT_FENSINO_SEC"] = (
    bgri_cluster_LP["N_IND_RESIDENT_FENSINO_SEC"]
    / bgri_cluster_LP["N_INDIVIDUOS_RESIDENT"]
)
bgri_cluster_LP["PER_IND_RESIDENT_FENSINO_POSSEC"] = (
    bgri_cluster_LP["N_IND_RESIDENT_FENSINO_POSSEC"]
    / bgri_cluster_LP["N_INDIVIDUOS_RESIDENT"]
)
bgri_cluster_LP["PER_IND_RESIDENT_FENSINO_SUP"] = (
    bgri_cluster_LP["N_IND_RESIDENT_FENSINO_SUP"]
    / bgri_cluster_LP["N_INDIVIDUOS_RESIDENT"]
)
bgri_cluster_LP["PER_IND_RESIDENT_ENSINCOMP_1BAS"] = (
    bgri_cluster_LP["N_IND_RESIDENT_ENSINCOMP_1BAS"]
    / bgri_cluster_LP["N_INDIVIDUOS_RESIDENT"]
)
bgri_cluster_LP["PER_IND_RESIDENT_ENSINCOMP_2BAS"] = (
    bgri_cluster_LP["N_IND_RESIDENT_ENSINCOMP_2BAS"]
    / bgri_cluster_LP["N_INDIVIDUOS_RESIDENT"]
)
bgri_cluster_LP["PER_IND_RESIDENT_ENSINCOMP_3BAS"] = (
    bgri_cluster_LP["N_IND_RESIDENT_ENSINCOMP_3BAS"]
    / bgri_cluster_LP["N_INDIVIDUOS_RESIDENT"]
)
bgri_cluster_LP["PER_IND_RESIDENT_ENSINCOMP_SEC"] = (
    bgri_cluster_LP["N_IND_RESIDENT_ENSINCOMP_SEC"]
    / bgri_cluster_LP["N_INDIVIDUOS_RESIDENT"]
)
bgri_cluster_LP["PER_IND_RESIDENT_ENSINCOMP_POSEC"] = (
    bgri_cluster_LP["N_IND_RESIDENT_ENSINCOMP_POSEC"]
    / bgri_cluster_LP["N_INDIVIDUOS_RESIDENT"]
)
bgri_cluster_LP["PER_IND_RESIDENT_ENSINCOMP_SUP"] = (
    bgri_cluster_LP["N_IND_RESIDENT_ENSINCOMP_SUP"]
    / bgri_cluster_LP["N_INDIVIDUOS_RESIDENT"]
)
bgri_cluster_LP["PER_IND_RESID_PENS_REFORM"] = (
    bgri_cluster_LP["N_IND_RESID_PENS_REFORM"]
    / bgri_cluster_LP["N_INDIVIDUOS_RESIDENT"]
)
bgri_cluster_LP["PER_IND_RESID_SEM_ACT_ECON"] = (
    bgri_cluster_LP["N_IND_RESID_SEM_ACT_ECON"]
    / bgri_cluster_LP["N_INDIVIDUOS_RESIDENT"]
)
bgri_cluster_LP["PER_IND_RESID_EMPREG_SECT_PRIM"] = (
    bgri_cluster_LP["N_IND_RESID_EMPREG_SECT_PRIM"]
    / bgri_cluster_LP["N_INDIVIDUOS_RESIDENT"]
)
bgri_cluster_LP["PER_IND_RESID_EMPREG_SECT_SEQ"] = (
    bgri_cluster_LP["N_IND_RESID_EMPREG_SECT_SEQ"]
    / bgri_cluster_LP["N_INDIVIDUOS_RESIDENT"]
)
bgri_cluster_LP["PER_IND_RESID_EMPREG_SECT_TERC"] = (
    bgri_cluster_LP["N_IND_RESID_EMPREG_SECT_TERC"]
    / bgri_cluster_LP["N_INDIVIDUOS_RESIDENT"]
)
bgri_cluster_LP["PER_IND_RESID_ESTUD_MUN_RESID"] = (
    bgri_cluster_LP["N_IND_RESID_ESTUD_MUN_RESID"]
    / bgri_cluster_LP["N_INDIVIDUOS_RESIDENT"]
)
bgri_cluster_LP["PER_IND_RESID_TRAB_MUN_RESID"] = (
    bgri_cluster_LP["N_IND_RESID_TRAB_MUN_RESID"]
    / bgri_cluster_LP["N_INDIVIDUOS_RESIDENT"]
)

bgri_cluster_LP.drop(
    [
        "N_INDIVIDUOS_RESIDENT",
        "N_INDIV_RESIDENT_N_LER_ESCRV",
        "N_IND_RESIDENT_FENSINO_1BAS",
        "N_IND_RESIDENT_FENSINO_2BAS",
        "N_IND_RESIDENT_FENSINO_3BAS",
        "N_IND_RESIDENT_FENSINO_SEC",
        "N_IND_RESIDENT_FENSINO_POSSEC",
        "N_IND_RESIDENT_FENSINO_SUP",
        "N_IND_RESIDENT_ENSINCOMP_1BAS",
        "N_IND_RESIDENT_ENSINCOMP_2BAS",
        "N_IND_RESIDENT_ENSINCOMP_3BAS",
        "N_IND_RESIDENT_ENSINCOMP_SEC",
        "N_IND_RESIDENT_ENSINCOMP_POSEC",
        "N_IND_RESIDENT_ENSINCOMP_SUP",
        "N_IND_RESID_DESEMP_PROC_1EMPRG",
        "N_IND_RESID_DESEMP_PROC_EMPRG",
        "N_IND_RESID_EMPREGADOS",
        "N_IND_RESID_PENS_REFORM",
        "N_IND_RESID_SEM_ACT_ECON",
        "N_IND_RESID_EMPREG_SECT_PRIM",
        "N_IND_RESID_EMPREG_SECT_SEQ",
        "N_IND_RESID_EMPREG_SECT_TERC",
        "N_IND_RESID_ESTUD_MUN_RESID",
        "N_IND_RESID_TRAB_MUN_RESID",
    ],
    axis=1,
    inplace=True,
)
bgri_cluster_LP.shape
bgri_cluster_LP.head()
a = list(bgri_cluster_LP.columns)
# retirar o primeiro elemento da lista, para poder standardizar os dados (a servirá como lista de indicadores no próximo passo)
a = a[3:]
a
# Standardizing the features
scaler = StandardScaler()
bgri_cluster_LP[a] = StandardScaler().fit_transform(bgri_cluster_LP[a])
bgri_cluster_LP.head()
bgri_cluster_LP.fillna(0, inplace=True)
X_pca = bgri_cluster_LP[a]
# rotina para verificar quais os pontos que apresentam NaN após a standardização - esses pontos, ou as linhas às quais pertencem, serão excluídas

# primeira iteração do código tinha o bloco "bgri_cluster.drop([24, 32, 47, 94, 101], axis=0, inplace=True)" não implementado, resultando numa
# lista de linhas com pontos NaN - essas linhas são então descartadas, eliminando o problema verificado na Fatorização

x, y = sp.coo_matrix(bgri_cluster_LP.isnull()).nonzero()
print(set(x))
# rotinas utilizadas para verificar a presença de NaNs - linhas dropadas

# set de pontos (X) com NaNs depois do standardscaler()
# este output só aparece se o bloco "bgri_cluster.drop([10, 14, 49, 56, 59, 61, 62], axis=0, inplace=True)" não estiver implementado


# {10, 14, 49, 56, 59, 61, 62}
bgri_cluster_LP
clusters_lp["Cluster_LP"] = clusters_lp["Cluster_LP"].astype("int64")
bgri_cluster_LP["Cluster_LP"] = bgri_cluster_LP["Cluster_LP"].astype("int64")
clusters_lp.reset_index(drop=True, inplace=True)
bgri_cluster_LP.reset_index(drop=True, inplace=True)
# filter clusters file, based on the bgri_cluster file (clusters existing in bgri_cluster)
clusters_lp = clusters_lp.loc[
    clusters_lp["Cluster_LP"].isin(bgri_cluster_LP["Cluster_LP"])
]
clusters_lug = clusters_lug.loc[clusters_lug["LUG11"].isin(clusters_lug["LUG11"])]
clusters_fr = clusters_fr.loc[clusters_fr["FR11"].isin(clusters_fr["FR11"])]
clusters_lp.reset_index(drop=True, inplace=True)
bgri_cluster_LP.reset_index(drop=True, inplace=True)
clusters_lp
len(bgri_cluster_LP.Cluster_LP.unique())
clusters_lp.shape
### 2.2 Factorization (socioeconomic variables)

# PCA para os indicadores de habitação
pca = PCA(n_components=20)

principalComponents = pca.fit_transform(X_pca)

principalDf = pd.DataFrame(
    data=principalComponents,
    columns=[
        "PCA_1",
        "PCA_2",
        "PCA_3",
        "PCA_4",
        "PCA_5",
        "PCA_6",
        "PCA_7",
        "PCA_8",
        "PCA_9",
        "PCA_10",
        "PCA_11",
        "PCA_12",
        "PCA_13",
        "PCA_14",
        "PCA_15",
        "PCA_16",
        "PCA_17",
        "PCA_18",
        "PCA_19",
        "PCA_20",
    ],
)
# variância explicada
sum(pca.explained_variance_ratio_)
pca.explained_variance_ratio_
# eigenvalues, todos superiores a 1
pca.explained_variance_
# olhando para eigenvalues e scree plot, decidiu-se optar por usar eigenvalues superior a 1 para o PCA
# poderia ter sido utilizado outro método, como o elbow method, mas este é mais simples e intuitivo e garante mais de 80% da variância explicada

PC_values = np.arange(pca.n_components) + 1
plt.plot(PC_values, pca.explained_variance_, "o-", linewidth=2, color="blue")
plt.title("Scree Plot")
plt.xlabel("Principal Component")
plt.ylabel("Variance Explained")
plt.show()
# PCA para os indicadores de habitação
pca = PCA(n_components=17)

principalComponents = pca.fit_transform(X_pca)

principalDf = pd.DataFrame(
    data=principalComponents,
    columns=[
        "PCA_1",
        "PCA_2",
        "PCA_3",
        "PCA_4",
        "PCA_5",
        "PCA_6",
        "PCA_7",
        "PCA_8",
        "PCA_9",
        "PCA_10",
        "PCA_11",
        "PCA_12",
        "PCA_13",
        "PCA_14",
        "PCA_15",
        "PCA_16",
        "PCA_17",
    ],
)
loadings = pd.DataFrame(
    pca.components_.T,
    columns=[
        "PCA_1",
        "PCA_2",
        "PCA_3",
        "PCA_4",
        "PCA_5",
        "PCA_6",
        "PCA_7",
        "PCA_8",
        "PCA_9",
        "PCA_10",
        "PCA_11",
        "PCA_12",
        "PCA_13",
        "PCA_14",
        "PCA_15",
        "PCA_16",
        "PCA_17",
    ],
    index=X_pca.columns,
)
bgri_cluster_LP.reset_index(inplace=True, drop=True)
principalDf.shape
# Concatenar os dados
bgri_cluster_LP = pd.concat([bgri_cluster_LP, principalDf], axis=1)
bgri_cluster_LP.shape
bgri_cluster_LP.drop(columns=["FR11", "LUG11"], inplace=True)
bgri_cluster_LP
#### 2.2.a PCA Loadings

# view loadings for first principal component group
loadings
### 2.3 Dentrogram (socioeconomic variables)

# calculo do dendrograma com método ward
dendrogram = sch.dendrogram(
    sch.linkage(
        bgri_cluster_LP[
            [
                "PCA_1",
                "PCA_2",
                "PCA_3",
                "PCA_4",
                "PCA_5",
                "PCA_6",
                "PCA_7",
                "PCA_8",
                "PCA_9",
                "PCA_10",
                "PCA_11",
                "PCA_12",
                "PCA_13",
                "PCA_14",
                "PCA_15",
                "PCA_16",
                "PCA_17",
            ]
        ],
        method="ward",
    )
)
plt.title("Dendrograma")
plt.xlabel("Clientes")
plt.ylabel("Distâncias Euclidianas")
plt.axhline(16.2, color="red", linestyle="--", linewidth=1)
plt.grid(False)
plt.show()
# number of clusters suggested by the dendrogram
n_clusters = 9
## 3. Clustering

### 3.1 Ward Linkage

# Agglomerative Clustering, no contiguity matrix, ward linkage
hc = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
y_hc = hc.fit_predict(bgri_cluster_LP)
# write labels in our dataframe
bgri_cluster_LP["Zona_Ward"] = y_hc.T
bgri_cluster_LP.head()
# prepare data for merging
bgri_cluster2 = bgri_cluster_LP.iloc[:, -18:]
bgri_cluster2
# prepare data for merging
bgri_cluster3 = bgri_cluster_LP.iloc[:, :1]
bgri_cluster3
# concatenate dataframes
bgri_cluster_LP = pd.concat([bgri_cluster2, bgri_cluster3], axis=1)
bgri_cluster_LP.head()
# merge relevant data (resulting from the PCA) with the cluster dataframe
clusters_lp = clusters_lp.merge(bgri_cluster_LP, how="left", on="Cluster_LP")
clusters_lp.head()
clusters_lp.shape
# remove the column with the cluster Zona Ward labels
bgri_cluster_LP.drop(["Zona_Ward"], axis=1, inplace=True)
# print the result from the clustering done above
ax = clusters_lp.plot(
    figsize=(10, 10),
    column="Zona_Ward",
    categorical=True,
    edgecolor="b",
    legend=True,
    linewidth=0.2,
    cmap="tab20",
)
cx.add_basemap(ax, crs=clusters_lp.crs, source=cx.providers.OpenStreetMap.Mapnik)
plt.title("Clusterização - Ward Linkage. Clusters = {}".format(n_clusters), fontsize=16)
### 3.2 Ward Linkage + Queen Contiguity

# Contiguity matrix Queen
RANDOM_SEED = 123456

wqueen = Queen.from_dataframe(clusters_lp)
# Contiguity matrix Queen (arry like)
df = pd.DataFrame(*wqueen.full()).astype(int)

arr = df.to_numpy()

arr2d = np.transpose(arr)
wqueen.set_transform("R")
plot_queen = plot_spatial_weights(wqueen, clusters_lp)
plt.title(
    "Matriz de Contiguidade 'Queen' aplicada às unidades territoriais base", fontsize=14
)
plt.show()
# Vamos repetir o processo para o método de Ward matriz de contiguidade

hc2 = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward", connectivity=arr2d)
y_hc2 = hc2.fit_predict(bgri_cluster_LP)
clusters_lp["Zona_Ward_Queen"] = y_hc2.T
clusters_lp.head()
# print the result from the clustering done above
ax = clusters_lp.plot(
    figsize=(10, 10),
    column="Zona_Ward_Queen",
    categorical=True,
    edgecolor="b",
    legend=True,
    linewidth=0.2,
    cmap="tab20",
)
cx.add_basemap(ax, crs=clusters_lp.crs, source=cx.providers.OpenStreetMap.Mapnik)
plt.title(
    "Clusterização - Ward Linkage. Matriz contiguidade Queen. Clusters = {}".format(
        n_clusters
    ),
    fontsize=14,
)
### 3.3 Max-P Regionalization


# load data (pickle) casasapo
casasapo = pd.read_pickle("Data/piclo_casasapo.piclo")

# load data (pickle) py
py = pd.read_pickle("Data/piclo_py.piclo")
# definition of seed
RANDOM_SEED = 123456
casasapo["a"] = 1
py["a"] = 1
casasapo["Cluster_LP"] = casasapo["Cluster_LP"].astype("Int64")
# preparation to calculate the number of dwellings per cluster
df_subtotal_cs = pd.DataFrame()
df_subtotal_py = pd.DataFrame()
df_subtotal_cs = casasapo.groupby("Cluster_LP", as_index=False)[["a"]].agg("sum")
df_subtotal_py = py.groupby("Cluster_LP", as_index=False)[["a"]].agg("sum")
# merge with clusters_lp dataframe
clusters_lp = clusters_lp.merge(
    df_subtotal_cs[["Cluster_LP", "a"]], how="left", on="Cluster_LP"
)
clusters_lp = clusters_lp.merge(
    df_subtotal_py[["Cluster_LP", "a"]], how="left", on="Cluster_LP"
)
# drop column 'a'
casasapo.drop(columns=["a"], inplace=True)
py.drop(columns=["a"], inplace=True)
# rename columns
clusters_lp.rename(columns={"a_x": "tot_cs", "a_y": "tot_py"}, inplace=True)
clusters_lp.head()
clusters_lp[["tot_cs", "tot_py"]].describe()
# define the minimum number of dwellings per cluster, pre and post intervention (casasapo and py)
clusters_lp["tot_min"] = clusters_lp[["tot_cs", "tot_py"]].min(axis=1)
attrs_name = list(
    clusters_lp[
        [
            "PCA_1",
            "PCA_2",
            "PCA_3",
            "PCA_4",
            "PCA_5",
            "PCA_6",
            "PCA_7",
            "PCA_8",
            "PCA_9",
            "PCA_10",
            "PCA_11",
            "PCA_12",
            "PCA_13",
            "PCA_14",
            "PCA_15",
            "PCA_16",
            "PCA_17",
        ]
    ]
)
# minimum number of dwellings per cluster
threshold = 102
# The number of top candidate regions to consider for enclave assignment.
top_n = 3
# criteria for the alghoritm - number of dwellings per cluster
threshold_name = "tot_min"
# model with MaxP
np.random.seed(RANDOM_SEED)
model_maxp = MaxP(clusters_lp, wqueen, attrs_name, threshold_name, threshold, top_n)
model_maxp.solve()
clusters_lp["Zona_Maxp"] = model_maxp.labels_
clusters_lp[["Zona_Maxp"]].groupby(by="Zona_Maxp").count()
a = model_maxp.p
a
# result from the Max-P Regionalization
ax = clusters_lp.plot(
    figsize=(10, 10),
    column="Zona_Maxp",
    categorical=True,
    edgecolor="b",
    legend=True,
    linewidth=0.2,
    cmap="tab10",
)
cx.add_basemap(ax, crs=clusters_lp.crs, source=cx.providers.OpenStreetMap.Mapnik)
plt.title(
    "Clusterização - Max-P. Threshold = %s imóveis. Clusters = %s" % (threshold, a),
    fontsize=16,
)
clusters_lp.head()
### 3.4 SKATER (Spatial ’K’luster Analysis by Tree Edge Removal)

# minimum number of zones per cluster
floor = 3
# Flag denoting whether to store intermediate labelings as the tree gets pruned. (default False)
trace = False
# Description of what to do with islands. If 'ignore', the algorithm will discover n_clusters regions, treating islands as their own regions.
# If “increase”, the algorithm will discover n_clusters regions, treating islands as separate from n_clusters. (default ‘increase’)
islands = "increase"
# standard definition for the spannig tree algorithm
default = dict(
    dissimilarity=skm.manhattan_distances,
    affinity=None,
    reduction=np.sum,
    center=np.mean,
    verbose=False,
)
# model using the skater algorithm
model_skater = spopt.region.Skater(
    clusters_lp,
    wqueen,
    attrs_name,
    n_clusters=n_clusters,
    floor=floor,
    trace=trace,
    islands=islands,
    spanning_forest_kwds=default,
)
model_skater.solve()
# write skater cluster info
clusters_lp["Zona_SKATER"] = model_skater.labels_
# resulting number of clusters, for the graph title
temp = len(clusters_lp["Zona_SKATER"].unique())
temp
# result of the SKATER Regionalization
ax = clusters_lp.plot(
    figsize=(10, 10),
    column="Zona_SKATER",
    categorical=True,
    edgecolor="b",
    legend=True,
    linewidth=0.2,
    cmap="tab20",
)
cx.add_basemap(ax, crs=clusters_lp.crs, source=cx.providers.OpenStreetMap.Mapnik)
plt.title("Clusterização - SKATER. Clusters = {}".format(temp), fontsize=16)
# result of the SKATER Regionalization
ax = clusters_lp.plot(
    figsize=(10, 10),
    column="Zona_SKATER",
    categorical=True,
    legend=False,
    linewidth=0.1,
    cmap="tab20",
)
cx.add_basemap(ax, crs=clusters_lp.crs, source=cx.providers.OpenStreetMap.Mapnik)
ax.set_title("SKATER Clusters", fontweight="bold", fontsize=16)
ax.set_axis_off()
# 4. Export Data to Pickle Files

# save data (pickle) bgri
bgri_cluster.to_pickle("Data/piclo_bgri_2.piclo")
# save data (pickle) clusters
clusters_lp.to_pickle("Data/piclo_clusters_2.piclo")
# 5. Cluster Metrics

x_val = clusters_lp[
    [
        "PCA_1",
        "PCA_2",
        "PCA_3",
        "PCA_4",
        "PCA_5",
        "PCA_6",
        "PCA_7",
        "PCA_8",
        "PCA_9",
        "PCA_10",
        "PCA_11",
        "PCA_12",
        "PCA_13",
        "PCA_14",
        "PCA_15",
        "PCA_16",
        "PCA_17",
    ]
]
clusters_lp.head()
## 5.1 - Metrics - Ward Linkage

score_ward = silhouette_score(x_val, clusters_lp.Zona_Ward, metric="manhattan")
silhouettes_ward = silhouette_samples(x_val, clusters_lp.Zona_Ward)
## 5.2 - Metrics - Ward Linkage + Queen Contiguity

score_ward_queen = silhouette_score(
    x_val, clusters_lp.Zona_Ward_Queen, metric="manhattan"
)
silhouettes_ward_queen = silhouette_samples(x_val, clusters_lp.Zona_Ward_Queen)
## 5.3 - Metrics - Max-P Regionalization

score_maxp = silhouette_score(x_val, clusters_lp.Zona_Maxp, metric="manhattan")
silhouettes_maxp = silhouette_samples(x_val, clusters_lp.Zona_Maxp)
## 5.4 - Metrics - SKATER (Spatial ’K’luster Analysis by Tree Edge Removal)

score_skater = silhouette_score(x_val, clusters_lp.Zona_SKATER, metric="manhattan")
silhouettes_skater = silhouette_samples(x_val, clusters_lp.Zona_SKATER)
## 5.5 - Metrics - Clustering - Comparison

### 5.5.1 - Comparison - Silhouette Score (average) - Sklearn

score = [score_ward, score_ward_queen, score_maxp, score_skater]
method = ["Ward", "Ward Queen", "Max-P", "SKATER"]
# Plot dos histogramas das silhuetas para cada clusterização (média), para cada método de clusterização

plt.bar(method, score)
plt.xlabel("Método de Clusterização", fontsize=12)
plt.ylabel("Coeficiente", fontsize=12)
plt.title("Coeficiente da Silhueta (média)", fontsize=16)
plt.grid(False)
plt.show()
### 5.5.2 - Comparison - Silhouette Score (Samples) - Sklearn

# Plot dos histogramas das silhuetas para cada cluster, para cada método de clusterização, com a média de cada cluster

f, ax = plt.subplots(4, 2, figsize=(8, 12))
ax[0, 0].hist(silhouettes_ward)
clusters_lp.plot(
    silhouettes_ward, ax=ax[0, 1], cmap="viridis", vmin=-0.5, vmax=0.5, legend=True
)
ax[1, 0].hist(silhouettes_ward_queen)
clusters_lp.plot(
    silhouettes_ward_queen,
    ax=ax[1, 1],
    cmap="viridis",
    vmin=-0.5,
    vmax=0.5,
    legend=True,
)
ax[2, 0].hist(silhouettes_maxp)
clusters_lp.plot(
    silhouettes_maxp, ax=ax[2, 1], cmap="viridis", vmin=-0.5, vmax=0.5, legend=True
)
ax[3, 0].hist(silhouettes_skater)
clusters_lp.plot(
    silhouettes_skater, ax=ax[3, 1], cmap="viridis", vmin=-0.5, vmax=0.5, legend=True
)
ax[0, 0].set_title("Ward - Coeficiente de Silhueta")
ax[0, 0].grid(False)
ax[0, 1].set_title("Ward - Coeficiente de Silhueta")
ax[0, 1].axes.get_xaxis().set_visible(False)
ax[0, 1].axes.get_yaxis().set_visible(False)
ax[1, 0].set_title("Ward + Queen - Coeficiente de Silhueta")
ax[1, 0].grid(False)
ax[1, 1].set_title("Ward + Queen - Coeficiente de Silhueta")
ax[1, 1].axes.get_xaxis().set_visible(False)
ax[1, 1].axes.get_yaxis().set_visible(False)
ax[2, 0].set_title("Max-P - Coeficiente de Silhueta")
ax[2, 0].grid(False)
ax[2, 1].set_title("Max-P - Coeficiente de Silhueta")
ax[2, 1].axes.get_xaxis().set_visible(False)
ax[2, 1].axes.get_yaxis().set_visible(False)
ax[3, 0].set_title("SKATER - Coeficiente de Silhueta")
ax[3, 0].grid(False)
ax[3, 1].set_title("SKATER - Coeficiente de Silhueta")
ax[3, 1].axes.get_xaxis().set_visible(False)
ax[3, 1].axes.get_yaxis().set_visible(False)
f.tight_layout()
plt.show()


# Modelling (AIC)

## 1 Data Preparation for Modelling

### 1.1 Loading and Transforming Datasets

all_data = pd.read_pickle("Data/all_data.piclo")
all_data.shape
# # filtro de outliers, calculados ao longo das regressões realizadas no notebook 3 (Modelling) e adicionado ao longo do processo
# # (linha acima foram usadas para atualizar esta lista até o retorno de um conjunto vazio)
# # a cada "run" do notebook, esta lista deve ser atualizada com os outliers identificados
# # parou-se de atualizar a lista quando o conjunto de outliers começou, repetidamente, a devolver apenas 2 ou 3 outliers

# run 1 - 13 outliers
all_data = all_data.loc[all_data["ID"] != 563242]
all_data = all_data.loc[all_data["ID"] != 1177680]
all_data = all_data.loc[all_data["ID"] != 1131818]
all_data = all_data.loc[all_data["ID"] != 2383113]
all_data = all_data.loc[all_data["ID"] != 2568707]
all_data = all_data.loc[all_data["ID"] != 2639959]
all_data = all_data.loc[all_data["ID"] != 2632030]
all_data = all_data.loc[all_data["ID"] != 2870116]
all_data = all_data.loc[all_data["ID"] != 2743956]
all_data = all_data.loc[all_data["ID"] != 2939163]
all_data = all_data.loc[all_data["ID"] != 2953594]
all_data = all_data.loc[all_data["ID"] != 2939578]
all_data = all_data.loc[all_data["ID"] != 3089766]

# run 2 - 12 outliers
all_data = all_data.loc[all_data["ID"] != 563276]
all_data = all_data.loc[all_data["ID"] != 1254649]
all_data = all_data.loc[all_data["ID"] != 1253964]
all_data = all_data.loc[all_data["ID"] != 2251268]
all_data = all_data.loc[all_data["ID"] != 2569071]
all_data = all_data.loc[all_data["ID"] != 2644045]
all_data = all_data.loc[all_data["ID"] != 2631437]
all_data = all_data.loc[all_data["ID"] != 2712461]
all_data = all_data.loc[all_data["ID"] != 2696442]
all_data = all_data.loc[all_data["ID"] != 2939264]
all_data = all_data.loc[all_data["ID"] != 2811941]
all_data = all_data.loc[all_data["ID"] != 3161153]

# run 3 - 12 outliers
all_data = all_data.loc[all_data["ID"] != 563391]
all_data = all_data.loc[all_data["ID"] != 1235755]
all_data = all_data.loc[all_data["ID"] != 1247318]
all_data = all_data.loc[all_data["ID"] != 2380527]
all_data = all_data.loc[all_data["ID"] != 2569178]
all_data = all_data.loc[all_data["ID"] != 2638746]
all_data = all_data.loc[all_data["ID"] != 2629208]
all_data = all_data.loc[all_data["ID"] != 2718713]
all_data = all_data.loc[all_data["ID"] != 2875618]
all_data = all_data.loc[all_data["ID"] != 2687064]
all_data = all_data.loc[all_data["ID"] != 2958943]
all_data = all_data.loc[all_data["ID"] != 3120730]
all_data
all_data.dtypes
all_data["Cluster_LP"].unique()
all_data[all_data.isnull().any(axis=1)]
# convertion of data
all_data["Year"] = all_data["Year"].astype("int64")
all_data["Cluster_LP"] = all_data["Cluster_LP"].astype("int64")
all_data.shape
all_data["T"].sum()
# a=set(outliers_sum)
# all_data.loc[list(a)]
all_data.shape
all_data["T"].value_counts()
all_data["Year"].value_counts()
# valores do Indice de Preços da Habitação para Aveiro (valores para 21, 22 e 23 são previsões, com base na tendência dos últimos anos)
IPI2005 = 116.5
IPI2006 = 110.9
IPI2007 = 114.3
IPI2008 = 109.4
IPI2009 = 103.1
IPI2010 = 104.4
IPI2018 = 100.8
IPI2019 = 114.2
IPI2020 = 126.2
IPI2021 = 132.1
IPI2022 = 133.7
IPI2023 = 135.4


# write IPI values to new column
def new_column_value(Year):
    if Year == 2005:
        return IPI2005
    elif Year == 2006:
        return IPI2006
    elif Year == 2007:
        return IPI2007
    elif Year == 2008:
        return IPI2008
    elif Year == 2009:
        return IPI2009
    elif Year == 2010:
        return IPI2010
    elif Year == 2018:
        return IPI2018
    elif Year == 2019:
        return IPI2019
    elif Year == 2020:
        return IPI2020
    elif Year == 2021:
        return IPI2021
    elif Year == 2022:
        return IPI2022
    elif Year == 2023:
        return IPI2023


all_data["IPI"] = all_data["Year"].apply(new_column_value)
all_data.shape
# valores da Taxa Anual de juro (TAA) de novos empréstimos à habitação (BdP)
TAA2005 = 3.38
TAA2006 = 4.01
TAA2007 = 4.8
TAA2008 = 5.44
TAA2009 = 2.73
TAA2010 = 2.47
TAA2018 = 1.41
TAA2019 = 1.22
TAA2020 = 1
TAA2021 = 0.81
TAA2022 = 1.82
TAA2023 = 3.76


# write IPI values to new column
def new_column_value(Year):
    if Year == 2005:
        return TAA2005
    elif Year == 2006:
        return TAA2006
    elif Year == 2007:
        return TAA2007
    elif Year == 2008:
        return TAA2008
    elif Year == 2009:
        return TAA2009
    elif Year == 2010:
        return TAA2010
    elif Year == 2018:
        return TAA2018
    elif Year == 2019:
        return TAA2019
    elif Year == 2020.0:
        return TAA2020
    elif Year == 2021.0:
        return TAA2021
    elif Year == 2022:
        return TAA2022
    elif Year == 2023:
        return TAA2023


all_data["TAA"] = all_data["Year"].apply(new_column_value)
all_data.shape
## poucos valores para os anos 2023 e 2005 (início de uma base de dados e fim da outra - serão removidos)

all_data = all_data.loc[all_data["Year"] != 2023]
all_data = all_data.loc[all_data["Year"] != 2005]
all_data.shape
all_data.columns
a = list(all_data[["Tot_AL", "IPI", "TAA"]])
# Standardizing the features
scaler = StandardScaler()
all_data[a] = StandardScaler().fit_transform(all_data[a])
all_data.head()
### 1.2 MCA for Intrinsic Features


all_data.head()
all_data.columns
a = ["Typology", "Nature", "Status"]
a
# OK
mca = prince.MCA(
    n_components=10,
    n_iter=3,
    copy=True,
    check_input=True,
    engine="sklearn",
    random_state=42,
)
mca.fit(all_data[a])
mca.eigenvalues_summary
mca.column_contributions_.style.format("{:.0%}")
PC_values = np.arange(mca.n_components) + 1
plt.plot(
    PC_values,
    [17.15, 11.88, 10.52, 10.11, 10, 10, 9.88, 9.30, 8.06, 3.10],
    "o-",
    linewidth=2,
    color="blue",
)
plt.axhline(10.14, color="green", linestyle="--", linewidth=1)
plt.title("Scree Plot")
plt.xlabel("Componentes Principais")
plt.ylabel("Variancia Explicada")
plt.show()
all_data["MCA_1"] = mca.transform(all_data[a])[0]
all_data["MCA_2"] = mca.transform(all_data[a])[1]
all_data["MCA_3"] = mca.transform(all_data[a])[2]
all_data["MCA_4"] = mca.transform(all_data[a])[3]
all_data.tail()
all_data.shape
all_data.reset_index(inplace=True, drop=True)
### 1.3 Generate Discriptive Statistics HTML

all_data.columns
all_data["Nature"] = all_data["Nature"].astype("category")
all_data["Typology"] = all_data["Typology"].astype("category")
all_data["Status"] = all_data["Status"].astype("category")
# Generate the report - Final

all_data_analysis = all_data.copy()
all_data_analysis = all_data_analysis[
    ["Log_P_A", "MCA_1", "MCA_2", "MCA_3", "MCA_4", "TAA", "IPI", "Tot_AL"]
]
profile = ProfileReport(
    all_data_analysis,
    title="AIC Data Profile Report",
    explorative=True,
    config_file="Data/config_default.yaml",
)
profile.to_notebook_iframe()
# # Save the report to .html
profile.to_file("AIC_Data_Profile_Report_MCA.html")
# Get min, max, mean, std dev for the dataset

all_data.describe()
all_data.describe()
### 1.4 Create GeoDataFrames with all_data + different territorial limits (Freguesias, Skater Cluster, Cluster LP)

#### 1.4.1 Create GeoDataFrames with all_data + Cluster_LP

Cluster_LP = pd.read_pickle("Data/piclo_clusters_2.piclo")
Cluster_LP.head()
Cluster_LP.shape
all_data.shape
all_data = all_data[~all_data["Cluster_LP"].isnull()]
all_data.shape
# merge dwelling data with clusters data
all_data_LP = all_data.merge(Cluster_LP, on="Cluster_LP", how="left")
all_data_LP.columns
all_data_LP.shape
# drop unnecessary columns
all_data_LP.drop(
    columns=[
        "PCA_1",
        "PCA_2",
        "PCA_3",
        "PCA_4",
        "PCA_5",
        "PCA_6",
        "PCA_7",
        "PCA_8",
        "PCA_9",
        "PCA_10",
        "PCA_11",
        "PCA_12",
        "PCA_13",
        "PCA_14",
        "PCA_15",
        "PCA_16",
        "PCA_17",
        "tot_cs",
        "tot_py",
        "tot_min",
        "Price",
    ],
    axis=1,
    inplace=True,
)
# drop dwellings with no lat lon information (geometry)
all_data_LP2 = all_data_LP[all_data_LP["geometry"].isna()]
all_data_LP2["geometry"].unique()
all_data_LP2.columns
all_data_LP2["Cluster_LP"].unique()
# drop dwellings with no lat lon information (geometry)
all_data_LP = all_data_LP[~all_data_LP["geometry"].isna()]
all_data_LP.shape
a = all_data_LP[all_data_LP["T"] == 1]
a
all_data_LP.columns
len(all_data_LP["Cluster_LP"].unique())
all_data_LP.dtypes
# convertion of data to float
all_data_LP["Zona_Ward"] = np.floor(
    pd.to_numeric(all_data_LP["Zona_Ward"], errors="coerce")
).astype("float64")
all_data_LP["Zona_Ward_Queen"] = np.floor(
    pd.to_numeric(all_data_LP["Zona_Ward_Queen"], errors="coerce")
).astype("float64")
all_data_LP["Zona_Maxp"] = np.floor(
    pd.to_numeric(all_data_LP["Zona_Maxp"], errors="coerce")
).astype("float64")
all_data_LP["Zona_SKATER"] = np.floor(
    pd.to_numeric(all_data_LP["Zona_SKATER"], errors="coerce")
).astype("float64")
all_data_LP["Cluster_LP"] = np.floor(
    pd.to_numeric(all_data_LP["Cluster_LP"], errors="coerce")
).astype("float64")
all_data_LP["Year"] = np.floor(
    pd.to_numeric(all_data_LP["Year"], errors="coerce")
).astype("float64")
all_data_LP["T"] = np.floor(pd.to_numeric(all_data_LP["T"], errors="coerce")).astype(
    "float64"
)
all_data_LP.dtypes
all_data_LP.head()
all_data_LP.columns
all_data_LP.shape
all_data_LP["geometry"].nunique()
#### 1.4.2 Create GeoDataFrames with all_data + Skater

all_data_skater = all_data_LP.copy()
all_data_skater.isnull().values.any()
all_data_skater.columns
all_data_skater = gpd.GeoDataFrame(all_data_skater, geometry="geometry")
all_data_skater_geo = all_data_skater.dissolve(by="Zona_SKATER").reset_index()
all_data_skater_geo = all_data_skater_geo[["Zona_SKATER", "geometry"]]
all_data_skater = all_data_skater.merge(
    all_data_skater_geo, on="Zona_SKATER", how="left"
)
all_data_skater.shape
all_data_skater.columns
all_data_skater.drop(columns=["geometry_x"], axis=1, inplace=True)
all_data_skater.rename(columns={"geometry_y": "geometry"}, inplace=True)
all_data_skater = gpd.GeoDataFrame(all_data_skater, geometry="geometry")
all_data_skater.shape
all_data_skater = all_data_skater[~all_data_skater["Cluster_LP"].isnull()]
all_data_skater = all_data_skater[~all_data_skater["Zona_SKATER"].isnull()]
all_data_skater.shape
len(all_data_skater["Zona_SKATER"].unique())
all_data_skater.dtypes
all_data_skater.head()
all_data_skater.columns
all_data_skater.shape
all_data_skater["geometry"].nunique()
all_data_skater["Year"].nunique()
#### 1.4.3 Create GeoDataFrames with all_data + FR

all_data_fr = all_data_LP.copy()
# transform data to geodataframe
all_data_fr = gpd.GeoDataFrame(all_data_fr, geometry="geometry")
all_data_fr.plot(
    column="Cluster_LP", categorical=True, legend=False, figsize=(10, 10), cmap="tab20c"
)
all_data_fr.columns
CLUSTER_FR = pd.read_pickle("Data/piclo_clusters_fr.piclo")
CLUSTER_FR
CLUSTER_FR.plot()
CLUSTER_LP = pd.read_pickle("Data/piclo_clusters_lp.piclo")
CLUSTER_LP.plot()
CLUSTER_LP["geometry2"] = CLUSTER_LP.centroid
CLUSTER_LP.rename(
    columns={"geometry": "geometry2", "geometry2": "geometry"}, inplace=True
)
CLUSTER_LP.drop(columns=["geometry2"], inplace=True)
CLUSTER_LP
CLUSTER = CLUSTER_FR.sjoin(CLUSTER_LP, how="inner", predicate="intersects")
CLUSTER.shape
CLUSTER.head()
CLUSTER["Cluster_LP"] = CLUSTER["Cluster_LP"].astype("float64")
CLUSTER["FR11"] = CLUSTER["FR11"].astype("Int64")
CLUSTER = CLUSTER[["Cluster_LP", "FR11", "geometry"]]
all_data_fr.shape
all_data_fr.head()
all_data_fr = all_data_fr.merge(CLUSTER, on="Cluster_LP", how="left")
all_data_fr.columns
all_data_fr.drop(columns=["geometry_x"], axis=1, inplace=True)
all_data_fr.rename(columns={"geometry_y": "geometry"}, inplace=True)
all_data_fr = all_data_fr[~all_data_fr["Cluster_LP"].isnull()]
all_data_fr = all_data_fr[~all_data_fr["Zona_SKATER"].isnull()]
all_data_fr = all_data_fr[~all_data_fr["FR11"].isnull()]
all_data_fr.head()
all_data_fr.shape
all_data_fr["geometry"].nunique()
## 2 Linear Regressions (Aveiro) - SKATER

### 2.1 Data Preparation for DID Linear Regression

# transform data to geodataframe
data_aveiro_skater = gpd.GeoDataFrame(all_data_skater, geometry="geometry")
data_aveiro_skater.columns
data_aveiro_skater["T"].value_counts()
# view Zona_SKATER clusters
ax = data_aveiro_skater.plot(
    figsize=(10, 10),
    column="Zona_SKATER",
    categorical=True,
    edgecolor="w",
    legend=True,
    linewidth=0.2,
    cmap="tab20",
)
cx.add_basemap(ax, crs=data_aveiro_skater.crs, source=cx.providers.OpenStreetMap.Mapnik)
# apply above list to data
data_aveiro_skater["D"] = np.where(
    (data_aveiro_skater["Zona_SKATER"] == 3.0)
    | (data_aveiro_skater["Zona_SKATER"] == 6.0)
    | (data_aveiro_skater["Zona_SKATER"] == 7.0),
    1,
    0,
)
data_aveiro_skater.head()
data_aveiro_skater.dtypes
# check no. of dwellings per cluster in Aveiro Center
pd.pivot_table(
    data_aveiro_skater,
    values="Log_P_A",
    index=["Zona_SKATER"],
    aggfunc=lambda x: len(x.unique()),
).head(10)
# não incluir cluster 80, 102 e 106, por escassez de dados
data_aveiro_skater.to_pickle("Data/data_aveiro_skater.piclo")
data_aveiro_skater["D"].value_counts()
a = data_aveiro_skater[data_aveiro_skater["D"] == 0]
a[["Log_P_A", "Tot_AL", "TAA", "IPI", "MCA_1", "MCA_2", "MCA_3", "MCA_4"]].describe()
sum(data_aveiro_skater["D"].value_counts())
# result of the SKATER Regionalization
ax = data_aveiro_skater.plot(
    figsize=(10, 10),
    column=data_aveiro_skater["D"],
    categorical=True,
    legend=True,
    linewidth=0.1,
    cmap="tab20",
)
cx.add_basemap(ax, crs=data_aveiro_skater.crs, source=cx.providers.OpenStreetMap.Mapnik)
ax.set_title(
    "Treatment Group (1) and Control Group (0)", fontweight="bold", fontsize=16
)
ax.set_axis_off()
# check areas defined as control and intervention areas (D=0 and D=1)
ax = data_aveiro_skater.plot(
    column=data_aveiro_skater["D"],
    categorical=True,
    legend=True,
    figsize=(10, 10),
    cmap="tab20",
)
plt.title("Zona de Intervenção (1) e Zona de Controlo (0)")
cx.add_basemap(ax, crs=data_aveiro_skater.crs, source=cx.providers.OpenStreetMap.Mapnik)
# calculate DT (true when both D and T are equal to 1)
data_aveiro_skater["DT"] = data_aveiro_skater["D"] * data_aveiro_skater["T"]
data_aveiro_skater["DT"].value_counts()
# check log_P_A distribution in the territory
ax = data_aveiro_skater.plot(
    column=np.exp(data_aveiro_skater["Log_P_A"]),
    legend=True,
    figsize=(10, 10),
    cmap="viridis",
    scheme="quantiles",
    k=6,
    linewidth=0.3,
    edgecolor="black",
)
plt.title("Preço médio dos imóveis (€/m2), por Cluster (SKATER)")
cx.add_basemap(ax, crs=data_aveiro_skater.crs, source=cx.providers.OpenStreetMap.Mapnik)
### 2.2 Linear Regression (focused in Aveiro Center - SKATER Cluster)

data_aveiro_skater.shape
# check number of dwellings per cluster
pd.pivot_table(
    data_aveiro_skater,
    values="Log_P_A",
    index=["Zona_SKATER"],
    aggfunc=lambda x: len(x.unique()),
).head(10)
# check number of dwellings per cluster
pd.pivot_table(
    data_aveiro_skater, values="DT", index=["Zona_SKATER"], aggfunc="sum"
).head(10)
# check number of dwellings per cluster
pd.pivot_table(
    data_aveiro_skater,
    values="Zona_SKATER",
    index=["D"],
    aggfunc=lambda x: len(x.unique()),
)
data_aveiro_skater.columns
# get dummies for skater zones
data_aveiro_skater_ols = pd.get_dummies(
    data_aveiro_skater, columns=["Zona_SKATER"], drop_first=True, dtype=float
)
data_aveiro_skater_ols.columns
# define X and y
c_y_eur_area_skater = data_aveiro_skater_ols["Log_P_A"].astype(float)

c_X_eur_area_skater = data_aveiro_skater_ols[
    [
        "DT",
        "MCA_1",
        "MCA_2",
        "MCA_3",
        "MCA_4",
        "IPI",
        "Tot_AL",
        "TAA",
        "Zona_SKATER_1.0",
        "Zona_SKATER_2.0",
        "Zona_SKATER_3.0",
        "Zona_SKATER_4.0",
        "Zona_SKATER_5.0",
        "Zona_SKATER_6.0",
        "Zona_SKATER_7.0",
        "Zona_SKATER_8.0",
    ]
].astype(float)
# linear model for casasapo with log of price per square meter as dependent variable
c_X_eur_area_skater = sm.add_constant(c_X_eur_area_skater)
model_c_eur_area_skater = sm.OLS(c_y_eur_area_skater, c_X_eur_area_skater)
results_c_eur_area_skater = model_c_eur_area_skater.fit()
skater_save = results_c_eur_area_skater.summary2()
results_c_eur_area_skater.summary2()
# # export results to csv
# skater_save.tables[0].to_csv('skater_0.csv')
# skater_save.tables[1].to_csv('skater_1.csv')

# base code for diagnostic plots

style_talk = "seaborn-talk"  # refer to plt.style.available


class Linear_Reg_Diagnostic:
    """
    Diagnostic plots to identify potential problems in a linear regression fit.
    Mainly,
        a. non-linearity of data
        b. Correlation of error terms
        c. non-constant variance
        d. outliers
        e. high-leverage points
        f. collinearity

    Author:
        Prajwal Kafle (p33ajkafle@gmail.com, where 3 = r)
        Does not come with any sort of warranty.
        Please test the code one your end before using.
    """

    def __init__(
        self,
        results: Type[statsmodels.regression.linear_model.RegressionResultsWrapper],
    ) -> None:
        """
        For a linear regression model, generates following diagnostic plots:

        a. residual
        b. qq
        c. scale location and
        d. leverage

        and a table

        e. vif

        Args:
            results (Type[statsmodels.regression.linear_model.RegressionResultsWrapper]):
                must be instance of statsmodels.regression.linear_model object

        Raises:
            TypeError: if instance does not belong to above object

        Example:
        >>> import numpy as np
        >>> import pandas as pd
        >>> import statsmodels.formula.api as smf
        >>> x = np.linspace(-np.pi, np.pi, 100)
        >>> y = 3*x + 8 + np.random.normal(0,1, 100)
        >>> df = pd.DataFrame({'x':x, 'y':y})
        >>> res = smf.ols(formula= "y ~ x", data=df).fit()
        >>> cls = Linear_Reg_Diagnostic(res)
        >>> cls(plot_context="seaborn-paper")

        In case you do not need all plots you can also independently make an individual plot/table
        in following ways

        >>> cls = Linear_Reg_Diagnostic(res)
        >>> cls.residual_plot()
        >>> cls.qq_plot()
        >>> cls.scale_location_plot()
        >>> cls.leverage_plot()
        >>> cls.vif_table()
        """

        if (
            isinstance(
                results, statsmodels.regression.linear_model.RegressionResultsWrapper
            )
            is False
        ):
            raise TypeError(
                "result must be instance of statsmodels.regression.linear_model.RegressionResultsWrapper object"
            )

        self.results = maybe_unwrap_results(results)

        self.y_true = self.results.model.endog
        self.y_predict = self.results.fittedvalues
        self.xvar = self.results.model.exog
        self.xvar_names = self.results.model.exog_names

        self.residual = np.array(self.results.resid)
        influence = self.results.get_influence()
        self.residual_norm = influence.resid_studentized_internal
        self.leverage = influence.hat_matrix_diag
        self.cooks_distance = influence.cooks_distance[0]
        self.nparams = len(self.results.params)

    def __call__(self, plot_context="seaborn-v0_8-paper"):
        # print(plt.style.available)
        with plt.style.context(plot_context):
            fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
            self.residual_plot(ax=ax[0, 0])
            self.qq_plot(ax=ax[0, 1])
            self.scale_location_plot(ax=ax[1, 0])
            self.leverage_plot(ax=ax[1, 1])
            plt.show()

        self.vif_table()
        return fig, ax

    def residual_plot(self, ax=None):
        """
        Residual vs Fitted Plot

        Graphical tool to identify non-linearity.
        (Roughly) Horizontal red line is an indicator that the residual has a linear pattern
        """
        if ax is None:
            fig, ax = plt.subplots()

        sns.residplot(
            x=self.y_predict,
            y=self.residual,
            lowess=True,
            scatter_kws={"alpha": 0.5},
            line_kws={"color": "red", "lw": 1, "alpha": 0.8},
            ax=ax,
        )

        # annotations
        residual_abs = np.abs(self.residual)
        abs_resid = np.flip(np.sort(residual_abs))
        abs_resid_top_3 = abs_resid[:3]
        for i, _ in enumerate(abs_resid_top_3):
            ax.annotate(i, xy=(self.y_predict[i], self.residual[i]), color="C3")

        ax.set_title("Residuals vs Fitted", fontweight="bold")
        ax.set_xlabel("Fitted values")
        ax.set_ylabel("Residuals")
        return ax

    def qq_plot(self, ax=None):
        """
        Standarized Residual vs Theoretical Quantile plot

        Used to visually check if residuals are normally distributed.
        Points spread along the diagonal line will suggest so.
        """
        if ax is None:
            fig, ax = plt.subplots()

        QQ = ProbPlot(self.residual_norm)
        QQ.qqplot(line="45", alpha=0.5, lw=1, ax=ax)

        # annotations
        abs_norm_resid = np.flip(np.argsort(np.abs(self.residual_norm)), 0)
        abs_norm_resid_top_3 = abs_norm_resid[:3]
        for r, i in enumerate(abs_norm_resid_top_3):
            ax.annotate(
                i,
                xy=(np.flip(QQ.theoretical_quantiles, 0)[r], self.residual_norm[i]),
                ha="right",
                color="C3",
            )

        ax.set_title("Normal Q-Q", fontweight="bold")
        ax.set_xlabel("Theoretical Quantiles")
        ax.set_ylabel("Standardized Residuals")
        return ax

    def scale_location_plot(self, ax=None):
        """
        Sqrt(Standarized Residual) vs Fitted values plot

        Used to check homoscedasticity of the residuals.
        Horizontal line will suggest so.
        """
        if ax is None:
            fig, ax = plt.subplots()

        residual_norm_abs_sqrt = np.sqrt(np.abs(self.residual_norm))

        ax.scatter(self.y_predict, residual_norm_abs_sqrt, alpha=0.5)
        sns.regplot(
            x=self.y_predict,
            y=residual_norm_abs_sqrt,
            scatter=False,
            ci=False,
            lowess=True,
            line_kws={"color": "red", "lw": 1, "alpha": 0.8},
            ax=ax,
        )

        # annotations
        abs_sq_norm_resid = np.flip(np.argsort(residual_norm_abs_sqrt), 0)
        abs_sq_norm_resid_top_3 = abs_sq_norm_resid[:3]
        for i in abs_sq_norm_resid_top_3:
            ax.annotate(
                i, xy=(self.y_predict[i], residual_norm_abs_sqrt[i]), color="C3"
            )
        ax.set_title("Scale-Location", fontweight="bold")
        ax.set_xlabel("Fitted values")
        ax.set_ylabel(r"$\sqrt{|\mathrm{Standardized\ Residuals}|}$")
        return ax

    def leverage_plot(self, ax=None):
        """
        Residual vs Leverage plot

        Points falling outside Cook's distance curves are considered observation that can sway the fit
        aka are influential.
        Good to have none outside the curves.
        """
        if ax is None:
            fig, ax = plt.subplots()

        ax.scatter(self.leverage, self.residual_norm, alpha=0.5)
        sns.regplot(
            x=self.leverage,
            y=self.residual_norm,
            scatter=False,
            ci=False,
            lowess=True,
            line_kws={"color": "red", "lw": 1, "alpha": 0.8},
            ax=ax,
        )

        # annotations
        leverage_top_3 = np.flip(np.argsort(self.cooks_distance), 0)[:3]
        for i in leverage_top_3:
            ax.annotate(i, xy=(self.leverage[i], self.residual_norm[i]), color="C3")

        xtemp, ytemp = self.__cooks_dist_line(0.5)  # 0.5 line
        ax.plot(xtemp, ytemp, label="Cook's distance", lw=1, ls="--", color="red")
        xtemp, ytemp = self.__cooks_dist_line(1)  # 1 line
        ax.plot(xtemp, ytemp, lw=1, ls="--", color="red")

        ax.set_xlim(0, max(self.leverage) + 0.01)
        ax.set_title("Residuals vs Leverage", fontweight="bold")
        ax.set_xlabel("Leverage")
        ax.set_ylabel("Standardized Residuals")
        ax.legend(loc="upper right")
        return ax

    def vif_table(self):
        """
        VIF table

        VIF, the variance inflation factor, is a measure of multicollinearity.
        VIF > 5 for a variable indicates that it is highly collinear with the
        other input variables.
        """
        vif_df = pd.DataFrame()
        vif_df["Features"] = self.xvar_names
        vif_df["VIF Factor"] = [
            variance_inflation_factor(self.xvar, i) for i in range(self.xvar.shape[1])
        ]

        print(vif_df.sort_values("VIF Factor").round(2))

    def __cooks_dist_line(self, factor):
        """
        Helper function for plotting Cook's distance curves
        """
        p = self.nparams
        formula = lambda x: np.sqrt((factor * p * (1 - x)) / x)
        x = np.linspace(0.001, max(self.leverage), 50)
        y = formula(x)
        return x, y


# plot diagnostics for the model
cls = Linear_Reg_Diagnostic(results_c_eur_area_skater)
fig, ax = cls()
# test = results_c_eur_area_skater.outlier_test()
# print('Bad data points (bonf(p) < 0.05):')
# test[test['bonf(p)'] < 0.05]
# outliers = test[test['bonf(p)'] < 0.05].index.values
# outliers_sum=list(outliers)
# all_data.loc[list(outliers)]
### 2.3 Difference in Difference validation

# copy data for the model (DID validation)
data_aveiro_skater_DID = data_aveiro_skater.copy()
data_aveiro_skater_DID.columns
data_aveiro_skater_DID["Year"].unique()
data_aveiro_skater_DID = data_aveiro_skater_DID[data_aveiro_skater_DID["T"] == 0]
data_aveiro_skater_DID["Year"].unique()
# check number of dwellings per cluster
pd.pivot_table(
    data_aveiro_skater_DID,
    values="Log_P_A",
    index=["Year"],
    aggfunc=lambda x: len(x.unique()),
)
# Dummy for year on the DT zone
for i in data_aveiro_skater_DID["Year"].unique():
    data_aveiro_skater_DID["Year" + str(i)] = np.where(
        data_aveiro_skater_DID["Year"] == i, data_aveiro_skater_DID["D"], 0
    )
data_aveiro_skater_DID.columns
data_aveiro_skater_DID.head()
# number of dwellings per cluster
pd.pivot_table(
    data_aveiro_skater_DID,
    values="Log_P_A",
    index=["Zona_SKATER"],
    aggfunc=lambda x: len(x.unique()),
).head(10)
# get dummies for skater zones
data_aveiro_skater_DID = pd.get_dummies(
    data_aveiro_skater_DID, columns=["Zona_SKATER"], drop_first=True, dtype=float
)
data_aveiro_skater_DID.columns
# define dependent variable and independent variables

DID_y_eur_area_skater = data_aveiro_skater_DID["Log_P_A"].astype(float)

DID_X_eur_area_skater = data_aveiro_skater_DID[
    [
        "Year2006.0",
        "Year2007.0",
        "Year2008.0",
        "Year2009.0",
        "MCA_1",
        "MCA_2",
        "MCA_3",
        "MCA_4",
        "Zona_SKATER_1.0",
        "Zona_SKATER_2.0",
        "Zona_SKATER_3.0",
        "Zona_SKATER_4.0",
        "Zona_SKATER_5.0",
        "Zona_SKATER_6.0",
        "Zona_SKATER_7.0",
        "Zona_SKATER_8.0",
        "Tot_AL",
        "IPI",
        "TAA",
    ]
].astype(float)
# run the model
DID_X_eur_area_skater = sm.add_constant(DID_X_eur_area_skater)
model_DID_eur_area_skater = sm.OLS(DID_y_eur_area_skater, DID_X_eur_area_skater)
results_DID_eur_area_skater = model_DID_eur_area_skater.fit()
DID_skater_save = results_DID_eur_area_skater.summary2()
results_DID_eur_area_skater.summary2()
# # export results to csv
# DID_skater_save.tables[0].to_csv('DID_skater_0.csv')
# DID_skater_save.tables[1].to_csv('DID_skater_1.csv')
# plot diagnostics for the model
cls = Linear_Reg_Diagnostic(results_DID_eur_area_skater)
fig, ax = cls()
# test = results_DID_eur_area_skater.outlier_test()
# print('Bad data points (bonf(p) < 0.05):')
# test[test['bonf(p)'] < 0.05]
# outliers = test[test['bonf(p)'] < 0.05].index.values
# outliers_sum=outliers_sum+list(outliers)
# all_data.loc[list(outliers)]
### 2.4 Spatial Linear Regression (focused in Aveiro Center - SKATER Cluster)

data_aveiro_skater_Robust = data_aveiro_skater.copy()
data_aveiro_skater_Robust.columns
data_aveiro_skater_Robust.shape
# Geração da matriz W, a partir do subset, usando a de matriz de contiguidade Queen
w_Queen_skater = weights.contiguity.Queen.from_dataframe(data_aveiro_skater_Robust)
# Standardização das linhas
w_Queen_skater.transform = "R"
# Histograma resultante da metodologia matriz Queen para pesos espaciais
pd.Series(w_Queen_skater.cardinalities).hist()
w_Queen_skater.mean_neighbors
# # image saved - this function is heavy and slows down the notebook
# plot_spatial_weights(w_Queen_skater, data_aveiro_skater_Robust)
ax = data_aveiro_skater_Robust.plot(
    column="Zona_SKATER", categorical=True, legend=True, figsize=(10, 10), cmap="tab20c"
)
cx.add_basemap(
    ax, crs=data_aveiro_skater_Robust.crs, source=cx.providers.OpenStreetMap.Mapnik
)
# apply above list to data
data_aveiro_skater_Robust["D"] = np.where(
    (data_aveiro_skater_Robust["Zona_SKATER"] == 3.0)
    | (data_aveiro_skater_Robust["Zona_SKATER"] == 6.0)
    | (data_aveiro_skater_Robust["Zona_SKATER"] == 7.0),
    1,
    0,
)
data_aveiro_skater_Robust["D"].value_counts()
sum(data_aveiro_skater_Robust["D"].value_counts())
# calculate DT (true when both D and T are equal to 1)
data_aveiro_skater_Robust["DT"] = (
    data_aveiro_skater_Robust["D"] * data_aveiro_skater_Robust["T"]
)
data_aveiro_skater_Robust["DT"].value_counts()
(data_aveiro_skater_Robust["Log_P_A"]).describe()
np.exp(0.367561)
from pysal.explore import esda

moran = esda.moran.Moran(data_aveiro_skater_Robust["Log_P_A"], w_Queen_skater)
moran.I
moran.p_sim
moran_l = esda.moran.Moran_Local(data_aveiro_skater_Robust["Log_P_A"], w_Queen_skater)
from pysal.viz import splot

figura, ax = moran_scatterplot(moran_l, p=0.05)
ax.set_title("Moran Scatterplot")
ax.set_xlabel("mediana_preço_hab")
ax.set_ylabel("Spatial Lag of mediana_preço_hab")
plt.show()
data_aveiro_skater_Robust.columns
data_aveiro_skater_Robust["MCA_1_lag"] = weights.lag_spatial(
    w_Queen_skater, data_aveiro_skater_Robust["MCA_1"]
)
data_aveiro_skater_Robust["MCA_2_lag"] = weights.lag_spatial(
    w_Queen_skater, data_aveiro_skater_Robust["MCA_2"]
)
data_aveiro_skater_Robust["MCA_3_lag"] = weights.lag_spatial(
    w_Queen_skater, data_aveiro_skater_Robust["MCA_3"]
)
data_aveiro_skater_Robust["MCA_4_lag"] = weights.lag_spatial(
    w_Queen_skater, data_aveiro_skater_Robust["MCA_4"]
)
data_aveiro_skater_Robust["Tot_AL_lag"] = weights.lag_spatial(
    w_Queen_skater, data_aveiro_skater_Robust["Tot_AL"]
)
data_aveiro_skater_Robust["Log_P_A_lag"] = weights.lag_spatial(
    w_Queen_skater, data_aveiro_skater_Robust["Log_P_A"]
)
# get dummies for skater zones
data_aveiro_skater_Robust = pd.get_dummies(
    data_aveiro_skater_Robust, columns=["Zona_SKATER"], drop_first=True, dtype=float
)
data_aveiro_skater_Robust.columns
# Criação de dataframe com variável dependente, para uso nos modelos
Dep_Var_SKATER = data_aveiro_skater_Robust["Log_P_A"].astype(float)

# Criação de dataframe com variáveis independente, para uso nos modelos
Ind_Var_SKATER = data_aveiro_skater_Robust[
    [
        "DT",
        "MCA_1",
        "MCA_2",
        "MCA_3",
        "MCA_4",
        "TAA",
        "IPI",
        "Tot_AL",
        "Zona_SKATER_1.0",
        "Zona_SKATER_2.0",
        "Zona_SKATER_3.0",
        "Zona_SKATER_4.0",
        "Zona_SKATER_5.0",
        "Zona_SKATER_6.0",
    ]
].astype(float)

Ind_Var_lag_SKATER_lagX = data_aveiro_skater_Robust[
    [
        "DT",
        "MCA_1",
        "MCA_2",
        "MCA_3",
        "MCA_4",
        "TAA",
        "IPI",
        "Tot_AL",
        "Zona_SKATER_1.0",
        "Zona_SKATER_2.0",
        "Zona_SKATER_3.0",
        "Zona_SKATER_4.0",
        "Zona_SKATER_5.0",
        "Zona_SKATER_6.0",
        "MCA_1_lag",
        "MCA_2_lag",
        "MCA_3_lag",
        "MCA_4_lag",
        "Tot_AL_lag",
    ]
].astype(float)

Ind_Var_lag_SKATER_lagY = data_aveiro_skater_Robust[
    [
        "DT",
        "MCA_1",
        "MCA_2",
        "MCA_3",
        "MCA_4",
        "TAA",
        "IPI",
        "Tot_AL",
        "Zona_SKATER_1.0",
        "Zona_SKATER_2.0",
        "Zona_SKATER_3.0",
        "Zona_SKATER_4.0",
        "Zona_SKATER_5.0",
        "Zona_SKATER_6.0",
        "Log_P_A_lag",
    ]
].astype(float)

Ind_Var_lag_SKATER_lagXY = data_aveiro_skater_Robust[
    [
        "DT",
        "MCA_1",
        "MCA_2",
        "MCA_3",
        "MCA_4",
        "TAA",
        "IPI",
        "Tot_AL",
        "Zona_SKATER_1.0",
        "Zona_SKATER_2.0",
        "Zona_SKATER_3.0",
        "Zona_SKATER_4.0",
        "Zona_SKATER_5.0",
        "Zona_SKATER_6.0",
        "Log_P_A_lag",
        "MCA_1_lag",
        "MCA_2_lag",
        "MCA_3_lag",
        "MCA_4_lag",
        "Tot_AL_lag",
    ]
].astype(float)
M_OLS_skater = spreg.OLS(
    Dep_Var_SKATER.values,  # Dependent variable
    Ind_Var_SKATER.values,  # Independent variable
    name_y="Log_P_A",  # Dependent variable name
    name_x=list(Ind_Var_SKATER.columns),  # Independent variable name
    w=w_Queen_skater,
    spat_diag=True,
    moran=True,
    name_w="w_Queen",
)

print(M_OLS_skater.summary)
data_aveiro_skater_Robust["residuals_OLS"] = M_OLS_skater.u
sns.displot(data_aveiro_skater_Robust["residuals_OLS"], bins=30, kde=True)
plt.title("Distributions of residuals")
# standardisation of the residuals (Z-scores)
data_aveiro_skater_Robust["Z_Score_residuals_OLS"] = stats.zscore(
    data_aveiro_skater_Robust["residuals_OLS"]
)
# distribution of the data against the expected normal distribution.
qqplot(data_aveiro_skater_Robust["Z_Score_residuals_OLS"], line="s")
plt.title("Normal Q-Q plot of residuals")
# Create a Map - Equal Intervals
f, ax = plt.subplots(1, figsize=(8, 12))
ax = data_aveiro_skater_Robust.plot(
    column="Z_Score_residuals_OLS",  # Data to plot
    scheme="EqualInterval",  # Classification scheme
    cmap="bwr",  # Color palette
    edgecolor="k",  # Borderline color
    linewidth=0.1,  # Borderline width
    legend=True,  # Add legend
    legend_kwds={"fmt": "{:.1f}"},  # Remove decimals in legend (for legibility)
    k=10,
    ax=ax,
)

ax.set_title("residuals_OLS")
ax.set_axis_off()
# Create a Map - Equal Intervals
f, ax = plt.subplots(1, figsize=(8, 12))
ax = data_aveiro_skater_Robust.plot(
    column="Z_Score_residuals_OLS",  # Data to plot
    scheme="StdMean",  # Classification scheme
    cmap="bwr",  # Color palette
    edgecolor="k",  # Borderline color
    linewidth=0.1,  # Borderline width
    legend=True,  # Add legend
    legend_kwds={
        "fmt": "{:.1f}",
        "loc": "lower right",
    },  # Remove decimals in legend (for legibility)
    ax=ax,
)

ax.set_title("residuals_OLS")
ax.set_axis_off()
## 3 Linear Regressions (Aveiro) - FR (Freguesias)

### 3.1 Data Preparation for DID Linear Regression

# transform data to geodataframe
data_aveiro_fr = gpd.GeoDataFrame(all_data_fr, geometry="geometry")
data_aveiro_fr.columns
# check number of dwellings per cluster
pd.pivot_table(
    data_aveiro_fr,
    values="Cluster_LP",
    index=["FR11"],
    aggfunc=lambda x: len(x.unique()),
).head(10)
data_aveiro_fr.dtypes
data_aveiro_fr.head()
ax = data_aveiro_fr.plot(
    column="FR11", categorical=True, legend=True, figsize=(10, 10), cmap="tab20"
)
cx.add_basemap(ax, crs=data_aveiro_fr.crs, source=cx.providers.OpenStreetMap.Mapnik)
# apply above list to data
data_aveiro_fr["D"] = np.where((data_aveiro_fr["FR11"] == 12), 1, 0)
data_aveiro_fr["D"].value_counts()
# check areas defined as control and intervention areas (D=0 and D=1)
ax = data_aveiro_fr.plot(
    column=data_aveiro_fr["D"],
    categorical=True,
    legend=True,
    figsize=(10, 10),
    cmap="tab20",
)
cx.add_basemap(ax, crs=data_aveiro_fr.crs, source=cx.providers.OpenStreetMap.Mapnik)
### 3.2 Linear Regression (focused in Aveiro Center - Freguesias)

# calculate DT (true when both D and T are equal to 1)
data_aveiro_fr["DT"] = data_aveiro_fr["D"] * data_aveiro_fr["T"]
data_aveiro_fr["DT"].value_counts()
data_aveiro_fr["D"].value_counts()
# get dummies for skater zones
data_aveiro_fr_ols = pd.get_dummies(
    data_aveiro_fr, columns=["FR11"], drop_first=True, dtype=float
)
data_aveiro_fr_ols.columns
# define X and y
c_y_eur_area_fr = data_aveiro_fr_ols["Log_P_A"].astype(float)

c_X_eur_area_fr = data_aveiro_fr_ols[
    [
        "DT",
        "MCA_1",
        "MCA_2",
        "MCA_3",
        "MCA_4",
        "IPI",
        "Tot_AL",
        "TAA",
        "FR11_5",
        "FR11_6",
        "FR11_10",
        "FR11_12",
        "FR11_13",
    ]
].astype(float)

# removido IPI - VIF elevado
# linear model for casasapo with log of price per square meter as dependent variable
c_X_eur_area_fr = sm.add_constant(c_X_eur_area_fr)
model_c_eur_area_fr = sm.OLS(c_y_eur_area_fr, c_X_eur_area_fr)
results_c_eur_area_fr = model_c_eur_area_fr.fit()
fr_save = results_c_eur_area_fr.summary2()
results_c_eur_area_fr.summary2()
# # export results to csv
# fr_save.tables[0].to_csv('fr_0.csv')
# fr_save.tables[1].to_csv('fr_1.csv')
# plot diagnostics for the model
cls = Linear_Reg_Diagnostic(results_c_eur_area_fr)
fig, ax = cls()
# test = results_c_eur_area_fr.outlier_test()
# print('Bad data points (bonf(p) < 0.05):')
# test[test['bonf(p)'] < 0.05]
# outliers = test[test['bonf(p)'] < 0.05].index.values
# outliers_sum=outliers_sum+list(outliers)
# all_data.loc[list(outliers)]
### 3.3 Difference in Difference validation

# copy data for the model (DID validation)
data_aveiro_fr_DID = data_aveiro_fr.copy()
data_aveiro_fr_DID.columns
data_aveiro_fr_DID["Year"].unique()
data_aveiro_fr_DID = data_aveiro_fr_DID[data_aveiro_fr_DID["T"] == 0]
data_aveiro_fr_DID["Year"].unique()
for i in data_aveiro_fr_DID["Year"].unique():
    data_aveiro_fr_DID["Year" + str(i)] = np.where(
        data_aveiro_fr_DID["Year"] == i, data_aveiro_fr_DID["D"], 0
    )
data_aveiro_fr_DID.head()
# number of dwellings per cluster
pd.pivot_table(
    data_aveiro_fr_DID,
    values="Log_P_A",
    index=["FR11"],
    aggfunc=lambda x: len(x.unique()),
).head(10)
# não incluir cluster 102 e 106, por escassez de dados
data_aveiro_fr_DID.columns
# get dummies for FR11 zones
data_aveiro_fr_DID = pd.get_dummies(
    data_aveiro_fr_DID, columns=["FR11"], drop_first=True, dtype=float
)
data_aveiro_fr_DID.columns
# define dependent variable and independent variables
DID_y_eur_area_fr = data_aveiro_fr_DID["Log_P_A"].astype(float)

DID_X_eur_area_fr = data_aveiro_fr_DID[
    [
        "Year2006.0",
        "Year2007.0",
        "Year2008.0",
        "Year2009.0",
        "MCA_1",
        "MCA_2",
        "MCA_3",
        "MCA_4",
        "Tot_AL",
        "IPI",
        "TAA",
        "FR11_5",
        "FR11_6",
        "FR11_10",
        "FR11_12",
        "FR11_13",
    ]
].astype(float)
#
# run the model
DID_X_eur_area_fr = sm.add_constant(DID_X_eur_area_fr)
model_DID_eur_area_fr = sm.OLS(DID_y_eur_area_fr, DID_X_eur_area_fr)
results_DID_eur_area_fr = model_DID_eur_area_fr.fit()
DID_fr_save = results_DID_eur_area_fr.summary2()
results_DID_eur_area_fr.summary2()
# # export results to csv
# DID_fr_save.tables[0].to_csv('DID_fr_0.csv')
# DID_fr_save.tables[1].to_csv('DID_fr_1.csv')

# plot diagnostics for the model
cls = Linear_Reg_Diagnostic(results_DID_eur_area_fr)
fig, ax = cls()
# test = results_DID_eur_area_fr.outlier_test()
# print('Bad data points (bonf(p) < 0.05):')
# test[test['bonf(p)'] < 0.05]
# outliers = test[test['bonf(p)'] < 0.05].index.values
# outliers_sum=outliers_sum+list(outliers)
# all_data.loc[list(outliers)]
### 3.4 Spatial Linear Regression (focused in Aveiro Center - FR11)

data_aveiro_fr_Robust = data_aveiro_fr.copy()
data_aveiro_fr_Robust.columns
data_aveiro_fr_Robust.shape
# Geração da matriz W, a partir do subset, usando a de matriz de contiguidade Queen
w_Queen_fr = weights.contiguity.Queen.from_dataframe(data_aveiro_fr_Robust)
# Standardização das linhas
w_Queen_fr.transform = "R"
# Histograma resultante da metodologia matriz Queen para pesos espaciais
pd.Series(w_Queen_fr.cardinalities).hist()
w_Queen_fr.mean_neighbors
# # image saved - this function is heavy and slows down the notebook
# plot_spatial_weights(w_Queen_fr, data_aveiro_fr_Robust)
ax = data_aveiro_fr_Robust.plot(
    column="FR11", categorical=True, legend=True, figsize=(10, 10), cmap="tab20c"
)
cx.add_basemap(
    ax, crs=data_aveiro_fr_Robust.crs, source=cx.providers.OpenStreetMap.Mapnik
)
# apply above list to data
data_aveiro_fr_Robust["D"] = np.where((data_aveiro_fr_Robust["FR11"] == 12), 1, 0)
data_aveiro_fr_Robust["D"].value_counts()
sum(data_aveiro_fr_Robust["D"].value_counts())
# calculate DT (true when both D and T are equal to 1)
data_aveiro_fr_Robust["DT"] = data_aveiro_fr_Robust["D"] * data_aveiro_fr_Robust["T"]
data_aveiro_fr_Robust["DT"].value_counts()
data_aveiro_fr_Robust.columns
data_aveiro_fr_Robust["MCA_1_lag"] = weights.lag_spatial(
    w_Queen_fr, data_aveiro_fr_Robust["MCA_1"]
)
data_aveiro_fr_Robust["MCA_2_lag"] = weights.lag_spatial(
    w_Queen_fr, data_aveiro_fr_Robust["MCA_2"]
)
data_aveiro_fr_Robust["MCA_3_lag"] = weights.lag_spatial(
    w_Queen_fr, data_aveiro_fr_Robust["MCA_3"]
)
data_aveiro_fr_Robust["MCA_4_lag"] = weights.lag_spatial(
    w_Queen_fr, data_aveiro_fr_Robust["MCA_4"]
)
data_aveiro_fr_Robust["Tot_AL_lag"] = weights.lag_spatial(
    w_Queen_fr, data_aveiro_fr_Robust["Tot_AL"]
)
data_aveiro_fr_Robust["Log_P_A_lag"] = weights.lag_spatial(
    w_Queen_skater, data_aveiro_fr_Robust["Log_P_A"]
)
# get dummies for skater zones
data_aveiro_fr_Robust = pd.get_dummies(
    data_aveiro_fr_Robust, columns=["FR11"], drop_first=True, dtype=float
)
data_aveiro_fr_Robust.columns
# Criação de dataframe com variável dependente, para uso nos modelos
Dep_Var_FR = data_aveiro_fr_Robust["Log_P_A"].astype(float)

# Criação de dataframe com variáveis independente, para uso nos modelos
Ind_Var_FR = data_aveiro_fr_Robust[
    [
        "DT",
        "MCA_1",
        "MCA_2",
        "MCA_3",
        "MCA_4",
        "TAA",
        "IPI",
        "Tot_AL",
        "FR11_5",
        "FR11_6",
        "FR11_10",
        "FR11_12",
        "FR11_13",
    ]
].astype(float)

Ind_Var_FR_lagX = data_aveiro_fr_Robust[
    [
        "DT",
        "MCA_1",
        "MCA_2",
        "MCA_3",
        "MCA_4",
        "TAA",
        "IPI",
        "Tot_AL",
        "FR11_5",
        "FR11_6",
        "FR11_10",
        "FR11_12",
        "FR11_13",
        "MCA_1_lag",
        "MCA_2_lag",
        "MCA_3_lag",
        "MCA_4_lag",
        "Tot_AL_lag",
    ]
].astype(float)

Ind_Var_FR_lagY = data_aveiro_fr_Robust[
    [
        "DT",
        "MCA_1",
        "MCA_2",
        "MCA_3",
        "MCA_4",
        "TAA",
        "IPI",
        "Tot_AL",
        "FR11_5",
        "FR11_6",
        "FR11_10",
        "FR11_12",
        "FR11_13",
        "Log_P_A_lag",
    ]
].astype(float)

Ind_Var_FR_lagXY = data_aveiro_fr_Robust[
    [
        "DT",
        "MCA_1",
        "MCA_2",
        "MCA_3",
        "MCA_4",
        "TAA",
        "IPI",
        "Tot_AL",
        "FR11_5",
        "FR11_6",
        "FR11_10",
        "FR11_12",
        "FR11_13",
        "Log_P_A_lag",
        "MCA_1_lag",
        "MCA_2_lag",
        "MCA_3_lag",
        "MCA_4_lag",
        "Tot_AL_lag",
    ]
].astype(float)
M_OLS_FR = spreg.OLS(
    Dep_Var_FR.values,  # Dependent variable
    Ind_Var_FR.values,  # Independent variable
    name_y="Log_P_A",  # Dependent variable name
    name_x=list(Ind_Var_FR.columns),  # Independent variable name
    w=w_Queen_fr,
    spat_diag=True,
    moran=True,
    name_w="w_Queen",
)

print(M_OLS_FR.summary)
data_aveiro_fr_Robust["residuals_OLS"] = M_OLS_FR.u
sns.displot(data_aveiro_fr_Robust["residuals_OLS"], bins=30, kde=True)
plt.title("Distributions of residuals")
# standardisation of the residuals (Z-scores)
data_aveiro_fr_Robust["Z_Score_residuals_OLS"] = stats.zscore(
    data_aveiro_fr_Robust["residuals_OLS"]
)
# distribution of the data against the expected normal distribution.
qqplot(data_aveiro_fr_Robust["Z_Score_residuals_OLS"], line="s")
plt.title("Normal Q-Q plot of residuals")
# Create a Map - Equal Intervals
f, ax = plt.subplots(1, figsize=(8, 12))
ax = data_aveiro_fr_Robust.plot(
    column="Z_Score_residuals_OLS",  # Data to plot
    scheme="EqualInterval",  # Classification scheme
    cmap="bwr",  # Color palette
    edgecolor="k",  # Borderline color
    linewidth=0.1,  # Borderline width
    legend=True,  # Add legend
    legend_kwds={"fmt": "{:.1f}"},  # Remove decimals in legend (for legibility)
    k=10,
    ax=ax,
)

ax.set_title("residuals_OLS")
ax.set_axis_off()
# Create a Map - Equal Intervals
f, ax = plt.subplots(1, figsize=(8, 12))
ax = data_aveiro_fr_Robust.plot(
    column="Z_Score_residuals_OLS",  # Data to plot
    scheme="StdMean",  # Classification scheme
    cmap="bwr",  # Color palette
    edgecolor="k",  # Borderline color
    linewidth=0.1,  # Borderline width
    legend=True,  # Add legend
    legend_kwds={
        "fmt": "{:.1f}",
        "loc": "lower right",
    },  # Remove decimals in legend (for legibility)
    ax=ax,
)

ax.set_title("residuals_OLS")
ax.set_axis_off()
## 4 Linear Regressions (Aveiro) - LP

### 4.1 Data Preparation for DID Linear Regression

# transform data to geodataframe
data_aveiro_LP = gpd.GeoDataFrame(all_data_LP, geometry="geometry")
data_aveiro_LP
data_aveiro_LP.columns
# view Zona_SKATER clusters
ax = data_aveiro_LP.plot(
    figsize=(10, 20),
    column=data_aveiro_LP["Cluster_LP"],
    categorical=True,
    edgecolor="w",
    legend=False,
    linewidth=0.2,
    cmap="tab20",
)
cx.add_basemap(ax, crs=data_aveiro_LP.crs, source=cx.providers.OpenStreetMap.Mapnik)
# Area de Intervenção, a ser considerada D = 1
lista_D1 = [
    19,
    22,
    31,
    40,
    41,
    42,
    80,
    102,
    105,
    107,
    109,
    110,
    156,
    159,
    160,
    161,
    165,
    166,
]
# lista de zonas na fronteira D1/D0:
# 26, 57, 101, 103, 104, 111, 155, 161, 165
data_aveiro_LP["D"] = np.where(data_aveiro_LP["Cluster_LP"].isin(lista_D1), 1, 0)
# view Zona_SKATER clusters
ax = data_aveiro_LP.plot(
    figsize=(10, 10),
    column=data_aveiro_LP["D"],
    linewidth=0.2,
    categorical=True,
    legend=True,
    cmap="tab20",
)
cx.add_basemap(ax, crs=data_aveiro_LP.crs, source=cx.providers.OpenStreetMap.Mapnik)
data_aveiro_LP.head()
# check no. of dwellings per cluster in Aveiro Center
pd.pivot_table(
    data_aveiro_LP,
    values="Log_P_A",
    index=["Cluster_LP"],
    aggfunc=lambda x: len(x.unique()),
).head(10)
# não incluir cluster 80, 102 e 106, por escassez de dados
data_aveiro_LP.to_pickle("Data/data_aveiro_LP.piclo")
data_aveiro_LP["D"].value_counts()
sum(data_aveiro_LP["D"].value_counts())
# calculate DT (true when both D and T are equal to 1)
data_aveiro_LP["DT"] = data_aveiro_LP["D"] * data_aveiro_LP["T"]
data_aveiro_LP["DT"].value_counts()
# check log_P_A distribution in the territory
ax = data_aveiro_LP.plot(
    column=data_aveiro_LP["Log_P_A"], legend=True, figsize=(10, 10), cmap="viridis"
)
cx.add_basemap(ax, crs=data_aveiro_LP.crs, source=cx.providers.OpenStreetMap.Mapnik)
### 4.2 Linear Regression (focused in Aveiro Center - LP)

# check number of dwellings per cluster
pd.pivot_table(
    data_aveiro_LP,
    values="Log_P_A",
    index=["Cluster_LP"],
    aggfunc=lambda x: len(x.unique()),
).head(10)
# check number of dwellings per cluster
pd.pivot_table(
    data_aveiro_LP, values="Cluster_LP", index=["D"], aggfunc=lambda x: len(x.unique())
)
data_aveiro_LP.columns
# get dummies for LP zones
data_aveiro_LP_ols = pd.get_dummies(
    data_aveiro_LP, columns=["Cluster_LP"], drop_first=True, dtype=float
)
data_aveiro_LP_ols.columns
# define X and y
c_y_eur_area_LP = data_aveiro_LP_ols["Log_P_A"].astype(float)

c_X_eur_area_LP = data_aveiro_LP_ols[
    [
        "DT",
        "MCA_1",
        "MCA_2",
        "MCA_3",
        "MCA_4",
        "IPI",
        "Tot_AL",
        "TAA",
        "Cluster_LP_5.0",
        "Cluster_LP_13.0",
        "Cluster_LP_14.0",
        "Cluster_LP_18.0",
        "Cluster_LP_19.0",
        "Cluster_LP_20.0",
        "Cluster_LP_21.0",
        "Cluster_LP_22.0",
        "Cluster_LP_24.0",
        "Cluster_LP_25.0",
        "Cluster_LP_26.0",
        "Cluster_LP_27.0",
        "Cluster_LP_28.0",
        "Cluster_LP_29.0",
        "Cluster_LP_30.0",
        "Cluster_LP_31.0",
        "Cluster_LP_37.0",
        "Cluster_LP_38.0",
        "Cluster_LP_39.0",
        "Cluster_LP_40.0",
        "Cluster_LP_41.0",
        "Cluster_LP_42.0",
        "Cluster_LP_43.0",
        "Cluster_LP_45.0",
        "Cluster_LP_57.0",
        "Cluster_LP_58.0",
        "Cluster_LP_59.0",
        "Cluster_LP_60.0",
        "Cluster_LP_63.0",
        "Cluster_LP_64.0",
        "Cluster_LP_65.0",
        "Cluster_LP_66.0",
        "Cluster_LP_67.0",
        "Cluster_LP_80.0",
        "Cluster_LP_82.0",
        "Cluster_LP_85.0",
        "Cluster_LP_89.0",
        "Cluster_LP_90.0",
        "Cluster_LP_91.0",
        "Cluster_LP_100.0",
        "Cluster_LP_101.0",
        "Cluster_LP_102.0",
        "Cluster_LP_103.0",
        "Cluster_LP_104.0",
        "Cluster_LP_105.0",
        "Cluster_LP_107.0",
        "Cluster_LP_108.0",
        "Cluster_LP_109.0",
        "Cluster_LP_110.0",
        "Cluster_LP_111.0",
        "Cluster_LP_124.0",
        "Cluster_LP_125.0",
        "Cluster_LP_129.0",
        "Cluster_LP_150.0",
        "Cluster_LP_151.0",
        "Cluster_LP_152.0",
        "Cluster_LP_153.0",
        "Cluster_LP_154.0",
        "Cluster_LP_155.0",
        "Cluster_LP_156.0",
        "Cluster_LP_157.0",
        "Cluster_LP_158.0",
        "Cluster_LP_159.0",
        "Cluster_LP_160.0",
        "Cluster_LP_161.0",
        "Cluster_LP_162.0",
        "Cluster_LP_163.0",
        "Cluster_LP_164.0",
        "Cluster_LP_165.0",
        "Cluster_LP_166.0",
    ]
].astype(float)
# linear model for casasapo with log of price per square meter as dependent variable
c_X_eur_area_LP = sm.add_constant(c_X_eur_area_LP)
model_c_eur_area_LP = sm.OLS(c_y_eur_area_LP, c_X_eur_area_LP)
results_c_eur_area_LP = model_c_eur_area_LP.fit()
lp_save = results_c_eur_area_LP.summary2()
results_c_eur_area_LP.summary2()
# # export results to csv
# lp_save.tables[0].to_csv('lp_0.csv')
# lp_save.tables[1].to_csv('lp_1.csv')

# plot diagnostics for the model
cls = Linear_Reg_Diagnostic(results_c_eur_area_LP)
fig, ax = cls()
# test = results_c_eur_area_LP.outlier_test()
# print('Bad data points (bonf(p) < 0.05):')
# test[test['bonf(p)'] < 0.05]
# outliers = test[test['bonf(p)'] < 0.05].index.values
# outliers_sum=outliers_sum+list(outliers)
# all_data.loc[list(outliers)]
### 4.3 Difference in Difference validation

# copy data for the model (DID validation)
data_aveiro_LP_DID = data_aveiro_LP.copy()
data_aveiro_LP_DID.columns
data_aveiro_LP_DID["Year"].unique()
data_aveiro_LP_DID = data_aveiro_LP_DID[data_aveiro_LP_DID["T"] == 0]
data_aveiro_LP_DID["Year"].unique()
for i in data_aveiro_LP_DID["Year"].unique():
    data_aveiro_LP_DID["Year" + str(i)] = np.where(
        data_aveiro_LP_DID["Year"] == i, data_aveiro_LP_DID["D"], 0
    )
data_aveiro_LP_DID.head()
# number of dwellings per cluster
pd.pivot_table(
    data_aveiro_LP_DID,
    values="Log_P_A",
    index=["Cluster_LP"],
    aggfunc=lambda x: len(x.unique()),
).head(10)
# não incluir cluster 102 e 106, por escassez de dados
data_aveiro_LP_DID.columns
# get dummies for LP zones
data_aveiro_LP_DID = pd.get_dummies(
    data_aveiro_LP_DID, columns=["Cluster_LP"], drop_first=True, dtype=float
)
data_aveiro_LP_DID.columns
# define dependent variable and independent variables
DID_y_eur_area_LP = data_aveiro_LP_DID["Log_P_A"].astype(float)

DID_X_eur_area_LP = data_aveiro_LP_DID[
    [
        "Year2006.0",
        "Year2007.0",
        "Year2008.0",
        "Year2009.0",
        "MCA_1",
        "MCA_2",
        "MCA_3",
        "MCA_4",
        "IPI",
        "Tot_AL",
        "TAA",
        "Cluster_LP_5.0",
        "Cluster_LP_13.0",
        "Cluster_LP_14.0",
        "Cluster_LP_18.0",
        "Cluster_LP_19.0",
        "Cluster_LP_20.0",
        "Cluster_LP_21.0",
        "Cluster_LP_22.0",
        "Cluster_LP_24.0",
        "Cluster_LP_25.0",
        "Cluster_LP_26.0",
        "Cluster_LP_27.0",
        "Cluster_LP_28.0",
        "Cluster_LP_29.0",
        "Cluster_LP_30.0",
        "Cluster_LP_31.0",
        "Cluster_LP_37.0",
        "Cluster_LP_38.0",
        "Cluster_LP_39.0",
        "Cluster_LP_40.0",
        "Cluster_LP_41.0",
        "Cluster_LP_42.0",
        "Cluster_LP_43.0",
        "Cluster_LP_45.0",
        "Cluster_LP_58.0",
        "Cluster_LP_59.0",
        "Cluster_LP_60.0",
        "Cluster_LP_63.0",
        "Cluster_LP_64.0",
        "Cluster_LP_65.0",
        "Cluster_LP_67.0",
        "Cluster_LP_82.0",
        "Cluster_LP_85.0",
        "Cluster_LP_89.0",
        "Cluster_LP_91.0",
        "Cluster_LP_100.0",
        "Cluster_LP_101.0",
        "Cluster_LP_104.0",
        "Cluster_LP_105.0",
        "Cluster_LP_109.0",
        "Cluster_LP_111.0",
        "Cluster_LP_124.0",
        "Cluster_LP_150.0",
        "Cluster_LP_151.0",
        "Cluster_LP_152.0",
        "Cluster_LP_153.0",
        "Cluster_LP_154.0",
        "Cluster_LP_155.0",
        "Cluster_LP_156.0",
        "Cluster_LP_157.0",
        "Cluster_LP_158.0",
        "Cluster_LP_159.0",
        "Cluster_LP_160.0",
        "Cluster_LP_161.0",
        "Cluster_LP_162.0",
        "Cluster_LP_163.0",
        "Cluster_LP_164.0",
        "Cluster_LP_166.0",
    ]
].astype(float)
# run the model
DID_X_eur_area_LP = sm.add_constant(DID_X_eur_area_LP)
model_DID_eur_area_LP = sm.OLS(DID_y_eur_area_LP, DID_X_eur_area_LP)
results_DID_eur_area_LP = model_DID_eur_area_LP.fit()
DID_lp_save = results_DID_eur_area_LP.summary2()
results_DID_eur_area_LP.summary2()
# # export results to csv
# DID_lp_save.tables[0].to_csv('DID_lp_0.csv')
# DID_lp_save.tables[1].to_csv('DID_lp_1.csv')

# plot diagnostics for the model
cls = Linear_Reg_Diagnostic(results_DID_eur_area_LP)
fig, ax = cls()
# test = results_DID_eur_area_LP.outlier_test()
# print('Bad data points (bonf(p) < 0.05):')
# test[test['bonf(p)'] < 0.05]
# outliers = test[test['bonf(p)'] < 0.05].index.values
# outliers_sum=outliers_sum+list(outliers)
# all_data.loc[list(outliers)]
### 4.4 Spatial Linear Regression (focused in Aveiro Center - LP)

data_aveiro_LP_Robust = data_aveiro_LP.copy()
data_aveiro_LP_Robust.columns
data_aveiro_LP_Robust.shape
# Geração da matriz W, a partir do subset, usando a de matriz de contiguidade Queen
w_Queen_LP = weights.contiguity.Queen.from_dataframe(data_aveiro_LP_Robust)
# Standardização das linhas
w_Queen_LP.transform = "R"
# Histograma resultante da metodologia matriz Queen para pesos espaciais
pd.Series(w_Queen_LP.cardinalities).hist()
w_Queen_LP.mean_neighbors
# image saved - this function is heavy and slows down the notebook
# plot_spatial_weights(w_Queen_LP, data_aveiro_LP_Robust)
ax = data_aveiro_LP_Robust.plot(
    column="Cluster_LP", categorical=True, legend=True, figsize=(10, 20), cmap="tab20c"
)
cx.add_basemap(
    ax, crs=data_aveiro_LP_Robust.crs, source=cx.providers.OpenStreetMap.Mapnik
)
data_aveiro_LP_Robust["D"] = np.where(
    data_aveiro_LP_Robust["Cluster_LP"].isin(lista_D1), 1, 0
)
data_aveiro_LP_Robust["D"].value_counts()
sum(data_aveiro_LP_Robust["D"].value_counts())
# calculate DT (true when both D and T are equal to 1)
data_aveiro_LP_Robust["DT"] = data_aveiro_LP_Robust["D"] * data_aveiro_LP_Robust["T"]
data_aveiro_LP_Robust["DT"].value_counts()
sum(data_aveiro_LP_Robust["DT"].value_counts())
data_aveiro_LP_Robust.columns
# get dummies for skater zones
data_aveiro_LP_Robust = pd.get_dummies(
    data_aveiro_LP_Robust, columns=["Cluster_LP"], drop_first=True, dtype=float
)
data_aveiro_LP_Robust.columns
data_aveiro_LP_Robust["MCA_1_lag"] = weights.lag_spatial(
    w_Queen_LP, data_aveiro_LP_Robust["MCA_1"]
)

data_aveiro_LP_Robust["MCA_2_lag"] = weights.lag_spatial(
    w_Queen_LP, data_aveiro_LP_Robust["MCA_2"]
)

data_aveiro_LP_Robust["MCA_3_lag"] = weights.lag_spatial(
    w_Queen_LP, data_aveiro_LP_Robust["MCA_3"]
)

data_aveiro_LP_Robust["MCA_4_lag"] = weights.lag_spatial(
    w_Queen_LP, data_aveiro_LP_Robust["MCA_4"]
)

data_aveiro_LP_Robust["Tot_AL_lag"] = weights.lag_spatial(
    w_Queen_LP, data_aveiro_LP_Robust["Tot_AL"]
)

data_aveiro_LP_Robust["Log_P_A_lag"] = weights.lag_spatial(
    w_Queen_LP, data_aveiro_LP_Robust["Log_P_A"]
)
data_aveiro_LP_Robust.columns
# Criação de dataframe com variável dependente, para uso nos modelos
Dep_Var_LP = data_aveiro_LP_Robust["Log_P_A"].astype(float)

# Criação de dataframe com variáveis independente, para uso nos modelos
Ind_Var_LP = data_aveiro_LP_Robust[
    [
        "DT",
        "MCA_1",
        "MCA_2",
        "MCA_3",
        "MCA_4",
        "TAA",
        "IPI",
        "Tot_AL",
        "Cluster_LP_5.0",
        "Cluster_LP_13.0",
        "Cluster_LP_14.0",
        "Cluster_LP_18.0",
        "Cluster_LP_19.0",
        "Cluster_LP_20.0",
        "Cluster_LP_21.0",
        "Cluster_LP_22.0",
        "Cluster_LP_24.0",
        "Cluster_LP_25.0",
        "Cluster_LP_26.0",
        "Cluster_LP_27.0",
        "Cluster_LP_28.0",
        "Cluster_LP_29.0",
        "Cluster_LP_30.0",
        "Cluster_LP_31.0",
        "Cluster_LP_37.0",
        "Cluster_LP_38.0",
        "Cluster_LP_39.0",
        "Cluster_LP_40.0",
        "Cluster_LP_41.0",
        "Cluster_LP_42.0",
        "Cluster_LP_43.0",
        "Cluster_LP_45.0",
        "Cluster_LP_57.0",
        "Cluster_LP_58.0",
        "Cluster_LP_59.0",
        "Cluster_LP_60.0",
        "Cluster_LP_63.0",
        "Cluster_LP_64.0",
        "Cluster_LP_65.0",
        "Cluster_LP_66.0",
        "Cluster_LP_67.0",
        "Cluster_LP_80.0",
        "Cluster_LP_82.0",
        "Cluster_LP_85.0",
        "Cluster_LP_89.0",
        "Cluster_LP_90.0",
        "Cluster_LP_91.0",
        "Cluster_LP_100.0",
        "Cluster_LP_101.0",
        "Cluster_LP_102.0",
        "Cluster_LP_103.0",
        "Cluster_LP_104.0",
        "Cluster_LP_105.0",
        "Cluster_LP_107.0",
        "Cluster_LP_108.0",
        "Cluster_LP_109.0",
        "Cluster_LP_110.0",
        "Cluster_LP_111.0",
        "Cluster_LP_124.0",
        "Cluster_LP_125.0",
        "Cluster_LP_129.0",
        "Cluster_LP_150.0",
        "Cluster_LP_151.0",
        "Cluster_LP_152.0",
        "Cluster_LP_154.0",
        "Cluster_LP_155.0",
        "Cluster_LP_156.0",
        "Cluster_LP_157.0",
        "Cluster_LP_158.0",
        "Cluster_LP_159.0",
        "Cluster_LP_160.0",
        "Cluster_LP_161.0",
        "Cluster_LP_163.0",
        "Cluster_LP_164.0",
        "Cluster_LP_165.0",
        "Cluster_LP_166.0",
    ]
].astype(float)

Ind_Var_LP_lagX = data_aveiro_LP_Robust[
    [
        "DT",
        "MCA_1",
        "MCA_2",
        "MCA_3",
        "MCA_4",
        "TAA",
        "IPI",
        "Tot_AL",
        "Cluster_LP_5.0",
        "Cluster_LP_13.0",
        "Cluster_LP_14.0",
        "Cluster_LP_18.0",
        "Cluster_LP_19.0",
        "Cluster_LP_20.0",
        "Cluster_LP_21.0",
        "Cluster_LP_22.0",
        "Cluster_LP_24.0",
        "Cluster_LP_25.0",
        "Cluster_LP_26.0",
        "Cluster_LP_27.0",
        "Cluster_LP_28.0",
        "Cluster_LP_29.0",
        "Cluster_LP_30.0",
        "Cluster_LP_31.0",
        "Cluster_LP_37.0",
        "Cluster_LP_38.0",
        "Cluster_LP_39.0",
        "Cluster_LP_40.0",
        "Cluster_LP_41.0",
        "Cluster_LP_42.0",
        "Cluster_LP_43.0",
        "Cluster_LP_45.0",
        "Cluster_LP_57.0",
        "Cluster_LP_58.0",
        "Cluster_LP_59.0",
        "Cluster_LP_60.0",
        "Cluster_LP_63.0",
        "Cluster_LP_64.0",
        "Cluster_LP_65.0",
        "Cluster_LP_66.0",
        "Cluster_LP_67.0",
        "Cluster_LP_80.0",
        "Cluster_LP_82.0",
        "Cluster_LP_85.0",
        "Cluster_LP_89.0",
        "Cluster_LP_90.0",
        "Cluster_LP_91.0",
        "Cluster_LP_100.0",
        "Cluster_LP_101.0",
        "Cluster_LP_102.0",
        "Cluster_LP_103.0",
        "Cluster_LP_104.0",
        "Cluster_LP_105.0",
        "Cluster_LP_107.0",
        "Cluster_LP_108.0",
        "Cluster_LP_109.0",
        "Cluster_LP_110.0",
        "Cluster_LP_111.0",
        "Cluster_LP_124.0",
        "Cluster_LP_125.0",
        "Cluster_LP_129.0",
        "Cluster_LP_150.0",
        "Cluster_LP_151.0",
        "Cluster_LP_152.0",
        "Cluster_LP_154.0",
        "Cluster_LP_155.0",
        "Cluster_LP_156.0",
        "Cluster_LP_157.0",
        "Cluster_LP_158.0",
        "Cluster_LP_159.0",
        "Cluster_LP_160.0",
        "Cluster_LP_161.0",
        "Cluster_LP_163.0",
        "Cluster_LP_164.0",
        "Cluster_LP_165.0",
        "Cluster_LP_166.0",
        "MCA_1_lag",
        "MCA_2_lag",
        "MCA_3_lag",
        "MCA_4_lag",
        "Tot_AL_lag",
    ]
].astype(float)

Ind_Var_LP_lagY = data_aveiro_LP_Robust[
    [
        "DT",
        "MCA_1",
        "MCA_2",
        "MCA_3",
        "MCA_4",
        "TAA",
        "IPI",
        "Tot_AL",
        "Cluster_LP_5.0",
        "Cluster_LP_13.0",
        "Cluster_LP_14.0",
        "Cluster_LP_18.0",
        "Cluster_LP_19.0",
        "Cluster_LP_20.0",
        "Cluster_LP_21.0",
        "Cluster_LP_22.0",
        "Cluster_LP_24.0",
        "Cluster_LP_25.0",
        "Cluster_LP_26.0",
        "Cluster_LP_27.0",
        "Cluster_LP_28.0",
        "Cluster_LP_29.0",
        "Cluster_LP_30.0",
        "Cluster_LP_31.0",
        "Cluster_LP_37.0",
        "Cluster_LP_38.0",
        "Cluster_LP_39.0",
        "Cluster_LP_40.0",
        "Cluster_LP_41.0",
        "Cluster_LP_42.0",
        "Cluster_LP_43.0",
        "Cluster_LP_45.0",
        "Cluster_LP_57.0",
        "Cluster_LP_58.0",
        "Cluster_LP_59.0",
        "Cluster_LP_60.0",
        "Cluster_LP_63.0",
        "Cluster_LP_64.0",
        "Cluster_LP_65.0",
        "Cluster_LP_66.0",
        "Cluster_LP_67.0",
        "Cluster_LP_80.0",
        "Cluster_LP_82.0",
        "Cluster_LP_85.0",
        "Cluster_LP_89.0",
        "Cluster_LP_90.0",
        "Cluster_LP_91.0",
        "Cluster_LP_100.0",
        "Cluster_LP_101.0",
        "Cluster_LP_102.0",
        "Cluster_LP_103.0",
        "Cluster_LP_104.0",
        "Cluster_LP_105.0",
        "Cluster_LP_107.0",
        "Cluster_LP_108.0",
        "Cluster_LP_109.0",
        "Cluster_LP_110.0",
        "Cluster_LP_111.0",
        "Cluster_LP_124.0",
        "Cluster_LP_125.0",
        "Cluster_LP_129.0",
        "Cluster_LP_150.0",
        "Cluster_LP_151.0",
        "Cluster_LP_152.0",
        "Cluster_LP_154.0",
        "Cluster_LP_155.0",
        "Cluster_LP_156.0",
        "Cluster_LP_157.0",
        "Cluster_LP_158.0",
        "Cluster_LP_159.0",
        "Cluster_LP_160.0",
        "Cluster_LP_161.0",
        "Cluster_LP_163.0",
        "Cluster_LP_164.0",
        "Cluster_LP_165.0",
        "Cluster_LP_166.0",
        "Log_P_A_lag",
    ]
].astype(float)

Ind_Var_LP_lagXY = data_aveiro_LP_Robust[
    [
        "DT",
        "MCA_1",
        "MCA_2",
        "MCA_3",
        "MCA_4",
        "TAA",
        "IPI",
        "Tot_AL",
        "Cluster_LP_5.0",
        "Cluster_LP_13.0",
        "Cluster_LP_14.0",
        "Cluster_LP_18.0",
        "Cluster_LP_19.0",
        "Cluster_LP_20.0",
        "Cluster_LP_21.0",
        "Cluster_LP_22.0",
        "Cluster_LP_24.0",
        "Cluster_LP_25.0",
        "Cluster_LP_26.0",
        "Cluster_LP_27.0",
        "Cluster_LP_28.0",
        "Cluster_LP_29.0",
        "Cluster_LP_30.0",
        "Cluster_LP_31.0",
        "Cluster_LP_37.0",
        "Cluster_LP_38.0",
        "Cluster_LP_39.0",
        "Cluster_LP_40.0",
        "Cluster_LP_41.0",
        "Cluster_LP_42.0",
        "Cluster_LP_43.0",
        "Cluster_LP_45.0",
        "Cluster_LP_57.0",
        "Cluster_LP_58.0",
        "Cluster_LP_59.0",
        "Cluster_LP_60.0",
        "Cluster_LP_63.0",
        "Cluster_LP_64.0",
        "Cluster_LP_65.0",
        "Cluster_LP_66.0",
        "Cluster_LP_67.0",
        "Cluster_LP_80.0",
        "Cluster_LP_82.0",
        "Cluster_LP_85.0",
        "Cluster_LP_89.0",
        "Cluster_LP_90.0",
        "Cluster_LP_91.0",
        "Cluster_LP_100.0",
        "Cluster_LP_101.0",
        "Cluster_LP_102.0",
        "Cluster_LP_103.0",
        "Cluster_LP_104.0",
        "Cluster_LP_105.0",
        "Cluster_LP_107.0",
        "Cluster_LP_108.0",
        "Cluster_LP_109.0",
        "Cluster_LP_110.0",
        "Cluster_LP_111.0",
        "Cluster_LP_124.0",
        "Cluster_LP_125.0",
        "Cluster_LP_129.0",
        "Cluster_LP_150.0",
        "Cluster_LP_151.0",
        "Cluster_LP_152.0",
        "Cluster_LP_154.0",
        "Cluster_LP_155.0",
        "Cluster_LP_156.0",
        "Cluster_LP_157.0",
        "Cluster_LP_158.0",
        "Cluster_LP_159.0",
        "Cluster_LP_160.0",
        "Cluster_LP_161.0",
        "Cluster_LP_163.0",
        "Cluster_LP_164.0",
        "Cluster_LP_165.0",
        "Cluster_LP_166.0",
        "Log_P_A_lag",
        "MCA_1_lag",
        "MCA_2_lag",
        "MCA_3_lag",
        "MCA_4_lag",
        "Tot_AL_lag",
    ]
].astype(float)
M_OLS_LP = spreg.OLS(
    Dep_Var_LP.values,  # Dependent variable
    Ind_Var_LP.values,  # Independent variable
    name_y="Log_P_A",  # Dependent variable name
    name_x=list(Ind_Var_LP.columns),  # Independent variable name
    w=w_Queen_LP,
    spat_diag=True,
    moran=True,
    name_w="w_Queen",
)

print(M_OLS_LP.summary)
data_aveiro_LP_Robust["residuals_OLS"] = M_OLS_LP.u
sns.displot(data_aveiro_LP_Robust["residuals_OLS"], bins=30, kde=True)
plt.title("Distributions of residuals")
# standardisation of the residuals (Z-scores)
data_aveiro_LP_Robust["Z_Score_residuals_OLS"] = stats.zscore(
    data_aveiro_LP_Robust["residuals_OLS"]
)
# distribution of the data against the expected normal distribution.
qqplot(data_aveiro_LP_Robust["Z_Score_residuals_OLS"], line="s")
plt.title("Normal Q-Q plot of residuals")
# Create a Map - Equal Intervals
f, ax = plt.subplots(1, figsize=(8, 12))
ax = data_aveiro_LP_Robust.plot(
    column="Z_Score_residuals_OLS",  # Data to plot
    scheme="EqualInterval",  # Classification scheme
    cmap="bwr",  # Color palette
    edgecolor="k",  # Borderline color
    linewidth=0.1,  # Borderline width
    legend=True,  # Add legend
    legend_kwds={"fmt": "{:.1f}"},  # Remove decimals in legend (for legibility)
    k=10,
    ax=ax,
)

ax.set_title("residuals_OLS")
ax.set_axis_off()
# Create a Map - Equal Intervals
f, ax = plt.subplots(1, figsize=(8, 12))
ax = data_aveiro_LP_Robust.plot(
    column="Z_Score_residuals_OLS",  # Data to plot
    scheme="StdMean",  # Classification scheme
    cmap="bwr",  # Color palette
    edgecolor="k",  # Borderline color
    linewidth=0.1,  # Borderline width
    legend=True,  # Add legend
    legend_kwds={
        "fmt": "{:.1f}",
        "loc": "lower right",
    },  # Remove decimals in legend (for legibility)
    ax=ax,
)

ax.set_title("residuals_OLS")
ax.set_axis_off()
