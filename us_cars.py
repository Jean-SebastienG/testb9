from io import BytesIO

import streamlit as st
import pandas as pd
import numpy as np
import streamlit.components.v1 as components
from pandas.api.types import (
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
)

import warnings
warnings.filterwarnings("ignore")


st.set_page_config(layout="wide")

st.markdown("<h1 style='text-align: center; color: orange;'>Ventes de voitures aux Etats-Unis</h1>", unsafe_allow_html=True)

DATA_PATH = "data/car_prices_clean.csv"
DATE_COLUMN = "saledate"
FIXED_FILTERS = ["date_de_vente", "marque_du_véhicule", 
                 "modèle_du_véhicule", "prix_de_vente"]

def format_column(c):
    return c.replace("_", " ").capitalize()

@st.cache_data
def load_data() -> pd.DataFrame:
    """
    Load data from CSV file

    Args:
        None

    Returns:
        pd.DataFrame: Loaded dataframe and rename columns
    """
    data = pd.read_csv(DATA_PATH)
    lowercase = lambda x: str(x).lower()
    data.rename(lowercase, axis='columns', inplace=True)
    data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN])
    data.columns = ["année", "marque_du_véhicule", "modèle_du_véhicule", "trim", "type_de_véhicule", 
                    "transmission", "etat", "condition", "kilomètres",
                    "couleur_du_vehicule", "intérieur", "nom_du_vendeur", "mmr", 
                    "prix_de_vente", "date_de_vente"]
    data.columns = [c.lower() for c in data.columns]
    column_order = ["marque_du_véhicule", "modèle_du_véhicule", "type_de_véhicule", 
                    "prix_de_vente", "date_de_vente", "année", "trim",
                    "transmission", "etat", "condition", "kilomètres",
                    "couleur_du_vehicule", "intérieur", "nom_du_vendeur", "mmr"]
    data = data[column_order]
    data["type_de_véhicule"] = pd.Categorical(data.type_de_véhicule)
    return data

def filter_dataframe(df: pd.DataFrame, fixed_filters: list) -> pd.DataFrame:
    """
    Adds a UI on top of a dataframe to let viewers filter columns

    Args:
        df (pd.DataFrame): Original dataframe

    Returns:
        pd.DataFrame: Filtered dataframe
    """

    df = df.copy()

    # Try to convert datetimes into a standard format (datetime, no timezone)
    for col in df.columns:
        if is_object_dtype(df[col]):
            try:
                df[col] = pd.to_datetime(df[col])
            except Exception:
                pass

        if is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.tz_localize(None)

    with st.sidebar:
        left, right = st.columns((1, 20))

        added_filters = right.multiselect("Ajouter un filtre", [c for c in df.columns if c not in fixed_filters])
        
        to_filter_columns = fixed_filters + added_filters
        
        for column in to_filter_columns:
            # Treat columns with < 10 unique values as categorical
            if is_categorical_dtype(df[column]) or df[column].nunique() < 10:
                user_cat_input = right.multiselect(
                    f"Values for {column}",
                    df[column].unique(),
                    default=list(df[column].unique()),
                )
                df = df[df[column].isin(user_cat_input)]
            
            elif is_numeric_dtype(df[column]):
                _min = float(df[column].min())
                _max = float(df[column].max())
                step = (_max - _min) / 100
                user_num_input = right.slider(
                    f"{format_column(column)}",
                    min_value=_min,
                    max_value=_max,
                    value=(_min, _max),
                    step=step,
                )
                df = df[df[column].between(*user_num_input)]
            
            elif is_datetime64_any_dtype(df[column]):
                user_date_input = right.date_input(
                    f"{format_column(column)}",
                    value=(
                        df[column].min(),
                        df[column].max(),
                    ),
                )
                if len(user_date_input) == 2:
                    user_date_input = tuple(map(pd.to_datetime, user_date_input))
                    start_date, end_date = user_date_input
                    df = df.loc[df[column].between(start_date, end_date)]
            
            else:
                user_text_input = right.text_input(
                    f"{format_column(column)}",
                    placeholder="Chaîne de caractère ou regex"
                )
                if user_text_input:
                    df = df[df[column].astype(str).str.contains(user_text_input)]
            
    return df

def group_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collect all requested data from user with streamlit inputs.
    Then, apply groupby on dataframe.

    Args:
        df (pd.DataFrame): Original dataframe

    Returns:
        pd.DataFrame: Grouped dataframe
    """
    with st.sidebar:
        left, right = st.columns((1, 20))

        by = right.selectbox(
            "Grouper sur cette colonne",
            df.columns,
            format_func=format_column,
            index=None
        )

        columns = right.multiselect(
                    f"Colonnes à agréger",
                    df.columns,
                    format_func=format_column,
                )
        numeric_agg = ["min", "max", "mean", "median", "sum"]
        categorical_agg = ["expand", "first", "last"]
        agg_dict = {}
        
        for column in columns:
            if is_numeric_dtype(df[column]):
                agg_func = right.selectbox(
                    f"Grouper les valeurs de {format_column(column)} avec",
                    numeric_agg,
                    index=None
                )
                agg_dict[column] = [agg_func]
            else:
                agg_func = right.selectbox(
                    f"Grouper les valeurs de {format_column(column)} avec",
                    categorical_agg,
                    index=None
                )
                agg_dict[column] = [agg_func]
                

        
        if by and columns and agg_dict and all([agg[0] for agg in agg_dict.values()]):
            for column, agg_func in agg_dict.copy().items():
                if agg_func == ["expand"]:
                    df_dummies = pd.get_dummies(df[column])
                    df = pd.concat([df, df_dummies], axis=1)
                    for c in df_dummies.columns:
                        agg_dict[c] = "sum"
                    del agg_dict[column]
                elif agg_func in ["first", "last"]:
                    agg_dict[column] = lambda x: x.iloc[0]
            
            df = df[[by]+list(agg_dict.keys())].groupby(by=by).agg(agg_dict)
    
    return df

def format_sort_options(ascending: bool) -> str:
    if ascending:
        return "Croissant"
    else:
        return "Décroissant"

def sort_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sort dataframe.

    Args:
        df (pd.DataFrame): Original dataframe

    Returns:
        pd.DataFrame: Sorted dataframe
    """
    with st.sidebar:
        left, right = st.columns((1, 20))

        by = right.selectbox(
            "Trier sur cette colonne",
            df.columns,
            placeholder="Choisir une colonne",
            format_func=format_column,
            index=None
        )

        ascending = right.selectbox(
            "Type de tri",
            [True, False],
            format_func=format_sort_options,
        )
    if by:
        return df.sort_values(by=by, ascending=ascending)
    else:
        return df


def exploration() -> pd.DataFrame:
    data = load_data()
    
    st.sidebar.markdown("### :orange[Trier les lignes]")
    data = sort_rows(data)

    st.sidebar.markdown("### :orange[Filtrer les lignes]")
    data = filter_dataframe(data, fixed_filters=FIXED_FILTERS)

    st.sidebar.markdown("### :orange[Grouper les données]")

    data = group_dataframe(data)
    
    return data

df = exploration()


column_config = {
    "prix_de_vente": st.column_config.NumberColumn(
                        "Prix de vente",
                        help="Prix de vente du véhicule en dollars américains",
                        format="$%d",
                    ),
    "année": st.column_config.NumberColumn(
                        "Année",
                        help="Année de fabrication du véhicule",
                        format="%d",
                    ),
    "marque_du_véhicule": st.column_config.TextColumn(
            "Marque",
            help="Marque du véhicule vendu",
        ),
    "modèle_du_véhicule": st.column_config.TextColumn(
            "Modèle",
            help="Modèle du véhicule vendu",
        ),
    "type_de_véhicule": st.column_config.TextColumn(
            "Type",
            help="Type du véhicule vendu",
        ),
    "date_de_vente": st.column_config.DateColumn(
            "Date de la vente",
            help="Date de la vente du véhicule",
            format="DD.MM.YYYY",
            step=1,
        ),
}

# Display dataframe
st.dataframe(df, use_container_width=True, height=600, 
             hide_index=True,
             column_config=column_config)

# Excel file
col1, col2, col3 = st.columns([2,2,2])
flnme = col3.text_input("Nom du fichier excel")
if not flnme:
    pass
else:
    if flnme.endswith(".xlsx") == False:  # add file extension if it is forgotten
        flnme = flnme + ".xlsx"
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name='Report')
    col3.download_button(label="Télécharger le fichier", data=buffer.getvalue(), file_name=flnme, mime="application/vnd.ms-excel")
