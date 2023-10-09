import string
import numpy as np
import pandas as pd
import scipy.spatial.distance as sc

class CDIN:
    def __init__(self, df):
        self.data = df
    # Métodos para clasificación de Datos en un DataFrame (Cualitativos, Cuantitativos, Binarios)
    
    def get_cuantitativos(df):
        df_cuantitativos = df.select_dtypes(include = "number")
        columns_cuantitativas = df_cuantitativos.columns
        return columns_cuantitativas, df_cuantitativos
    
    def get_categoricos(df):
        df_cat = df.select_dtypes(include = "object")
        columns_cat = df_cat.columns
        return columns_cat, df_cat
    
    def get_categoricos_non_binaries(df):
        df_cat = df.select_dtypes(include=['object']).copy()
        column_cat = df_cat.columns
        column_non_binaries = []
        for col in column_cat:
            if df_cat[col].nunique() != 2:
                column_non_binaries.append(col)
        return column_non_binaries, df[column_non_binaries]

    def get_binaries(df):
        df_cat = df.select_dtypes(include=['object']).copy()
        column_cat = df_cat.columns
        column_binaries = []
        for col in column_cat:
            if df_cat[col].nunique() == 2:
                column_binaries.append(col)
        return column_binaries, df[column_binaries]
    
    ## Métodos para la limpieza de datos
    
    # remover signos de puntuación
    @staticmethod
    def remove_punctuation(x):
        try:
            x = ''.join(ch for ch in x if ch not in string.punctuation)
        except:
            print(f'{x} no es una cadena de caracteres')
            pass
        return x
    @staticmethod
    def remove_digits(x):
        try:
            x = "".join(ch for ch in x if ch not in string.digits)
        except:
            print(f'{x} no es una cadena de caracteres')
            pass
        return x
    #remover espacios en blanco
    @staticmethod
    def remove_whitespace(x):
        try:
            x="".join(x.split())
        except:
            pass
        return x
    #convertimos a minúsculas
    @staticmethod
    def lower_text(x):
        try:
            x=x.lower()
        except:
            pass
        return x
    #Convertimos a mayúsculas
    @staticmethod
    def upper_text(x):
        try:
            x=x.upper()
        except:
            pass
        return x
    # función convierta a mayúsculas la primera letra
    @staticmethod
    def capitalize_text(x):
        try:
            x = x.capitalize()
        except:
            pass
        return x
    @staticmethod
    def replace_text(x,to_replace, replacement):
        try:
            x = x.replace(to_replace,replacement)
        except:
            pass
        return x
    @staticmethod
    def remove_letters(x):
        try:
            x = "".join(ch for ch in x if ch not in string.ascii_letters)
        except:
            pass
        return x

    
    def dqr(data):
        columnas = list(data.columns.values)
        columns = pd.DataFrame(columnas, columns = ['Nombre_Columnas'], index=columnas)
        dtypes = pd.DataFrame(data.dtypes, columns=['Tipo_Datos'])
        no_nulos = pd.DataFrame(data.count(), columns = ['Valores_Presentes'] )
        missing_values = pd.DataFrame(data.isnull().sum(), columns = ['Valores_Faltantes'])
        unique_values = pd.DataFrame(columns = ['Valores_Unicos'])
        for col in columnas:
            unique_values.loc[col] = [data[col].nunique()]



        # Se puede crear una columna que sea boolena /// Categorical
        categorical = pd.DataFrame(columns = ['Categorical'], index=columnas)
        for col in columnas:
            if data[col].dtypes == 'object':
                categorical.loc[col] = [True]
            else:
                categorical.loc[col] = [False]
        # Otra que muestre las categorias si son pocas
        categories = pd.DataFrame(columns = ['Categories'], index=columnas)
        for col in columnas:    
            if categorical.loc[col][0] == True:
                if data[col].nunique() <= 10:
                    categories.loc[col] = [data[col].unique()]
                else:
                    categories.loc[col] = "Too much categories"
        # Aplicar el análisis estadístico solo a las columnas que son numericas

        max_values = pd.DataFrame(columns = ['Max_Values'])
        for col in columnas:
            try:
                if categorical.loc[col][0] == False:
                    max_values.loc[col] = [data[col].max()]
            except:
                pass

        min_values = pd.DataFrame(columns = ['Min_Values'])
        for col in columnas:
            try:
                if categorical.loc[col][0] == False:
                    min_values.loc[col] = [data[col].min()]
            except:
                pass

        mean = pd.DataFrame(columns = ['Mean'])
        for col in columnas:
            try:
                if categorical.loc[col][0] == False:
                    mean.loc[col] = [round(data[col].mean(),3)]
            except:
                pass

        std = pd.DataFrame(columns = ['Std'])
        for col in columnas:
            try:
                if categorical.loc[col][0] == False:
                    std.loc[col] = [round(data[col].std(),3)]
            except:
                pass

        return columns.join(dtypes).join(no_nulos).join(missing_values).join(unique_values).join(max_values).join(min_values).join(mean).join(std).join(categorical).join(categories)
    
    def pdistance_matrix(df, metric):
        '''
        Este método obtiene la matriz de distancias de df apartir de la métrica dada en el argumento de entrada
        df: DataFrame
        metric: Métrica (string)
        '''
        # Validación de las métricas de similitud programadas en pdist
        values = ['braycurtis', 'canberra', 'chebyshev', 'cityblock',
        'correlation', 'cosine', 'dice', 'euclidean', 'hamming',
        'jaccard', 'jensenshannon', 'kulczynski1',
        'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto',
        'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath',
        'sqeuclidean', 'yule']
        if metric in values:
            D = sc.squareform(sc.pdist(df.values,metric))
            return pd.DataFrame(D)
        else:
            print("Valor de métrica inválido")