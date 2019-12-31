class pipeLine:
    
    def __init__(self, dataframe, nominales=None, ordinales=None, numericas=None, target=None):
        """
            Inicializador de clase

            :param dataframe: pandas DataFrame
            :param nominales=None: lista se strings con nombre de columnas clasificadas como nominales
            :param ordinales=None: lista se strings con nombre de columnas clasificadas como ordinales
            :param numericas=None: lista se strings con nombre de columnas clasificadas como numéricas
            :param target='SalePrice': string con el nombre de la columna de la variable 'objetivo'

            El instancia/objeto creado contiene las siguiwentes propiedades de clase:

            :nominales: lista de con el nombre de las variables clasificadas como nominales
            :ordinales: lista de con el nombre de las variables clasificadas como ordinales
            :numericas: lista de con el nombre de las variables clasificadas como numéricas
            :target: sting con el nombre de la variables target/objetivo
            :columns: lista con el nombre de todas las variables del dataframe
            :dataframe: pandas DataFrame que contiene el set de datos 
        """
        self.target = target
        self.df = dataframe.copy()
        self.columns = self.df.columns
        self.nominales = list(nominales) 
        self.ordinales = list(ordinales) 
        self.numericas = list(numericas) 

    def addCols(self, colsDict=None):
        """    
            Añade variables nuevas creadas a la propiedades de clase: nominales, ordinales y numéricas.
                                Actualiza además la propiedad columns

            :param colsDic=None: columnas a añadir. Se necesita crear un diccionario con la siguiente
                                forma: {"nominales": [vars], "ordinales": [vars], "numericas": [vars]}
        """            
        if colsDict is not None:
            for key, vals in colsDict.items():
                if key == 'nominales':
                    self.nominales.extend(vals)
                elif key == 'ordinales':
                    self.ordinales.extend(vals)
                elif key == 'numericas':
                    self.numericas.extend(vals)
                else:
                    pass

            self.columns = self.df.columns
        
    def remCols(self, cols=None):
        """    
            Elimina variables de las propiedades de clase: nominales, ordinales y numéricas y 
                                 actualiza además la propiedad columns

            :param cols=None: lista de strings con el nombre de las variables a eliminar.
        """         
        self.df.drop(cols, axis=1, inplace=True)

        if cols is not None:
            for col in cols:
                if col in self.nominales:
                    self.nominales.remove(col)
                elif col in self.ordinales:
                    self.ordinales.remove(col)
                elif col in self.numericas:
                    self.numericas.remove(col)
                else:
                    pass

            self.columns = self.df.columns  
        
    def elimReg(self, index=None):
        """   
            Elimina registros/observaciones 

            :param index=None: lista con los índices de los registros/obsevaciones a eliminar
        """   
        self.df.drop(index, axis=0, inplace=True)
    
    def chgCatNone(self, cols=None, fill='None'): 
        """    
            Recategoriza variables con datos 'missings' 

            :param cols=None: lista de strings con el nombre de las variables a recategorizar.
            :param fill='None': Valor de sustitución en las variables con categorías 'NaN'
        """   
        self.df[cols] = self.df[cols].apply(lambda x: x.fillna(fill), axis=0)
            
    def indicatorMiss(self, cols=None, concat=False):
        """    
            Genera variables indicadoras dicotómicas de la presencia de observaciones missings manteniendo las
                            variables originales.

            :param cols=None: lista de strings con el nombre de las variables de las que se quieren generar
            varibles indicadoras.
            :param concat=False: booleano que indica si se quiere o no concatenar las variables indicadoras
            generadas a la propiedad de clase 'dataframe'.

            Valor retorno: en el caso de concat=False se devuelve un pandas DataFrame con las variables 
            indicadoras generadas. 
        """
        import numpy as np
        import pandas as pd
        from sklearn.impute import MissingIndicator
        
        if cols is not None:
            names = [col + "_miss" for col in cols]
            dfmiss = pd.DataFrame(MissingIndicator(features='missing-only', error_on_new=False).fit_transform(self.df[cols]), 
                                columns = names, dtype=np.int64)
            
            if concat == False:
                return dfmiss
                                
            elif concat == True:
                self.df = pd.concat([self.df, dfmiss], axis=1)
                self.addCols({'nominales': names})
            else:
                pass
        else:
            pass
        
    def imputerVar(self, cols=None, metodoCont="median", metodoCat="most_frequent"):
        """    
            Imputa valores a los missings de las variables pasadas por parámetro según los métodos determinados
            a través de los parámetros 'metodoCont' y 'metodoCat' para las variables continuas y categóricas
            respectivamente.

            :param cols=None: lista de strings con el nombre de las variables a imputar.
            :param metodoCont='median': método de inmputación para variables continuas. Son válidos los métodos 
            'mean' para la media, 'median' para la mediana, 'most_frequent' para la moda y 'constant' para una 
            constante.
            :param metodoCat='most_frequent': método de inmputación para variables categóricas. Son válidos los 
            métodos 'most_frequent' para la moda y 'constant' para una constante.           
        """
        from sklearn.impute import SimpleImputer
        
        if cols is not None:
            nom= []
            ordn = []
            num = []
            
            #separa variables por tipo
            for var in cols:
                if var in self.nominales:
                    nom.append(var)
                elif var in self.ordinales:
                    ordn.append(var)
                elif var in self.numericas:
                    num.append(var)
                else:
                    pass
            
            if len(num) > 0:
                self.df[num] = SimpleImputer(strategy=metodoCont).fit_transform(self.df[num])
            if len(ordn) > 0:
                self.df[ordn] = SimpleImputer(strategy=metodoCat).fit_transform(self.df[ordn])
            if len(nom) > 0:
                self.df[nom] = SimpleImputer(strategy=metodoCat).fit_transform(self.df[nom])
        else:
            pass  
            
    def featureVCramers(self, cols=None, y=None, threshold=0.3):
        """
            Calcula la importancia de las variables, respecto a la variable target/objetivo, según la 
            VCrammer.

            :param cols=None: lista de strings con nombre de las variables a comparar
            :param y=None: variable target/objetivo de comparación
            :param threshold=0.3

            Valor de retorno: devuelve un objeto pandas Serie con el resultado de la VCrammer ordenando
            las variables de mayor a menor de acuerdo a la importancia según dicho criterio. 
        """
        import scipy.stats as ss
        import numpy as np
        import pandas as pd
        
        VC = {}
    
        for col in cols:
            confusion_matrizCont = pd.crosstab(self.df[col], self.df[y])
            chi2 = ss.chi2_contingency(confusion_matrizCont)[0]
            n = confusion_matrizCont.sum().sum()
            phi2 = chi2/n
            r, k = confusion_matrizCont.shape
            phi2corr = max(0, phi2-((k-1)*(r-1))/(n-1))
            rcorr = r-((r-1)**2)/(n-1)
            kcorr = k-((k-1)**2)/(n-1)
        
            VC.update({col: np.sqrt(phi2corr/min((kcorr-1),(rcorr-1)))})
        
        ps = pd.DataFrame(pd.Series(VC).sort_values(ascending=False), columns=['VCrammer'])
            
        return ps[ps.VCrammer > threshold].style.background_gradient()

    def featureCorrel(self, cols=None, y=None, threshold=0.6):
        """
            Calcula la importancia de las variables, respecto a la variable target/objetivo, según la 
            el valor de la correlación.

            :param cols=None: lista de strings con nombre de las variables a comparar
            :param y=None: variable target/objetivo de comparación.

            Valor de retorno: devuelve un objeto pandas Serie con el resultado de la correlación ordenando
            las variables de mayor a menor de acuerdo a la importancia según dicho criterio. 
        """
        dfCorr = self.df[cols + [y]].corr()[[y]].sort_values(by=str(y), ascending=False)[1:]
        dfCorr.columns = ['Pearson Corr']
        return dfCorr[abs(dfCorr['Pearson Corr']) > threshold].style.background_gradient()
    
    def nzv(self, cols=None, threshold=0.995, thresholdRel = 95/5, delete=True):
        """
            Determina variables cuya relación entre frecuencia más elevada y la total es superior al parámetro 
            threshold. Se retornando dichas variables.

            :param cols=None: lista de strings con nombre de las variables a calcular
            :param threshold=0.995: umbral por el que que se determinada las variables que tienen menos variación.
            calcula la relación entre la moda y el total de observaciones.
            :param thresholdRel=95/5: umbral por el que que se determinada las variables que tienen menos variación.
            Calcula la relación entre la moda y la segunda categoría más frecuente.
            :param delete=True: indicador que determina si las variables capturadas tiene que ser eliminadas o no
            dependiendo si el valor es True o False respectivamente. Por defecto son eliminadas.

            Valor de retorno: se retorna una lista de strings con los nombres de  las variables con menos variación
            si 'delete' es False. Si 'delete' es True se borran las variables estimadas
        """        
        if cols is not None:
            nzv = []
            
            for col in cols:
                counts = self.df[col].value_counts()
                
                cond1 = counts.iloc[0] / len(self.df) > threshold
                try:
                    cond2 = counts.iloc[0] / counts.iloc[1] > thresholdRel
                except IndexError:
                    cond2 = True
                
                if cond1 and cond2:
                    nzv.append(col)
            
            if delete == False:
                return nzv
            elif delete == True:
                self.remCols(nzv)
                return nzv
            else:
                pass
    
    def ordEncoder(self, cols=None, categorias=None):
        """
            Codifica las variables ordinales asígnándoles un número de acuerdo a un orden.

            :param cols=None: lista de strings con nombre de las variables ordinales a calcular
            :param categorias=None: lista de listas. Las listas internas incluyen las categorías de las variables
            a codificas. Ver listado 'cat' más abajo.

            Lista 'cat':
            cat = [["Reg", "IR1", "IR2", "IR3"], #LotShape
                ["AllPub", "NoSewr", "NoSeWa", "ELO"], #Utilities
                ["Gtl", "Mod", "Sev"], #LandSlope
                ["Ex", "Gd", "TA", "Fa", "Po"], #ExternQual   
                ["Ex", "Gd", "TA", "Fa", "Po"], #ExternCond
                ["Ex", "Gd", "TA", "Fa", "Po", "None"], #BsmtQual
                ["Ex", "Gd", "TA", "Fa", "Po", "None"], #BsmtCond
                ["Gd", "Av", "Mn", "No", "None"], #BsmtExposure
                ["GLQ", "ALQ", "BLQ", "Rec", "LwQ", "Unf", "None"], #BsmtFinType1
                ["GLQ", "ALQ", "BLQ", "Rec", "LwQ", "Unf", "None"], #BsmtFinType2
                ["Ex", "Gd", "TA", "Fa", "Po"], #HeatingQC  
                ["SBrkr", "FuseA", "FuseF", "FuseP", "Mix", "None"], #Electrical
                ["Ex", "Gd", "TA", "Fa", "Po"], #KitchenQual
                ["Typ", "Min1", "Min2", "Mod", "Maj1", "Maj2", "Sev", "Sal"], #Functional
                ["Ex", "Gd", "TA", "Fa", "Po", "None"], #FireplaceQu
                ["Fin", "RFn", "Unf", "None"], #GarageFinish
                ["Ex", "Gd", "TA", "Fa", "Po", "None"], #GarageQual
                ["Ex", "Gd", "TA", "Fa", "Po", "None"], #GarageCond
                ["Y", "P", "N"], #PavedDrive 
                ["Ex", "Gd", "TA", "Fa", "None"], #PoolQC
                ["GdPrv", "MnPrv", "GdWo", "MnWw", "None"], #Fence]
                [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], #OverallQual
                [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] #OverallCond
        """   
        from sklearn.preprocessing import OrdinalEncoder

        if cols is not None:
            self.df[cols] = OrdinalEncoder(categories=categorias).fit_transform(self.df[cols])

    def ohEncoder(self, cols=None):
        """
            Genera variables 'dummies' (One Hot Encoding OHE) de las variables pasadas como parámetro. 

            :param cols=None: lista de strings con nombre de las variables categóricas para el OHE.
        """
        import pandas as pd

        if cols is not None:

            self.df[cols] = self.df[cols].astype("object")

            dff = pd.get_dummies(self.df[cols], drop_first= True, prefix=cols)
            
            self.remCols(cols)

            self.df = pd.concat([self.df, dff], axis=1)
            self.nominales = []
            self.numericas.extend(dff.columns)

            self.columns = self.df.columns

    def normVar(self, cols=None):
        """
            Aplica transformaciones óptimas de tipo 'Yeo-Johnson' de las variables numéricas pasadas como 
            parámetro. 

            :param cols=None: lista de strings con nombre de las variables numéricas para ser transformadas.
        """
        import pandas as pd
        from sklearn.preprocessing import PowerTransformer

        if cols is not None:

            columns = list(map(lambda x: x + "_norm", cols))
            dff = pd.DataFrame(PowerTransformer().fit_transform(self.df[cols]), columns = columns)

            self.df = pd.concat([self.df, dff], axis=1)

            self.numericas.extend(columns)

            self.columns = self.df.columns

    def scalers(self, cols=None, metodo='StandardScaler'):
        """
            Aplica transformaciones de escalado de las variables numéricas pasadas como parámetro de acuerdo 
            al parámero 'metodo'. 

            :param cols=None: lista de strings con nombre de las variables numéricas para ser escaladas.
            :param metodo='StandardScaler': indica el método de escalado a aplicar. Los posibles valores son:
            StandardScaler, RobustScaler, MinMaxScaler      
        """
        from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

        if cols is not None:
            if metodo == 'StandardScaler':
                self.df[cols] = StandardScaler().fit_transform(self.df[cols])
            if metodo == 'MinMaxScaler':
                self.df[cols] = MinMaxScaler().fit_transform(self.df[cols])  
            if metodo == 'RobustScaler':
                self.df[cols] = RobustScaler().fit_transform(self.df[cols])

    def featureEng(self):
        """
            Genera las nuevas variables construidas a partir de las variables del dataset original.
            Añade variables nuevas creadas a la propiedades de clase: nominales, ordinales y numéricas.
            Actualiza además la propiedad columns. 
        """
        ########################## Numéricas ########################## 
        #self.df['YrBltAndRemod'] = self.df['YearBuilt'] + self.df['YearRemodAdd']   
     
        self.df['TotalSF'] = self.df['TotalBsmtSF'] + self.df['1stFlrSF'] + self.df['2ndFlrSF'] + self.df['GarageArea']
        
        self.df['Total_sqr_footage'] = self.df['BsmtFinSF1'] + self.df['BsmtFinSF2'] + self.df['1stFlrSF'] + self.df['2ndFlrSF']

        self.df['Total_Bathrooms'] = self.df['FullBath'] + (0.5 * self.df['HalfBath']) + self.df['BsmtFullBath'] + (0.5 * self.df['BsmtHalfBath'])

        self.df['Total_porch_sf'] = self.df['OpenPorchSF'] + self.df['3SsnPorch'] + self.df['EnclosedPorch'] + self.df['ScreenPorch'] + self.df['WoodDeckSF']

        # self.df['Avg_NeigSalePrice'] = self.df["Neighborhood"].map(dict(self.df.groupby("Neighborhood")[self.target].median()))

        ########################## Nominales ########################## 
        self.df['haspool'] = self.df['PoolArea'].apply(lambda x: 1 if x > 0 else 0)

        self.df['has2ndfloor'] = self.df['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)

        self.df['hasgarage'] = self.df['GarageArea'].apply(lambda x: 1 if x > 0 else 0)

        self.df['hasbsmt'] = self.df['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)

        self.df['hasfireplace'] = self.df['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)

        self.df['hasporch'] = self.df['Total_porch_sf'].apply(lambda x: 1 if x > 0 else 0)

        self.df["hasRemodeled"] = (self.df["YearRemodAdd"] - self.df["YearBuilt"]).apply(lambda x: 1 if x > 0 else 0)

        self.df["MSSubClassMap"] = self.df["MSSubClass"].map({20: "1-story", 30: "1-story", 40: "1-story", 120: "1-story",  
                                                           45: "1-1/2-story", 50: "1-1/2-story", 150: "1-1/2-story",
                                                           60: "2-story", 70: "2-story", 75: "2-story", 160: "2-story", 190: "2-story" ,
                                                           80: "1-story", 85: "1-story", 
                                                           90: "1/2-story", 
                                                           180: "multi"})
        # Se actualizan propiedades      

        self.addCols({"numericas": ['TotalSF', 'Total_sqr_footage', 'Total_Bathrooms', 'Total_porch_sf'],
                      "nominales": ['haspool', 'has2ndfloor', 'hasgarage', 'hasbsmt', 'hasfireplace', 'hasporch', "hasRemodeled", "MSSubClassMap"]})
              
        self.columns = self.df.columns
        
        
    def corrDel(self, cols = None, Threshold = 0.90):
        """
            Elimina variables que están correlacionadas según un threshold pasado por parámetro. 

            :param cols=None: lista de strings con nombre de las variables numéricas.
            :param Threshold=0.9: umbral por encima del cual variables correlacionadas serán eliminadas.
            
            Valor de retorno: lista de strings con los nombres de las variables a eliminar 
        """
        import numpy as np
        from itertools import chain
        
        if cols is not None:
            
            cor = self.df[cols].corr()
            cor.loc[:,:] =  np.tril(cor, k=-1)
            cor = cor.stack()
            correlated = cor[cor > Threshold].reset_index().loc[:,['level_0','level_1']].query('level_0 not in level_1')
            correlated_array =  correlated.groupby('level_0').agg(lambda x: set(chain(x.level_0,x.level_1))).values
        
            correlated_features = []

            for sets in correlated_array:
                element_list = list(sets[0])
                for idx, elem in enumerate(element_list):
                    if idx is not 0:
                        correlated_features.append(elem)
        
            self.remCols(correlated_features)
            
            return correlated_features
        
    
    def featureRFE(self, method='RF', cv=20, rango=10):
        """   
            Aplicación del método de seleccion de variables para los siguientes estimadores:
            -LR: regresión Lineal
            -RF: random Forest 

            :param method='RF': aplicación del estimador
            :param cv=20: número de particiones para crossvalidation
            :rango: número de niveles para requeridos del ranking de variables según importancia
            
            Valor de retorno:
            -lista de booleanos con variables de nivel 1, True, en el ranking de importancia
            -lista de enteros con el valor del nivel según importancia
            -diccionario con nombre de variables hasta nivel del ranking determinado en rango
        """
        import numpy as np
        from sklearn.feature_selection import RFECV
    
        y = self.df[self.target].copy()
        X = self.df.drop(self.target, axis=1).copy()
    
        if method == 'LR':
            from sklearn.linear_model import LinearRegression
        
            rfe = RFECV(LinearRegression(), cv=cv).fit(X, y)
        
        elif method == "RF":
            from sklearn.ensemble import RandomForestRegressor
        
            rfe = RFECV(RandomForestRegressor(), cv=cv).fit(X, y)
        
        else:
            pass
                
        imp = {i: list(X.columns[[True if j==i  else False for j in rfe.ranking_]]) for i in range(1,10)}
        
        return rfe.support_, rfe.ranking_, imp
    
    
    def featureEmb(self, method='Lasso', cv=20, Threshold=0.0, table=False):
        """   
            Aplicación del método de seleccion de variables para los siguientes estimadores:
            -Lasso: regresión Lasso
            -Ridge: regresión Ridge

            :param method='Lasso': aplicación del estimador
            :param cv=20: número de particiones para crossvalidation
            :param Threshold=0.0: umbral por encima del cual las variables son consideradas importantes 
            :param table=False: booleano que indica si se quiere o no devolver algo en la función
            
            Valor de retorno: en el caso de table=True se devuelve un pandas Dataframe con nombre de variables.
        """
        import pandas as pd
        import matplotlib.pyplot as plt
        
        y = self.df[self.target].copy()
        X = self.df.drop(self.target, axis=1).copy()
        
        if method == 'Lasso':
            from sklearn.linear_model import LassoCV
            coef = pd.Series(LassoCV(cv=cv).fit(X, y).coef_, index = X.columns)
        
        elif method == 'Ridge':
            from sklearn.linear_model import RidgeCV
            coef = pd.Series(RidgeCV(cv=cv).fit(X, y).coef_, index = X.columns)
        
        else:
            pass

        imp_coef = coef[abs(coef) > Threshold].sort_values(ascending=False)  
        
        plt.rc('xtick', labelsize=25)   
        plt.rc('ytick', labelsize=25)
        plt.rcParams.update({'figure.figsize': (40, 20)})
        
        imp_coef.plot(kind = "bar")
        plt.title("IMPORTANCIA DE VARIABLES SEGÚN MODELO {}\n(Threshold: {})".format(method, Threshold), {'size':'30'})
        
        if table == True:
            return pd.DataFrame(imp_coef[abs(imp_coef) > 0], columns=['coef']).style.background_gradient()
        
        
    def outliersNaN(self, dic=None):
        """   
            Transforma a NaN los valores de las variables pasadas por parámetro dic

            :param dic=None: diccionario cuya 'key' es el nombre de la variable (tipo string) y cuyos 'value'
            es una lista con los índices de las observaciones a transformar
        """
        import numpy as np
        
        if dic is not None:
            for key, value in dic.items():
                self.df[key].loc[value] = np.nan
                
        
    