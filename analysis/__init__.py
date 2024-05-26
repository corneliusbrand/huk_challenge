import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import arff
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

class Analysis:
    def __init__(self) -> None:
        self.df = None 

        self.nnls = None
        self.dtr = None

        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None

    def load_and_clean(self):
        data_freq = arff.load('freMTPL2freq.arff')
        df_freq = pd.DataFrame(data_freq, columns=["IDpol", "ClaimNb", "Exposure", "Area", "VehPower",
        "VehAge","DrivAge", "BonusMalus", "VehBrand", "VehGas", "Density", "Region"])
        data_sev = arff.load('freMTPL2sev.arff')
        df_sev = pd.DataFrame(data_sev, columns=["IDpol", "ClaimAmount"])

        # Left join, weil es 6 Policen mit Schäden gibt, die nicht in df_freq erfasst sind
        df_clean = df_freq.merge(df_sev, on="IDpol", how='left')
        df_clean.fillna(0.0, inplace=True)
        
        # Abhängige Variable definieren, Indexspalte nicht benötigt
        df_clean["ClaimPerTime"] = df_clean["ClaimAmount"] / df_clean["Exposure"]
        df_clean = df_clean.drop(columns=["IDpol"])

        # Region 21 wird ausgeschlossen
        df_clean = df_clean[df_clean["Region"] != "'R21'"]

        # Ausreißer ("black swans") werden ausgeschlossen
        df_clean = df_clean[df_clean["ClaimPerTime"] < 7_500_000]
        self.df = df_clean

    # Helper-Funktion, um eindimensionale Daten zu clustern
    @staticmethod
    def cluster(n_clusters, data):
        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        kmeans.fit(np.array(data).reshape(-1,1))
        return kmeans

    def transform_features(self):
        regions = self.df.groupby("Region")["ClaimPerTime"].mean()
        new_regions = Analysis.cluster(3, regions)
        region_mapping = {k : str(v) for (k,v) in zip(regions.index,new_regions.labels_)}

        brands = self.df.groupby("VehBrand")["ClaimPerTime"].mean()
        new_brands = Analysis.cluster(3, brands)
        brand_mapping = {k : str(v) for (k,v) in zip(brands.index,new_brands.labels_)}

        areas = self.df.groupby("Area")["ClaimPerTime"].mean()
        new_areas = Analysis.cluster(2, areas)
        area_mapping =  {k : str(v) for (k,v) in zip(areas.index,new_areas.labels_)}

        # Dimensionsreduktion auf Features
        df_dimred = self.df
        df_dimred["VehBrand"] = df_dimred["VehBrand"].map(lambda k: brand_mapping[k])
        df_dimred["Region"] = df_dimred["Region"].map(lambda k: region_mapping[k])
        df_dimred["Area"] = df_dimred["Area"].map(lambda k: area_mapping[k])

        # One-hot-encodings
        df_engineered = pd.get_dummies(df_dimred, columns=["Area", "VehBrand", "Region"])
        df_engineered =  pd.get_dummies(df_engineered, columns=["VehGas"], drop_first=True)

        # Konvertieren von bool in int für kompaktere Ausgabe, eigentlich unnötig
        df_engineered = df_engineered.mask(df_engineered.dtypes == bool, df_engineered.astype(int))
        self.df = df_engineered

    def train_test_split(self):
        X = self.df.drop(columns=["ClaimNb","Exposure", "ClaimAmount", "ClaimPerTime"])
        y = self.df['ClaimPerTime']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Nicht-negative Least Squares fitten
    def train_nnls(self):
        self.nnls = LinearRegression(positive=True, fit_intercept=False)
        self.nnls.fit(self.X_train, self.y_train)

    def eval_nnls(self):
        y_pred = self.nnls.predict(self.X_test)

        print("Nicht-Negative Lineare Regression:")
        print("Mittlere Absoluter Fehler:", np.sqrt(mean_absolute_error(self.y_test,y_pred)))
        print("Mittlerer Gewinn pro Jahr pro Kunde: ", (y_pred - self.y_test).mean())
        for label, coef in zip(self.X_train.columns, self.nnls.coef_):
            print(f"{label}: {coef:.2f}")
        print("\n")

    # Entscheidungsbaum-Regressor fitten
    def train_dtr(self):        
        self.dtr = DecisionTreeRegressor(random_state=42, criterion="squared_error", max_depth=3)
        self.dtr.fit(self.X_train, self.y_train)

    def eval_dtr(self):
        y_pred = self.dtr.predict(self.X_test)

        print("Entscheidungsbaum-Regressor:")
        print("Mittlere Absoluter Fehler:", np.sqrt(mean_absolute_error(self.y_test,y_pred)))
        print("Mittlerer Gewinn pro Jahr pro Kunde: ", (y_pred - self.y_test).mean())
        clf = self.dtr
        
        # Aus den sklearn-docs
        n_nodes = clf.tree_.node_count
        children_left = clf.tree_.children_left
        children_right = clf.tree_.children_right
        feature = clf.tree_.feature
        threshold = clf.tree_.threshold
        values = clf.tree_.value

        node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
        is_leaves = np.zeros(shape=n_nodes, dtype=bool)
        stack = [(0, 0)]  # start with the root node id (0) and its depth (0)
        while len(stack) > 0:
            # `pop` ensures each node is only visited once
            node_id, depth = stack.pop()
            node_depth[node_id] = depth

            # If the left and right child of a node is not the same we have a split
            # node
            is_split_node = children_left[node_id] != children_right[node_id]
            # If a split node, append left and right children and depth to `stack`
            # so we can loop through them
            if is_split_node:
                stack.append((children_left[node_id], depth + 1))
                stack.append((children_right[node_id], depth + 1))
            else:
                is_leaves[node_id] = True

        print(
            "The binary tree structure has {n} nodes and has "
            "the following tree structure:\n".format(n=n_nodes)
        )
        for i in range(n_nodes):
            if is_leaves[i]:
                print(
                    "{space} {node} leaf with value={value:.1f}.".format(
                        space=node_depth[i] * "\t", node=i, value=values[i][0][0]
                    )
                )
            else:
                print(
                    "{space} {node} split at {value:.1f}: "
                    "{left} if {feature} <= {threshold} "
                    "else {right}.".format(
                        space=node_depth[i] * "\t",
                        node=i,
                        left=children_left[i],
                        feature=self.X_train.columns[feature[i]],
                        threshold=threshold[i],
                        right=children_right[i],
                        value=values[i][0][0],
                    )
                )

    def predict_dtr(self,x):
        return self.dtr.predict(np.array(x).reshape(-1,1))
    
    def predict_nnls(self,x):
        return self.nnls.predict(np.array(x).reshape(-1,1))