# IMPORTS #
import torch # tensor operations
import pandas as pd # dataframes
import numpy as np # np arrays 
from pathlib import Path # for getting data
from sklearn.model_selection import KFold # for k-fold cross validation
import graphlearning as gl # for Calder implementation
from sklearn.preprocessing import StandardScaler # for standardizing data
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error, mean_squared_log_error # error metrics
#from torch_geometric.utils import to_undirected

# for pGNN
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from sklearn.neighbors import kneighbors_graph
from scipy.sparse import coo_matrix

import sys
sys.path.append(str(Path(__file__).parent.parent))
import src
from src import models
import matplotlib.pyplot as plt
import seaborn as sns

# for parallelization
import joblib # would be good if 
import multiprocessing as mp # good for the standard approach to create multiple os processes for each step 

# import sys # for dow imports:
# sys.path.append(str(Path(__file__).parent.parent / "src"))
# from torch.utils.data import DataLoader
# from dataset import MemoryDataset
# from distance import generate_knn_weights, EuclideanWeight, WeightedDistance, apply_laplacian, TrainingMethod
# from models.laplacian import PLaplacian1, PLaplacian2, PLaplacian3
# from models.basic import TinyNet

# IMPORTS #


# LOAD DATA #
# want to go one out of the directory of this file, then into the data folder, then grab the csv there
data_path = Path(__file__).parent.parent / "data" / "penn_data.csv"
data = pd.read_csv(data_path)
if (data.empty):
    print("ERR: Data not loaded!")
    exit()
else:
    print("Data loaded!")
# LOAD DATA #

'''
loop order:
1) dataset (m/f/c)
2) target (alm, bmd, bfp, age_w_y, age_wo_y) # set X & y here
3) model (calder_gl, GNN) # define hyperparameter grid for specific model
4) hyperparamter1 (k)
5) hyperparamter2 (p)
6) run (10)
7) fold (5)
8) train model => calculate all metrics => store all metrics
'''

# INIT RESULTS TENSOR #
# Define variables in the order they appear in the loop
seaborn_x_labels = ["Male", "Female", "Combined"] # dataset splits (1st loop)
seaborn_y_labels = ["ALM", "BMD", "BFP", "AGE_w_Y", "AGE_wo_Y"] # target (2nd loop)
# TODO: add your test_case here!

test_cases = [
    "calder_gl",           # baseline (Calder)
    "zach_p_l2",           # baseline (your solver)
    "zach_p_l2_c_alm",     # |corr(feature, ALM)|
    "zach_p_l2_c2_alm",    # |corr|^2 with ALM
    "zach_p_l2_c_bmd",
    "zach_p_l2_c2_bmd",
    "zach_p_l2_c_bfp",
    "zach_p_l2_c2_bfp"#,
    #"zach_top_aware_distance",
    #"zach_hom_reg_lap_updates"
]

'''
test_cases = ["calder_gl, zach_p_l2"]#,
              "zach_p_l1", 
              "zach_p_cos", 
              "zach_p_p3", 
              "zach_p_p5", 
              "zach_p_p10", 
              "dow_pGNN",
              "dow_pl_corr",
              "dow_pl",
              "dow_tinynet"]#"dow_pl1", "dow_pl2", "dow_pl3", "dow_tinynet", "dow_rl_pl3"]#, "GNN", "p_GNN", "zach_rewrite_euclidean", "dow_cpp_rewrite", "dow_cpp_vectorized", "zach_embedded_distance"] # model (3rd loop)
'''
k_values = [2, 5, 10] # number of neighbors for knn graph (4th loop)
p_values = [2, 2.5, 3]#, 10, 100, 100000] # p-laplacian parameter (5th loop)
hy = "k"
hx = "p"
number_of_runs = 1 # run (6th loop)
k_folds = 5 # fold (7th loop)

test_metrics = ["rmse", "mae", "mse", "r2", "mape", "msle", "huber", "relative_rmse"] # metric (8th loop)
def relative_rmse(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    if np.mean(y_true**2) < 1e-9:
        return np.inf
    return 100 * rmse / np.sqrt(np.mean(y_true**2))
def huber_loss(y_true, y_pred, delta=1.0):
    error = y_true - y_pred
    is_small_error = np.abs(error) <= delta
    squared_loss = 0.5 * error[is_small_error] ** 2
    linear_loss = delta * (np.abs(error[~is_small_error]) - 0.5 * delta)
    return np.mean(np.concatenate([squared_loss, linear_loss]))
metric_function = {
    "rmse": lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)),
    "mae": mean_absolute_error,
    "mse": mean_squared_error,
    "r2": r2_score,
    "mape": mean_absolute_percentage_error,
    "msle": mean_squared_log_error,
    "huber": lambda y_true, y_pred: huber_loss(y_true, y_pred),
    "relative_rmse": relative_rmse
}



# Updated tensor dimensions to match loop order: dataset, target, model, k, p, run, fold, metric
results_tensor = torch.zeros(len(seaborn_x_labels),     # dataset (m/f/c)
                             len(seaborn_y_labels),     # target (alm, bmd, bfp, age_w_y, age_wo_y)
                             len(test_cases),           # model (calder_gl, GNN)
                             len(k_values),             # hyperparameter1 (k)
                             len(p_values),             # hyperparameter2 (p)
                             number_of_runs,            # run (10)
                             k_folds,                   # fold (5)
                             len(test_metrics)          # metric
)




'''
idea for new TDA methods:

considering any waveform can be approximated by a fourier series, a combination of sin/cos waves as a basis,
and neural networks can approximate any cts function (UAT) with a basis of activation functions and combination of layers,
hopefully any combination 


'''


# DISTANCE FUNCTIONS #

def l1_distance(X1, X2):
    return torch.cdist(X1, X2, p=1)

def l2_distance(X1, X2):
    return torch.cdist(X1, X2, p=2)

def cos_distance(X1,X2):
    epsilon = 1e-8 # for numerical stability...
    x1_norm = X1.norm(dim=1)[:, None]
    x2_norm = X2.norm(dim=1)[:, None]

    cos_similarity = torch.mm(X1, X2.t()) / (x1_norm * x2_norm.t() + epsilon)
    return 1 - cos_similarity
    
def p10_distance(X1, X2):
    return torch.cdist(X1, X2, p=10)

def p5_distance(X1, X2):
    return torch.cdist(X1, X2, p=5)

def p3_distance(X1, X2):
    return torch.cdist(X1, X2, p=3)

def weighted_l2_distance(X1, X2, alpha_weights):
    # If no weights are provided, this is just the standard L2 distance.
    if alpha_weights is None:
        return torch.cdist(X1, X2, p=2)

    alpha_weights = alpha_weights.to(X1.device) # weights on same device as data...
    
    # Scale the feature matrices by alpha weights
    X1_scaled = X1 * alpha_weights
    X2_scaled = X2 * alpha_weights
    
    return l2_distance(X1_scaled, X2_scaled) # reusing method below

# DISTANCE FUNCTIONS #


# CORRELATION FNs #
def compute_alpha_from_corr(X_train_raw: np.ndarray, y_train_for_weights: np.ndarray, power: int) -> np.ndarray:
    """
    Per-feature weights α_j >= 0 from Pearson corr(x_j, y_weight) on TRAIN ONLY.
    power=1 -> |corr|, power=2 -> |corr|^2. Normalized so mean(α)=1.
    """
    d = X_train_raw.shape[1]
    corr = np.zeros(d, dtype=float)
    for j in range(d):
        xj = X_train_raw[:, j]
        if np.std(xj) < 1e-12:
            corr[j] = 0.0
        else:
            cmat = np.corrcoef(xj, y_train_for_weights)
            corr[j] = 0.0 if np.isnan(cmat[0,1]) else cmat[0,1]
    alpha = np.abs(corr) ** power
    alpha /= (alpha.mean() + 1e-12)
    return alpha.astype(np.float32)

def parse_corr_variant(test_case: str):
    t = test_case.lower()
    if "_c2_" in t:
        biom = t.split("_c2_")[-1].upper()
        return 2, biom
    if "_c_" in t:
        biom = t.split("_c_")[-1].upper()
        return 1, biom
    return None, None


# CORRELATION FNs #

# pGNN #

# Add PGNN classes (copy from the file)
class pLaplacianConv(nn.Module):
    def __init__(self, K=5, p=2.0, mu=0.01):
        super().__init__()
        self.K = K
        self.p = p
        self.mu = mu

    def forward(self, h, edge_index):
        row, col = edge_index
        for _ in range(self.K):
            diff = h[row] - h[col]
            norm_diff = torch.norm(diff, dim=1).clamp(min=1e-6)
            weights = (norm_diff / norm_diff.max()).pow(self.p - 2)
            agg = torch.zeros_like(h)
            for dim in range(h.size(1)):
                agg[:, dim].index_add_(0, row, weights * (h[col, dim] - h[row, dim]))
            h = h - self.mu * agg
        return h

class PGNNRegressor(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, K=5, p=2.0, mu=0.01, dropout=0.3):
        super().__init__()
        self.lin1 = nn.Linear(in_dim, hid_dim)
        self.bn1 = nn.BatchNorm1d(hid_dim)
        self.lin2 = nn.Linear(hid_dim, hid_dim)
        self.bn2 = nn.BatchNorm1d(hid_dim)
        self.drop = nn.Dropout(dropout)
        self.pconv = pLaplacianConv(K, p, mu)
        self.out = nn.Linear(hid_dim, out_dim)

    def forward(self, data):
        h, edge_index = data.x, data.edge_index
        h = F.relu(self.bn1(self.lin1(h)))
        h = self.drop(h)
        h = F.relu(self.bn2(self.lin2(h)))
        h = self.drop(h)
        h = self.pconv(h, edge_index)
        return self.out(h)

# pGNN #    


# HELPERS # 

def custom_to_undirected(edge_index, num_nodes):
    if edge_index.numel() == 0:
        return edge_index
    row, col = edge_index
    combined = torch.cat([torch.stack([row, col], dim=0), torch.stack([col, row], dim=0)], dim=1)
    combined = torch.unique(combined, dim=1)
    return combined

def custom_scatter_add(src, index, dim_size):
    out = torch.zeros(dim_size, dtype=src.dtype, device=src.device)
    return out.scatter_add_(0, index, src)

def custom_scatter_max(src, index, dim_size, fill_value=-float('inf')):
    if len(src) == 0:
        return torch.full((dim_size,), fill_value, dtype=src.dtype, device=src.device)
    sorted_idx = torch.sort(index).indices
    sorted_index = index[sorted_idx]
    sorted_src = src[sorted_idx]
    unique_index, counts = torch.unique_consecutive(sorted_index, return_counts=True)
    max_values = torch.full((dim_size,), fill_value, dtype=src.dtype, device=src.device)
    cumsum = torch.cumsum(counts, dim=0)
    starts = torch.cat([torch.tensor([0], device=src.device), cumsum[:-1]])
    for i, (start, count, u_idx) in enumerate(zip(starts, counts, unique_index)):
        segment = sorted_src[start:start + count]
        max_values[u_idx] = segment.max()
    return max_values

def custom_scatter_min(src, index, dim_size, fill_value=float('inf')):
    if len(src) == 0:
        return torch.full((dim_size,), fill_value, dtype=src.dtype, device=src.device)
    sorted_idx = torch.sort(index).indices
    sorted_index = index[sorted_idx]
    sorted_src = src[sorted_idx]
    unique_index, counts = torch.unique_consecutive(sorted_index, return_counts=True)
    min_values = torch.full((dim_size,), fill_value, dtype=src.dtype, device=src.device)
    cumsum = torch.cumsum(counts, dim=0)
    starts = torch.cat([torch.tensor([0], device=src.device), cumsum[:-1]])
    for i, (start, count, u_idx) in enumerate(zip(starts, counts, unique_index)):
        segment = sorted_src[start:start + count]
        min_values[u_idx] = segment.min()
    return min_values

# ZACH P-LAPLACIAN IMPLEMENTATION #
class pLaplacian_Zimplementation:
    def __init__(self, X_all_features, y_all_labels, train_mask, test_mask,
                 distance_fn,
                 k=10,
                 p_laplacian_param=2.0,
                 weight_gamma=1.0,
                 device='cpu'):
        self.X_all = X_all_features.to(device)
        self.y_all_true = y_all_labels.to(device)
        self.train_mask = train_mask.to(device)
        self.test_mask = test_mask.to(device)
        self.distance_fn = distance_fn
        self.k = k
        self.p_laplacian_param = float(p_laplacian_param)
        self.weight_gamma = weight_gamma
        self.device = device
        self.num_nodes = X_all_features.shape[0]
        self.u_solution = torch.zeros(self.num_nodes, dtype=torch.float32, device=self.device)
        
        self._build_graph_and_initialize_u()

    def _build_knn_graph(self):
        if not isinstance(self.k, int) or self.k <= 0:
            raise ValueError("Parameter 'k' for k-NN graph must be a positive integer.")
        if self.num_nodes <= 1:
            self.edge_index = torch.empty((2, 0), dtype=torch.long, device=self.device)
            self.edge_attr = torch.empty((0,), dtype=torch.float32, device=self.device)
            return
        
        current_k = min(self.k, self.num_nodes - 1)
        full_dist_matrix = self.distance_fn(self.X_all, self.X_all)
        full_dist_matrix.fill_diagonal_(float('inf'))
        
        _, indices = torch.topk(full_dist_matrix, k=current_k, dim=1, largest=False)
        source_nodes = torch.arange(self.num_nodes, device=self.device).view(-1, 1).repeat(1, current_k)
        edge_index_knn_directed = torch.stack([source_nodes.view(-1), indices.view(-1)], dim=0)
        
        self.edge_index = custom_to_undirected(edge_index_knn_directed, self.num_nodes)
        
        if self.edge_index.numel() > 0:
            row, col = self.edge_index
            distances_on_edges = full_dist_matrix[row, col]
            self.edge_attr = torch.exp(-self.weight_gamma * distances_on_edges**2).to(self.device)
        else:
            self.edge_attr = torch.tensor([], dtype=torch.float32, device=self.device)

    def _build_graph_and_initialize_u(self):
        self._build_knn_graph()
        self.u_solution[self.train_mask] = self.y_all_true[self.train_mask].float()
        nodes_to_initialize = self.test_mask & ~self.train_mask
        if self.train_mask.sum() > 0:
            mean_train_y = self.y_all_true[self.train_mask].float().mean()
            self.u_solution[nodes_to_initialize] = mean_train_y
        else:
            self.u_solution[nodes_to_initialize] = 0.0

    def _calculate_random_walk_component(self, u_prev, row, col, safe_weighted_degrees):
        if self.edge_index.numel() == 0:
            return torch.zeros(self.num_nodes, device=self.device)
        messages_to_i = self.edge_attr * u_prev[col]
        sum_w_u_neighbors = custom_scatter_add(messages_to_i, row, self.num_nodes)
        return sum_w_u_neighbors / safe_weighted_degrees

    def _calculate_min_max_component(self, u_prev, row, col):
        if self.edge_index.numel() == 0:
            return u_prev
        max_val_neighbors = custom_scatter_max(u_prev[col], row, self.num_nodes, fill_value=-torch.inf)
        min_val_neighbors = custom_scatter_min(u_prev[col], row, self.num_nodes, fill_value=torch.inf)
        is_isolated = (max_val_neighbors == -torch.inf)
        max_val_neighbors[is_isolated] = u_prev[is_isolated]
        min_val_neighbors[is_isolated] = u_prev[is_isolated]
        return (max_val_neighbors + min_val_neighbors) / 2.0

    def fit(self, max_iterations=100, tolerance=1e-6):
        if self.p_laplacian_param < 2.0:
            raise ValueError("p must be >= 2!")
        alpha = 1.0 / (self.p_laplacian_param - 1.0) if self.p_laplacian_param != float('inf') else 0.0
        
        if self.edge_index.numel() == 0: return self.u_solution
        
        row, col = self.edge_index
        weighted_degrees = custom_scatter_add(self.edge_attr, row, self.num_nodes)
        safe_weighted_degrees = weighted_degrees.clamp(min=1e-12)
        updatable_test_nodes_mask = self.test_mask & (weighted_degrees > 0) & ~self.train_mask

        for iteration in range(max_iterations):
            u_prev = self.u_solution.clone()
            rw_component = self._calculate_random_walk_component(u_prev, row, col, safe_weighted_degrees)
            
            if alpha < 1.0:
                min_max_component = self._calculate_min_max_component(u_prev, row, col)
                combined_term = alpha * rw_component + (1.0 - alpha) * min_max_component
            else:
                combined_term = rw_component
                
            self.u_solution[updatable_test_nodes_mask] = combined_term[updatable_test_nodes_mask]
            
            if updatable_test_nodes_mask.sum() > 0:
                diff = torch.norm(self.u_solution[updatable_test_nodes_mask] - u_prev[updatable_test_nodes_mask])
                if diff < tolerance and iteration > 0: break
            else:
                break
        return self.u_solution

    def predict(self):
        return self.u_solution[self.test_mask]
# ZACH P-LAPLACIAN IMPLEMENTATION #


# INIT RESULTS TENSOR #

target_columns = ["ALM", "BMD - Total", "% fat - Total"]
excluded_columns = ["0", "PPT ID", "Site", "Race"]
# scale data








# RUNNING LOOP # 
for dataset_idx, seaborn_x_label in enumerate(seaborn_x_labels):

    if seaborn_x_label == "Male":
        df_filtered = data[data["Gender"] == "Male"]
    elif seaborn_x_label == "Female":
        df_filtered = data[data["Gender"] == "Female"]
    else:
        df_filtered = data.copy()


    # biomarker vectors aligned with df_filtered rows (used only to form α)
    y_all_map = {
        "ALM": df_filtered["ALM"].values,
        "BMD": df_filtered["BMD - Total"].values,
        "BFP": df_filtered["% fat - Total"].values,
        # not including these for now... but may later...
        #"AGE_w_Y": df_filtered["Age"].values,
        #"AGE_wo_Y": df_filtered["Age"].values,
    }

    # get the power and biom
    #power, biom = parse_corr_variant(test_case)





    for target_idx, seaborn_y_label in enumerate(seaborn_y_labels):

        if seaborn_y_label == "Age_w_Y":
            target_column = "Age"
            target_map = {
                "ALM": "ALM",
                "BMD": "BMD - Total",
                "BFP": "% fat - Total",
                "AGE_w_Y": "Age",
                "AGE_wo_Y": "Age"
            }
            columns_to_exclude = excluded_columns
        elif seaborn_y_label == "Age_wo_Y":
            target_column = "Age"
            columns_to_exclude = excluded_columns + target_columns
        else:
            target_map = {
                "ALM": "ALM",
                "BMD": "BMD - Total",
                "BFP": "% fat - Total",
                "AGE_w_Y": "Age",
                "AGE_wo_Y": "Age"
            }
            target_column = seaborn_y_label
            columns_to_exclude = excluded_columns + [t for t in target_columns if t != target_column]
        
        X = df_filtered.drop(columns=columns_to_exclude + [target_column, 'Gender'], errors='ignore').values
        y = df_filtered[target_map[seaborn_y_label]].values

        # TODO: modify the custom parameters for separate cases if needed
        for model_idx, test_case in enumerate(test_cases):
            # modify hyperparameter tests if unique model (will keep !SAME SIZE! for simplicity)
            # default stays as definded above
            if test_case == "GNN": 
                print("GNN model detected, changing hyperparameters to epsilon and alpha - rename the plot!!!!!")
                k_values = [2, 5]#, 10] # now represent epsilon (y-axis)
                p_values = [2, 2.5]#, 3#, 10, 100, 100000] # now represent alpha (x-axis)
                hy = "e"
                hx = "a"
            elif test_case == "dow_pGNN":
                print("p_GNN model detected, changing hyperparameters to layers/hops and the p-based - rename the plot!!!!!")
                k_values = [2, 5]#, 10] # now represent number of layers/hops (y-axis)
                p_values = [2, 2.5]#, 3#, 10, 100, 100000] # still represents p (x-axis)
                hy = "lh"
                hx = "p"
            elif test_case == "dow_tinynet":
                k_values = [1e-2, 1e-3]#, 1e-4] # now represent lr (y-axis)
                p_values = [20, 40]#, 60, 80, 100]#, 200] # now represent epochs (x-axis)
                hy = "e"
                hx = "a"
            elif test_case == "dow_rl_pl3":
                print("Keeping default hyperparameters of k (nearest neighbors) and p (p-laplacian tug-of-war parameter)")
            else:
                print("Keeping default hyperparameters of k (nearest neighbors) and p (p-laplacian tug-of-war parameter)")

            print("------------------------------------------------")

            for k_idx, k in enumerate(k_values):
                for p_idx, p in enumerate(p_values):
                    print(f"Running {test_case} to predict {seaborn_y_label} on {seaborn_x_label} dataset ({hy}={k}, {hx}={p})")
                    for run in range(number_of_runs): # where parallelization will section off 
                        k_fold = KFold(n_splits=k_folds, shuffle=True, random_state=run)
                        for fold_idx, (train_index, test_index) in enumerate(k_fold.split(X)):

                            # PARALLELIZE: parallelize the code to work s.t. each run gets a core (or see if there can be more to this)
                            
                            X_train_raw, X_test_raw = X[train_index], X[test_index]
                            y_train, y_test = y[train_index], y[test_index] # targets are usually not scaled
                            

                            # standardize for modeling (correlation will use RAW)
                            scaler = StandardScaler()
                            X_train = scaler.fit_transform(X_train_raw)
                            X_test  = scaler.transform(X_test_raw)

                            # decide if this model is a correlation-weighted variant
                            power, biom = parse_corr_variant(test_case)
                            if biom is not None:
                                y_weight_train = y_all_map[biom][train_index]  # TRAIN ONLY (no leakage)
                                alpha_np = compute_alpha_from_corr(X_train_raw, y_weight_train, power=power)
                            else:
                                alpha_np = np.ones(X_train.shape[1], dtype=np.float32)
                            alpha_t = torch.from_numpy(alpha_np)


                            # TODO: write your own custom implementation
                            # IMPLEMENTATIONS #
                            if test_case == "calder_gl_OLD":
                                # combine data for semi-supervised graph construction
                                X_combined = np.vstack((X_train, X_test)) # ie = [X_train, X_test]
                                num_train = X_train.shape[0]
                                labeled_indices = np.arange(num_train)
                                labeled_values = y_train

                                # create graph
                                W = gl.weightmatrix.knn(X_combined, k)
                                G = gl.graph(W)
                                yhat_combined = G.plaplace(labeled_indices, labeled_values, p)

                                # get arrays for error predictions
                                y_pred = yhat_combined[num_train:]
                                y_true = y_test

                                pass
                            elif test_case == "calder_gl":

                                # scale features by α (α=1 for baseline)
                                X_train_w = X_train * alpha_np
                                X_test_w  = X_test  * alpha_np

                                X_combined = np.vstack((X_train_w, X_test_w))
                                num_train = X_train.shape[0]
                                labeled_indices = np.arange(num_train)
                                labeled_values = y_train

                                W = gl.weightmatrix.knn(X_combined, k)
                                G = gl.graph(W)
                                yhat_combined = G.plaplace(labeled_indices, labeled_values, p)

                                y_pred = yhat_combined[num_train:]
                                y_true = y_test
                                pass 
                            elif test_case.startswith("zach_p_l2"):
                                
                                X_combined_t = torch.tensor(np.vstack((X_train, X_test)), dtype=torch.float32)
                                y_combined_t = torch.tensor(np.concatenate((y_train, y_test)), dtype=torch.float32)
                                train_mask = torch.tensor([True]*len(X_train) + [False]*len(X_test))
                                test_mask  = torch.tensor([False]*len(X_train) + [True]*len(X_test))

                                # α-weighted L2 (α=1 vector for baseline; corr-weights for c-variants)
                                dist_fn = lambda A, B: weighted_l2_distance(A, B, alpha_t)

                                model = pLaplacian_Zimplementation(
                                    X_all_features=X_combined_t,
                                    y_all_labels=y_combined_t,
                                    train_mask=train_mask,
                                    test_mask=test_mask,
                                    distance_fn=dist_fn,
                                    k=k,
                                    p_laplacian_param=p,
                                    device='cpu'
                                )
                                model.fit(max_iterations=60, tolerance=1e-4)
                                y_pred = model.predict().cpu().numpy()
                                y_true = y_test
                                
                                pass
                            elif test_case == "GNN":
                                # GNN implementation
                                pass
                            elif test_case == "dow_pGNN":
                                # build graph
                                A = kneighbors_graph(X_train, n_neighbors=10, mode='connectivity', include_self=False)
                                coo = coo_matrix(A)
                                edge_index = torch.tensor(np.vstack((coo.row, coo.col)), dtype=torch.long)

                                # create training data
                                train_data = Data(
                                    x=torch.tensor(X_train, dtype=torch.float32),
                                    edge_index=edge_index,
                                    y=torch.tensor(y_train[:, None], dtype=torch.float32)  # Single target
                                )

                                #  
                                # Step size mapping for different p values
                                step_sizes = {2: 0.01, 2.5: 0.01, 3: 0.01, 10: 0.002, 100: 0.0005, 100000: 0.0001}
                                mu = step_sizes.get(p, 0.01)
    
                                # Create and train model
                                model = PGNNRegressor(
                                    in_dim=X_train.shape[1], 
                                    hid_dim=64, 
                                    out_dim=1, 
                                    K=k,  # k is your K parameter
                                    p=p, 
                                    mu=mu, 
                                    dropout=0.3
                                )
    
                                optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
                                loss_fn = nn.MSELoss()
    
                                # Training loop (reduced epochs for speed)
                                epochs = 20  # Reduced for speed
                                model.train()
                                for epoch in range(epochs):
                                    optimizer.zero_grad()
                                    out = model(train_data)
                                    loss = loss_fn(out.squeeze(), train_data.y.squeeze())
                                    loss.backward()
                                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                                    optimizer.step()
    
                                # Prediction on test set
                                model.eval()
                                with torch.no_grad():
                                    # Forward pass through layers manually for test data
                                    X_test_t = torch.tensor(X_test, dtype=torch.float32)
                                    h = F.relu(model.bn1(model.lin1(X_test_t)))
                                    h = model.drop(h)
                                    h = F.relu(model.bn2(model.lin2(h)))
                                    h = model.drop(h)
                                    y_pred = model.out(h).squeeze().numpy()
    
                                y_true = y_test
                            elif test_case == "zach_p_l1":
                                # Option 1: Use masks for semi-supervised graph learning
                                X_combined_t = torch.tensor(np.vstack((X_train, X_test)), dtype=torch.float32)
                                y_combined_t = torch.tensor(np.concatenate((y_train, y_test)), dtype=torch.float32)
                                train_mask = torch.tensor([True] * len(X_train) + [False] * len(X_test))
                                test_mask = torch.tensor([False] * len(X_train) + [True] * len(X_test))
                                model = pLaplacian_Zimplementation(
                                    X_combined_t,
                                    y_combined_t,
                                    train_mask,
                                    test_mask,
                                    distance_fn=l1_distance,
                                    k=k,
                                    p_laplacian_param=p,
                                    device='cpu')
                                model.fit(max_iterations=60, tolerance=1e-4)
                                y_pred = model.predict().cpu().numpy()
                                y_true = y_test
                            elif test_case == "zach_p_l2":
                                # Option 1: Use masks for semi-supervised graph learning
                                X_combined_t = torch.tensor(np.vstack((X_train, X_test)), dtype=torch.float32)
                                y_combined_t = torch.tensor(np.concatenate((y_train, y_test)), dtype=torch.float32)
                                train_mask = torch.tensor([True] * len(X_train) + [False] * len(X_test))
                                test_mask = torch.tensor([False] * len(X_train) + [True] * len(X_test))
                                model = pLaplacian_Zimplementation(
                                    X_combined_t,
                                    y_combined_t,
                                    train_mask,
                                    test_mask,
                                    distance_fn=l2_distance,
                                    k=k,
                                    p_laplacian_param=p,
                                    device='cpu')
                                model.fit(max_iterations=60, tolerance=1e-4)
                                y_pred = model.predict().cpu().numpy()
                                y_true = y_test
                            elif test_case == "zach_p_cos":
                                # Option 1: Use masks for semi-supervised graph learning
                                X_combined_t = torch.tensor(np.vstack((X_train, X_test)), dtype=torch.float32)
                                y_combined_t = torch.tensor(np.concatenate((y_train, y_test)), dtype=torch.float32)
                                train_mask = torch.tensor([True] * len(X_train) + [False] * len(X_test))
                                test_mask = torch.tensor([False] * len(X_train) + [True] * len(X_test))
                                model = pLaplacian_Zimplementation(
                                    X_combined_t,
                                    y_combined_t,
                                    train_mask,
                                    test_mask,
                                    distance_fn=cos_distance,
                                    k=k,
                                    p_laplacian_param=p,
                                    device='cpu')
                                model.fit(max_iterations=60, tolerance=1e-4)
                                y_pred = model.predict().cpu().numpy()
                                y_true = y_test
                            elif test_case == "zach_p_p3":
                                # Option 1: Use masks for semi-supervised graph learning
                                X_combined_t = torch.tensor(np.vstack((X_train, X_test)), dtype=torch.float32)
                                y_combined_t = torch.tensor(np.concatenate((y_train, y_test)), dtype=torch.float32)
                                train_mask = torch.tensor([True] * len(X_train) + [False] * len(X_test))
                                test_mask = torch.tensor([False] * len(X_train) + [True] * len(X_test))
                                model = pLaplacian_Zimplementation(
                                    X_combined_t,
                                    y_combined_t,
                                    train_mask,
                                    test_mask,
                                    distance_fn=p3_distance,
                                    k=k,
                                    p_laplacian_param=p,
                                    device='cpu')
                                model.fit(max_iterations=60, tolerance=1e-4)
                                y_pred = model.predict().cpu().numpy()
                                y_true = y_test
                            elif test_case == "zach_p_p5":
                                # Option 1: Use masks for semi-supervised graph learning
                                X_combined_t = torch.tensor(np.vstack((X_train, X_test)), dtype=torch.float32)
                                y_combined_t = torch.tensor(np.concatenate((y_train, y_test)), dtype=torch.float32)
                                train_mask = torch.tensor([True] * len(X_train) + [False] * len(X_test))
                                test_mask = torch.tensor([False] * len(X_train) + [True] * len(X_test))
                                model = pLaplacian_Zimplementation(
                                    X_combined_t,
                                    y_combined_t,
                                    train_mask,
                                    test_mask,
                                    distance_fn=p5_distance,
                                    k=k,
                                    p_laplacian_param=p,
                                    device='cpu')
                                model.fit(max_iterations=60, tolerance=1e-4)
                                y_pred = model.predict().cpu().numpy()
                                y_true = y_test
                            elif test_case == "zach_p_p10":
                                # Option 1: Use masks for semi-supervised graph learning
                                X_combined_t = torch.tensor(np.vstack((X_train, X_test)), dtype=torch.float32)
                                y_combined_t = torch.tensor(np.concatenate((y_train, y_test)), dtype=torch.float32)
                                train_mask = torch.tensor([True] * len(X_train) + [False] * len(X_test))
                                test_mask = torch.tensor([False] * len(X_train) + [True] * len(X_test))
                                model = pLaplacian_Zimplementation(
                                    X_combined_t,
                                    y_combined_t,
                                    train_mask,
                                    test_mask,
                                    distance_fn=p10_distance,
                                    k=k,
                                    p_laplacian_param=p,
                                    device='cpu')
                                model.fit(max_iterations=60, tolerance=1e-4)
                                y_pred = model.predict().cpu().numpy()
                                y_true = y_test
                            elif test_case == "dow_pl":
                                test_set = src.MemoryDataset(X_test, y_test)
                                train_set = src.MemoryDataset(X_train, y_train)
                                weights = src.generate_knn_weights(src.EuclideanWeight(),
                                                                   torch.cat((X_train, X_test),dim=1))
                                model = models.PLaplacian3(p, train_set, test_set, weights)
                                model.fit()
                                y_pred = model.predicted_labels[len(train_set):,:]
                            elif test_case == "dow_pl_corr":
                                test_set = src.MemoryDataset(X_test, y_test)
                                train_set = src.MemoryDataset(X_train, y_train)
                                weights = src.generate_knn_weights(src.CorrelationDistance(2),
                                                                   torch.cat((X_train, X_test),dim=1))
                                model = models.PLaplacian3(p, train_set, test_set, weights)
                                model.fit()
                                y_pred = model.predicted_labels[len(train_set):,:]
                            elif test_case == "dow_tinynet":
                                test_set = src.MemoryDataset(X_test, y_test)
                                train_set = src.MemoryDataset(X_train, y_train)
                                m = train_set.targets.size(1)
                                f = train_set.features.size(1)
                                model = models.TinyNet(f, m)    
                                
                                trainloader = torch.utils.data.DataLoader(
                                    train_set,
                                    batch_size=16,
                                    shuffle=True,
                                    generator=torch.Generator(device=torch.get_default_device())
                                )
                                trainer = src.TrainingMethod(model, trainloader)
                                trainer.train(epochs=p,
                                            verbose=False)
                                y_pred = model(test_set.features)
                            else:
                                print("ERR: Model not found!")
                                exit()

                            # IMPLEMENTATIONS #


                            # METRICS # 
                            for metric_idx, test_metric in enumerate(test_metrics):
                                # TODO: Store results in tensor
                                # results_tensor[dataset_idx, target_idx, model_idx, k_idx, p_idx, run, fold_idx, metric_idx] = metric_value
                                metric_func = metric_function.get(test_metric)
                                if metric_func is None:
                                    print(f"ERR: Metric {test_metric} not found!")
                                    exit()
                                metric_value = metric_func(y_true, y_pred)
                                results_tensor[dataset_idx, target_idx, model_idx, k_idx, p_idx, run, fold_idx, metric_idx] = metric_value
                            # METRICS # 

                            
            #print("Mean RMSE: ", results_tensor[dataset_idx, target_idx, model_idx, k_idx, p_idx, run, fold_idx, test_metrics.index("relative_rmse")].mean())
            metric_idx = test_metrics.index("relative_rmse")
            block = results_tensor[dataset_idx, target_idx, model_idx, k_idx, p_idx, :, :, metric_idx]
            print(f"[{test_case}] {seaborn_y_label}/{seaborn_x_label} (k={k}, p={p}) "
                  f"mean RelRMSE = {block.mean().item():.3f}%")
 

            # LIVE OUTPUT SEABORN HEATMAP # 
            # not doing this but here is where it would go 
            # LIVE OUTPUT SEABORN HEATMAP #      

# RUNNING LOOP # 








# PLOTTING LOOP # 

# now all results are in the tensor, just need to create the heatmaps:

# PARALLELIZE at the model level... each model could get its own core in the plotting process

for model_idx, test_case in enumerate(test_cases): # iterate through the model because that will be the bottleneck... the problem is that we have to share the same results tensor but not same parts are accessed

    # title the seaborn plot & init it
    fig, axes = plt.subplots(nrows=len(seaborn_y_labels), ncols=len(seaborn_x_labels), figsize=(15, 20))
    fig.suptitle(f"{test_case} Results", fontsize=16, fontweight='bold')

    # Re-define hyperparameter labels if they were changed for specific models
    current_k_values = k_values
    current_p_values = p_values
    current_hy = hy
    current_hx = hx
    
    if test_case == "GNN": 
        current_hy = "e"  # epsilon
        current_hx = "a"  # alpha
    elif test_case == "dow_pGNN":
        current_hy = "lh"  # layers/hops
        current_hx = "p"   # p
    elif test_case == "dow_tinynet":
        current_hy = "e"  # epsilon
        current_hx = "a"  # alpha

    for target_idx, seaborn_y_label in enumerate(seaborn_y_labels):
        for dataset_idx, seaborn_x_label in enumerate(seaborn_x_labels):
            ax = axes[target_idx, dataset_idx]

            # Get the relative_rmse metric data for this specific combination
            metric_idx = test_metrics.index("relative_rmse")
            
            # Average across runs and folds to get mean performance for each (k,p) combination
            # Shape: [k_values, p_values, runs, folds] -> [k_values, p_values]
            heatmap_data = results_tensor[dataset_idx, target_idx, model_idx, :, :, :, :, metric_idx].mean(dim=(2, 3))
            
            # Convert to numpy for seaborn
            heatmap_data_np = heatmap_data.numpy()
            
            # Create the heatmap
            sns.heatmap(heatmap_data_np,
                        annot=True,
                        fmt=".2f",
                        cmap="viridis_r",  # _r for reverse (lower values = better = darker)
                        xticklabels=current_p_values,
                        yticklabels=current_k_values,
                        ax=ax,
                        cbar_kws={'label': 'Relative RMSE (%)'})
            
            # Set labels and title for this subplot
            ax.set_title(f"{seaborn_y_label} - {seaborn_x_label}", fontsize=12, fontweight='bold')
            ax.set_xlabel(f"{current_hx} values", fontsize=10)
            ax.set_ylabel(f"{current_hy} values", fontsize=10)
            
            # Print summary statistics for this combination
            mean_performance = heatmap_data_np.mean()
            min_performance = heatmap_data_np.min()
            print(f"{test_case} - {seaborn_y_label} - {seaborn_x_label}: Mean={mean_performance:.3f}, Best={min_performance:.3f}")

    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)  # Make room for the main title
    
    # Show or save the plot
    # plt.show()
    # Optional: save the figure
    seaborn_output_dir = Path(__file__).parent / "seaborn_output"
    seaborn_output_dir.mkdir(parents=True, exist_ok=True)
    #plt.savefig(seaborn_output_dir / f"{test_case}_heatmaps.png", dpi=300, bbox_inches='tight')
    plt.savefig(seaborn_output_dir / f"{test_case}_heatmaps.pdf", bbox_inches='tight')

    
    
    plt.close()  # Close to free memory

# PLOTTING LOOP # 


# ADDITIONAL ANALYSIS BELOW #
# ANALYSIS OF RESULTS #
print("\n" + "="*50)
print("PERFORMANCE ANALYSIS")
print("="*50 + "\n")

# Get the index for the metric we want to analyze (relative_rmse)
metric_idx = test_metrics.index("relative_rmse")

# Average the results across all runs and folds to get a stable performance measure
# Shape changes from (dataset, target, model, k, p, run, fold, metric) -> (dataset, target, model, k, p, metric)
mean_results_per_hyperparam = results_tensor.mean(dim=(5, 6))

# Now, average across all other experimental variables (dataset, target, k, p)
# to get a single performance score for each model.
overall_model_performance = mean_results_per_hyperparam[:, :, :, :, :, metric_idx].mean(dim=(0, 1, 3, 4))

# Store results in a more friendly dictionary
performance_dict = {
    test_case: overall_model_performance[model_idx].item()
    for model_idx, test_case in enumerate(test_cases)
}

print("--- Overall Mean Relative RMSE for Each Model ---")
for model, perf in performance_dict.items():
    print(f"{model:<25}: {perf:.3f}%")
print("-" * 45)

# --- Group Analysis ---
# Helper function to calculate average performance for a group of models
def analyze_group(group_name, identifier_list):
    group_perfs = [
        perf for model, perf in performance_dict.items()
        if any(identifier in model for identifier in identifier_list)
    ]
    if not group_perfs:
        return 0.0
    return np.mean(group_perfs)

# 1. Compare |corr| vs |corr|^2
c1_perf = analyze_group("|corr|", ["_c_"])
c2_perf = analyze_group("|corr|^2", ["_c2_"])

# 2. Compare different biomarker weights
alm_perf = analyze_group("ALM-weighted", ["_alm"])
bmd_perf = analyze_group("BMD-weighted", ["_bmd"])
bfp_perf = analyze_group("BFP-weighted", ["_bfp"])

# --- Print Takeaways ---
print("\n--- General Takeaways ---")

# Compare c vs c2
print("\n[Comparison: Correlation Power]")
print(f"Average performance of |corr| models      : {c1_perf:.3f}%")
print(f"Average performance of |corr|^2 models     : {c2_perf:.3f}%")
if c2_perf < c1_perf:
    print(">> Takeaway: Using squared correlation (|corr|^2) generally works better. ✅")
else:
    print(">> Takeaway: Using absolute correlation (|corr|) generally works better or is comparable. ❌")

# Compare biomarkers
print("\n[Comparison: Biomarker Weighting]")
print(f"Average performance of ALM-weighted models : {alm_perf:.3f}%")
print(f"Average performance of BMD-weighted models : {bmd_perf:.3f}%")
print(f"Average performance of BFP-weighted models : {bfp_perf:.3f}%")
best_biomarker = min({"ALM": alm_perf, "BMD": bmd_perf, "BFP": bfp_perf}.items(), key=lambda item: item[1])
print(f">> Takeaway: Weighting features by correlation with '{best_biomarker[0]}' seems to be the most effective strategy overall. 🚀")
print("="*50 + "\n")