import os
import numpy as np
import time
import math
import pickle
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.decomposition import PCA


################
num_of_samples = 1000

npz_files = [
    "sum_1B.npz",
    "sum_3B.npz",
    "sum_4B.npz",
    "sum_8B.npz",
    "sum_gpt-4o-mini.npz",
    "sum_gpt-5-mini.npz"
]

##########################

def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero_mask = y_true != 0
    return np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100


##################

def plot_cdfs(vectors, names, title="CDF Plot", xlabel="Value"):

    if len(vectors) != len(names):
        raise ValueError("vectors and names must have same length")

    plt.figure(figsize=(10, 6))

    for vec, name in zip(vectors, names):
        vec = np.array(vec)
        sorted_vec = np.sort(vec)
        cdf = np.arange(1, len(sorted_vec) + 1) / len(sorted_vec)
        plt.plot(sorted_vec, cdf, label=name)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("CDF")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.show()


##################
# SECTION 1: A_Q MODEL
##################

X_list = []
Y_quality_list = []
cpu_load_list = []
gpu_mem_list = []
bert_list = []
rouge_list = []

for file_name in npz_files:
    data = np.load(file_name, allow_pickle=True)

    if len(X_list) == 0:
        X_list.append(data["X"][:num_of_samples])

    Y = data["Y"][:num_of_samples]

    Y_quality_list.append(Y[:, 0].reshape(-1, 1))
    bert_list.append(Y[:, 0].reshape(-1, 1))
    rouge_list.append(Y[:, 1].reshape(-1, 1))
    cpu_load_list.append(Y[:, 3].reshape(-1, 1))
    gpu_mem_list.append(Y[:, 4].reshape(-1, 1))

X = X_list[0]
Y_quality = np.hstack(Y_quality_list)

meta_cpu = np.hstack(cpu_load_list)
meta_gpu = np.hstack(gpu_mem_list)
meta_bert = np.hstack(bert_list)
meta_rouge = np.hstack(rouge_list)

print("A_Q shapes:")
print("X:", X.shape)
print("Y_quality:", Y_quality.shape)

X_train, X_test, y_train, y_test, cpu_train, cpu_test, gpu_train, gpu_test, \
bert_train, bert_test, rouge_train, rouge_test = train_test_split(
    X, Y_quality, meta_cpu, meta_gpu, meta_bert, meta_rouge,
    test_size=0.3, random_state=42
)

rf = RandomForestRegressor(n_estimators=400, max_depth=50, n_jobs=-1)

start = time.time()
rf.fit(X_train, y_train)
train_time = time.time() - start

print(f"training time {train_time}")

y_pred = rf.predict(X_test)

rmse_aq = np.sqrt(np.mean((y_test - y_pred) ** 2))
mae_aq = mean_absolute_error(y_test, y_pred)

print(f"A_Q RF BERT: RMSE={rmse_aq:.4f}, MAE={mae_aq:.4f}")

mae_per_column = mean_absolute_error(y_test, y_pred, multioutput='raw_values')
print("MAE per column:", mae_per_column)


joblib.dump({
    "model": rf,
    "X_train": X_train,
    "X_test": X_test,
    "y_train": y_train,
    "y_test": y_test,
    "meta_cpu_train": cpu_train,
    "meta_cpu_test": cpu_test,
    "meta_gpu_train": gpu_train,
    "meta_gpu_test": gpu_test,
    "meta_bert_train": bert_train,
    "meta_bert_test": bert_test,
    "meta_rouge_train": rouge_train,
    "meta_rouge_test": rouge_test,
    "time_for_training": train_time
}, "rf_model_A_Q.pkl")

print("Saved A_Q model")


############################
# SECTION 2: A_L MODEL
############################

files = npz_files

PCA_DIM = 32


def is_gpt_file(name):
    return "gpt" in name.lower()


slm_X_raw, slm_tokens_list, slm_targets_list = None, None, []
gpt_X_raw, gpt_tokens_list, gpt_bw_list, gpt_targets_list = None, None, None, []

slm_cpu, slm_gpu, slm_bert, slm_rouge = [], [], [], []
gpt_cpu, gpt_gpu, gpt_bert, gpt_rouge = [], [], [], []
gpt_meas_bw, gpt_ctrl_bw = [], []

for f in files:
    data = np.load(f, allow_pickle=True)

    X_array = data["X"]
    prompts = data["prompts"]
    Y_array = data["Y"]

    tokens = np.array([len(p.split()) for p in prompts]).reshape(-1, 1)
    latency = Y_array[:, 2].reshape(-1, 1)

    if is_gpt_file(f):
        bw = data["controlled_bw"].reshape(-1, 1)

        if gpt_X_raw is None:
            gpt_X_raw = X_array
            gpt_tokens_list = tokens
            gpt_bw_list = bw

        gpt_targets_list.append(latency)

        gpt_cpu.append(Y_array[:, 3].reshape(-1, 1))
        gpt_gpu.append(Y_array[:, 4].reshape(-1, 1))
        gpt_bert.append(Y_array[:, 0].reshape(-1, 1))
        gpt_rouge.append(Y_array[:, 1].reshape(-1, 1))
        gpt_meas_bw.append(data["measured_bw"].reshape(-1, 1))
        gpt_ctrl_bw.append(data["controlled_bw"].reshape(-1, 1))

    else:
        if slm_X_raw is None:
            slm_X_raw = X_array
            slm_tokens_list = tokens

        slm_targets_list.append(latency)

        slm_cpu.append(Y_array[:, 3].reshape(-1, 1))
        slm_gpu.append(Y_array[:, 4].reshape(-1, 1))
        slm_bert.append(Y_array[:, 0].reshape(-1, 1))
        slm_rouge.append(Y_array[:, 1].reshape(-1, 1))


slm_targets = np.hstack(slm_targets_list)
slm_tokens = slm_tokens_list

meta_slm_cpu = np.hstack(slm_cpu)
meta_slm_gpu = np.hstack(slm_gpu)
meta_slm_bert = np.hstack(slm_bert)
meta_slm_rouge = np.hstack(slm_rouge)

gpt_targets = np.hstack(gpt_targets_list)
gpt_tokens = gpt_tokens_list
gpt_bw = gpt_bw_list

meta_gpt_cpu = np.hstack(gpt_cpu)
meta_gpt_gpu = np.hstack(gpt_gpu)
meta_gpt_bert = np.hstack(gpt_bert)
meta_gpt_rouge = np.hstack(gpt_rouge)
meta_gpt_meas_bw = np.hstack(gpt_meas_bw)
meta_gpt_ctrl_bw = np.hstack(gpt_ctrl_bw)


slm_token_scaler = StandardScaler()
slm_tokens_scaled = slm_token_scaler.fit_transform(slm_tokens)

gpt_token_scaler = StandardScaler()
gpt_bw_scaler = StandardScaler()

gpt_tokens_scaled = gpt_token_scaler.fit_transform(gpt_tokens)
gpt_bw_scaled = gpt_bw_scaler.fit_transform(gpt_bw)


slm_pca = PCA(n_components=PCA_DIM, whiten=True)
X_slm_final = np.hstack([slm_pca.fit_transform(slm_X_raw), slm_tokens_scaled])

gpt_pca = PCA(n_components=PCA_DIM, whiten=True)
X_gpt_final = np.hstack([
    gpt_pca.fit_transform(gpt_X_raw),
    gpt_tokens_scaled,
    gpt_bw_scaled
])


X_train_slm, X_test_slm, y_train_slm, y_test_slm = train_test_split(
    X_slm_final, slm_targets, test_size=0.3, random_state=42
)

rf_slm = RandomForestRegressor(n_estimators=400, max_depth=50, n_jobs=-1)

start = time.time()
rf_slm.fit(X_train_slm, y_train_slm)
train_time_slm = time.time() - start

y_pred_slm = rf_slm.predict(X_test_slm)
print("SLM MAE:", mean_absolute_error(y_test_slm, y_pred_slm))


X_train_gpt, X_test_gpt, y_train_gpt, y_test_gpt = train_test_split(
    X_gpt_final, gpt_targets, test_size=0.3, random_state=42
)

y_train_gpt[:, 0] *= 2
y_test_gpt[:, 0] *= 2

plot_cdfs(
    [y_train_slm[:,0], y_train_slm[:,1], y_train_slm[:,2], y_train_slm[:,3],
     y_train_gpt[:,0], y_train_gpt[:,1]],
    ['1B', '3B','4B','8B','4o','5-mini']
)

rf_gpt = RandomForestRegressor(n_estimators=400, max_depth=50, n_jobs=-1)

start = time.time()
rf_gpt.fit(X_train_gpt, y_train_gpt)
train_time_llm = time.time() - start

y_pred_gpt = rf_gpt.predict(X_test_gpt)
print("GPT MAE:", mean_absolute_error(y_test_gpt, y_pred_gpt))


pickle.dump({
    "model": rf_slm,
    "pca": slm_pca,
    "token_scaler": slm_token_scaler,
    "X_train": X_train_slm,
    "X_test": X_test_slm,
    "y_train": y_train_slm,
    "y_test": y_test_slm,
    "time_for_training": train_time_slm
}, open("rf_slm_model_A_L.pkl", "wb"))

pickle.dump({
    "model": rf_gpt,
    "pca": gpt_pca,
    "token_scaler": gpt_token_scaler,
    "bw_scaler": gpt_bw_scaler,
    "X_train": X_train_gpt,
    "X_test": X_test_gpt,
    "y_train": y_train_gpt,
    "y_test": y_test_gpt,
    "meta_measured_bw_train": meas_bw_train,
    "meta_measured_bw_test": meas_bw_test,
    "meta_controlled_bw_train": ctrl_bw_train,
    "meta_controlled_bw_test": ctrl_bw_test,
    "time_for_training": train_time_llm
}, open("rf_llm_model_A_L.pkl", "wb"))

print("Saved SLM and GPT models")