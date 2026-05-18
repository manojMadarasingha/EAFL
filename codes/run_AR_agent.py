import joblib
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error

# -----------------------------
# Load A_Q RF model
# -----------------------------
aq_data = joblib.load("rf_model_A_Q.pkl")
rf_aq = aq_data["model"]
X_test_aq = aq_data["X_test"]
y_test_aq = aq_data["y_test"]

y_pred_aq = rf_aq.predict(X_test_aq)

rmse_aq = np.sqrt(np.mean((y_test_aq - y_pred_aq) ** 2))
mae_aq = mean_absolute_error(y_test_aq, y_pred_aq)

print(f"A_Q RF BERT: RMSE={rmse_aq:.4f}, MAE={mae_aq:.4f}")

sns.histplot(y_pred_aq[:, 5].flatten(), bins=30, kde=False, color='skyblue', stat='density')
plt.show()

sns.histplot(y_test_aq[:, 5].flatten(), bins=30, kde=False, color='skyblue', stat='density')
plt.show()


# -----------------------------
# Load A_L SLM RF model
# -----------------------------
slm_data = pickle.load(open("rf_slm_model_A_L.pkl", "rb"))
rf_slm = slm_data["model"]
X_test_slm = slm_data["X_test"]
y_test_slm = slm_data["y_test"]

y_pred_slm = rf_slm.predict(X_test_slm)

rmse_slm = np.sqrt(np.mean((y_test_slm - y_pred_slm) ** 2))
mae_slm = mean_absolute_error(y_test_slm, y_pred_slm)

print(f"A_L SLM Latency: RMSE={rmse_slm:.4f}, MAE={mae_slm:.4f}")


# -----------------------------
# Load A_L LLM RF model
# -----------------------------
llm_data = pickle.load(open("rf_llm_model_A_L.pkl", "rb"))
rf_llm = llm_data["model"]
X_test_llm = llm_data["X_test"]
y_test_llm = llm_data["y_test"]

y_pred_llm = rf_llm.predict(X_test_llm)

rmse_llm = np.sqrt(np.mean((y_test_llm - y_pred_llm) ** 2))
mae_llm = mean_absolute_error(y_test_llm, y_pred_llm)

print(f"A_L LLM Latency: RMSE={rmse_llm:.4f}, MAE={mae_llm:.4f}")


# -----------------------------
# Final vectors
# -----------------------------
bert_vector_pred = y_pred_aq
bert_vector_test = y_test_aq

latency_vector_slm = y_pred_slm
latency_vector_llm = y_pred_llm

latency_vector_pred = np.hstack([latency_vector_slm, latency_vector_llm])
latency_vector_test = np.hstack([y_test_slm, y_test_llm])

print("BERT vector shape:", bert_vector_pred.shape)
print("Latency vector shape (SLM+LLM):", latency_vector_pred.shape)


# =========================================================
# USER REQUIREMENTS GENERATION
# =========================================================

mu = 12
sigma = 3
min_gb = 1
max_gb = 24

gpu_mem_samples = np.random.normal(mu, sigma, 300)
gpu_mem_samples = np.clip(gpu_mem_samples, min_gb, max_gb)

mu_q = 0.8516866634761262
sigma_q = 0.024341867397121516

quality_samples = np.random.normal(mu_q, sigma_q, 300)
quality_samples = np.clip(quality_samples, 0.0, 1.0)

loc_l = 0.0
scale_l = 4.174222222222222

latency_samples = np.random.exponential(scale=scale_l, size=300) + loc_l

user_reqirments = []

for i in range(300):
    user_reqirments.append({
        "quality_req": float(quality_samples[i]),
        "latency_req": float(latency_samples[i]),
        "gpu_available": float(gpu_mem_samples[i])
    })


for sample in user_reqirments[:10]:
    print(sample)


# =========================================================
# FUZZY LOGIC
# =========================================================

def triangular(x, a, b, c):
    if x <= a or x >= c:
        return 0.0
    if x == b:
        return 1.0
    if a < x < b:
        return (x - a) / (b - a)
    return (c - x) / (c - b)


def trapezoid(x, a, b, c, d):
    if x <= a or x >= d:
        return 0.0
    elif a < x < b:
        return (x - a) / (b - a)
    elif b <= x <= c:
        return 1.0
    else:
        return (d - x) / (d - c)


def crisp_score(memberships, scores=[1, 2, 3]):
    total = sum(memberships) + 1e-9
    norm = [m / total for m in memberships]
    return sum(s * m for s, m in zip(scores, norm))


# -----------------------------
# SLM WEIGHTS
# -----------------------------
def compute_weights_slm(quality_req, latency_req, gpu_available, gpu_total):

    gpu_ratio = gpu_available / gpu_total

    q_low  = trapezoid(quality_req, 0.00, 0.00, 0.8053, 0.8310)
    q_med  = trapezoid(quality_req, 0.8053, 0.8310, 0.8515, 0.8670)
    q_high = trapezoid(quality_req, 0.8515, 0.8940, 1.00, 1.00)

    l_low  = trapezoid(latency_req, 0, 3.489, 7.089, 8.720)
    l_med  = trapezoid(latency_req, 6.264, 7.889, 8.720, 10.731)
    l_high = trapezoid(latency_req, 8.720, 9.637, 10.731, 40.411)

    g_low  = trapezoid(gpu_ratio, 0.00, 0.00, 0.2568, 0.3870)
    g_med  = trapezoid(gpu_ratio, 0.2568, 0.3870, 0.4851, 0.5582)
    g_high = trapezoid(gpu_ratio, 0.4851, 0.6787, 1.00, 1.00)

    w_q = crisp_score([q_low, q_med, q_high])
    w_l = crisp_score([l_low, l_med, l_high])
    w_s = crisp_score([g_low, g_med, g_high])

    total = w_q + w_l + w_s
    return w_q / total, w_l / total, w_s / total


# -----------------------------
# LLM WEIGHTS
# -----------------------------
def compute_weights_llm(quality_req, latency_req):

    q_low  = trapezoid(quality_req, 0.00, 0.00, 0.8053, 0.8310)
    q_med  = trapezoid(quality_req, 0.8053, 0.8310, 0.8515, 0.8670)
    q_high = trapezoid(quality_req, 0.8515, 0.8940, 1.00, 1.00)

    l_low  = trapezoid(latency_req, 0.00, 0.00, 1.00, 6.683)
    l_med  = trapezoid(latency_req, 6.00, 7.50, 9.50, 11.00)
    l_high = trapezoid(latency_req, 9.50, 14.00, 40.00, 40.00)

    w_q = crisp_score([q_low, q_med, q_high])
    w_l = crisp_score([l_low, l_med, l_high])

    total = w_q + w_l
    return w_q / total, w_l / total


# =========================================================
# MODEL SELECTION FUNCTION
# =========================================================

def model_score(model_name, already_loaded, switch_cost, gpu_mem_required, gpu_total, gpu_available):
    mem_pressure = gpu_mem_required / gpu_available
    switch_component = 0.0 if already_loaded else switch_cost
    return (switch_component + mem_pressure) / 2


def select_best_model(models, current_loaded, freq_usage,
                      quality_req, latency_req, gpu_available, gpu_total):

    w_q_slm, w_l_slm, w_s_slm = compute_weights_slm(
        quality_req, latency_req, gpu_available, gpu_total
    )
    w_q_llm, w_l_llm = compute_weights_llm(quality_req, latency_req)

    utilities = {}

    for m, info in models.items():

        already_loaded = (m in current_loaded)
        switch_cost = 1.0 if not already_loaded else 0.0

        is_gpt = "gpt" in m.lower()

        if not is_gpt:

            S_m = model_score(
                m,
                already_loaded,
                switch_cost,
                info["mem"],
                gpu_total,
                gpu_available
            )

            U = 2 * w_q_slm * info["Q"] - w_l_slm * info["L"] - w_s_slm * S_m

        else:

            U = 2 * w_q_llm * info["Q"] - w_l_llm * info["L"]

        utilities[m] = U

    best_model = max(utilities, key=utilities.get)
    print(best_model)

    if best_model not in current_loaded:

        if len(current_loaded) > 0:
            least_used = min(freq_usage, key=freq_usage.get)
            current_loaded.remove(least_used)

        current_loaded.add(best_model)
        freq_usage[best_model] = 0

    freq_usage[best_model] += 1

    return best_model, utilities, (w_q_slm, w_l_slm, w_s_slm), (w_q_llm, w_l_llm)


# =========================================================
# MAIN LOOP
# =========================================================

if __name__ == "__main__":

    theta_qs = [0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    theta_ls = [1,2,3,4,5,6,7,8,9,10,11]
    gpu_total = 22.0

    Q_samples = bert_vector_pred
    L_samples = latency_vector_pred

    Q_samples_test = bert_vector_test
    L_samples_test = latency_vector_test

    for theta_q in theta_qs:
        for theta_l in theta_ls:

            decisions = []

            for i in range(300):

                s = user_reqirments[i]

                models = {
                    "Llama3_1b": {"Q": Q_samples[i, 0], "L": L_samples[i, 0], "mem": 4.5},
                    "Llama3_3b": {"Q": Q_samples[i, 1], "L": L_samples[i, 1], "mem": 6.5},
                    "Qwen4b": {"Q": Q_samples[i, 2], "L": L_samples[i, 2], "mem": 6.5},
                    "Llama3_8b": {"Q": Q_samples[i, 3], "L": L_samples[i, 3], "mem": 11},
                    "GPT-4o-mini": {"Q": Q_samples[i, 4], "L": L_samples[i, 4], "mem": 26.0},
                    "GPT-5o-mini": {"Q": Q_samples[i, 5], "L": L_samples[i, 5], "mem": 29.0}
                }

                models_test = {
                    "Llama3_1b": {"Q": Q_samples_test[i, 0], "L": L_samples_test[i, 0], "mem": 4.5},
                    "Llama3_3b": {"Q": Q_samples_test[i, 1], "L": L_samples_test[i, 1], "mem": 6.5},
                    "Qwen4b": {"Q": Q_samples_test[i, 2], "L": L_samples_test[i, 2], "mem": 6.5},
                    "Llama3_8b": {"Q": Q_samples_test[i, 3], "L": L_samples_test[i, 3], "mem": 11},
                    "GPT-4o-mini": {"Q": Q_samples_test[i, 4], "L": L_samples_test[i, 4], "mem": 26.0},
                    "GPT-5o-mini": {"Q": Q_samples_test[i, 5], "L": L_samples_test[i, 5], "mem": 29.0}
                }

                freq_usage = {
                    "Llama3_1b": 5,
                    "Llama3_3b": 2,
                    "Llama3_8b": 4,
                    "Qwen4b": 3,
                    "GPT-4o-mini": 4,
                    "GPT-5o-mini": 4
                }

                filtered_models = {
                    m: info for m, info in models.items()
                    if info["Q"] >= theta_q and info["L"] <= theta_l
                }

                current_loaded = {"Llama3_1b"}

                gpu_available = s["gpu_available"]

                if not filtered_models:

                    decisions.append({
                        "sample_index": i,
                        "models": models,
                        "models_gt": models_test,
                        "best_model": "",
                        "utilities": [],
                        "weights_slm": [],
                        "weights_llm": [],
                        "current_loaded": list(current_loaded),
                        "gpu_available": gpu_available
                    })

                else:

                    best_model, utilities, weights_slm, weights_llm = select_best_model(
                        models=filtered_models,
                        current_loaded=current_loaded,
                        freq_usage=freq_usage,
                        quality_req=s["quality_req"],
                        latency_req=s["latency_req"],
                        gpu_available=gpu_available,
                        gpu_total=gpu_total
                    )

                    decisions.append({
                        "sample_index": i,
                        "models": models,
                        "models_gt": models_test,
                        "best_model": best_model,
                        "utilities": utilities,
                        "weights_slm": weights_slm,
                        "weights_llm": weights_llm,
                        "current_loaded": list(current_loaded),
                        "gpu_available": gpu_available
                    })

            output_file = f"analysis/exp1_change_theta_q_theta_l/q_{theta_q}_l_{theta_l}_decisions_output.npz"

            np.savez(output_file, decisions=np.array(decisions, dtype=object))