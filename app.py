import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog
import pandas as pd

st.set_page_config(layout="wide")

st.title("📊 Cost Optimization with Risk Control")

# -----------------------------
# DEMAND
# -----------------------------
st.header("Demand Input")
D = st.number_input("Enter Total Demand", value=1000)

# -----------------------------
# SUPPLIER INPUT
# -----------------------------
st.header("Supplier Details")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Supplier A")
    cost_A = st.number_input("Cost A", value=10)
    risk_A = st.number_input("Risk A", value=0.2)
    cap_A = st.number_input("Capacity A", value=500)

with col2:
    st.subheader("Supplier B")
    cost_B = st.number_input("Cost B", value=12)
    risk_B = st.number_input("Risk B", value=0.5)
    cap_B = st.number_input("Capacity B", value=700)

with col3:
    st.subheader("Supplier C")
    cost_C = st.number_input("Cost C", value=11)
    risk_C = st.number_input("Risk C", value=0.3)
    cap_C = st.number_input("Capacity C", value=400)

cost = np.array([cost_A, cost_B, cost_C])
risk = np.array([risk_A, risk_B, risk_C])
capacity = np.array([cap_A, cap_B, cap_C])

suppliers = ["A", "B", "C"]

# -----------------------------
# SUPPLIER SELECTION
# -----------------------------
st.header("Supplier Strategy")

mode = st.radio("Select sourcing strategy",
                ["Use All Suppliers", "Select Suppliers Manually"])

selected = [1, 1, 1]

if mode == "Select Suppliers Manually":
    selected = []
    for s in suppliers:
        val = st.checkbox(f"Use Supplier {s}", value=True)
        selected.append(1 if val else 0)

# -----------------------------
# RISK APPETITE
# -----------------------------
st.header("Risk Appetite")

risk_level = st.selectbox("Select risk level",
                         ["Low Risk", "Medium Risk", "High Risk"])

if risk_level == "Low Risk":
    R_max = 0.25
elif risk_level == "Medium Risk":
    R_max = 0.40
else:
    R_max = 0.60

# -----------------------------
# PREFERENCE SLIDER
# -----------------------------
st.header("Preference")
lam = st.slider("Cost vs Risk Preference (λ)", 0.0, 1.0, 0.5)

# -----------------------------
# SOLVER FUNCTION
# -----------------------------
def solve_model(R_limit):

    A_ub = [
        [-1, -1, -1],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        risk.tolist()
    ]

    b_ub = [
        -D,
        capacity[0] * selected[0],
        capacity[1] * selected[1],
        capacity[2] * selected[2],
        R_limit * D
    ]

    bounds = [(0, None)] * 3

    return linprog(c=cost, A_ub=A_ub, b_ub=b_ub, bounds=bounds)

# -----------------------------
# CURRENT SOLUTION
# -----------------------------
res = solve_model(R_max)

if res.success:
    x = res.x
    C_star = np.dot(cost, x)
    R_star = np.dot(risk, x) / D
else:
    st.error("No feasible solution")
    st.stop()

# -----------------------------
# GENERATE MANY SOLUTIONS
# -----------------------------
cost_list = []
risk_list = []

risk_levels = np.linspace(0.05, 1.0, 60)

for r_lim in risk_levels:
    res_temp = solve_model(r_lim)

    if res_temp.success:
        x_temp = res_temp.x
        cost_list.append(float(np.dot(cost, x_temp)))
        risk_list.append(float(np.dot(risk, x_temp) / D))

# -----------------------------
# PARETO FILTER
# -----------------------------
points = list(set(zip(cost_list, risk_list)))
points = sorted(points, key=lambda x: x[0])

pareto = []
min_risk = float('inf')

for c, r in points:
    if r < min_risk:
        pareto.append((c, r))
        min_risk = r

cost_p, risk_p = zip(*pareto)

# -----------------------------
# RECOMMENDED SOLUTION
# -----------------------------
C_max = max(cost_p)

scores = []
for c, r in zip(cost_p, risk_p):
    score = lam * (c / C_max) + (1 - lam) * r
    scores.append(score)

best_index = int(np.argmin(scores))
C_best = cost_p[best_index]
R_best = risk_p[best_index]

# -----------------------------
# GRAPH
# -----------------------------
st.header("Pareto Trade-off Curve")

fig, ax = plt.subplots()

# All solutions
ax.scatter(cost_list, risk_list, alpha=0.3, label="All Solutions")

# Pareto curve
ax.plot(cost_p, risk_p, color='blue', marker='o', linewidth=2, label="Pareto Frontier")

# Selected solution
ax.scatter(C_star, R_star, color='red', s=120, label="Current Solution")

# Recommended solution
ax.scatter(C_best, R_best, color='green', s=150, label="Recommended")

# Indifference line
C_vals = np.linspace(min(cost_p), max(cost_p), 100)
k = lam*(C_best/C_max) + (1-lam)*R_best

R_vals = (k - lam*(C_vals/C_max)) / (1-lam + 1e-6)

ax.plot(C_vals, R_vals, linestyle='--', color='green', label="Preference Line")

ax.set_xlabel("Cost")
ax.set_ylabel("Risk")
ax.grid(True)
ax.legend()

st.pyplot(fig)

# -----------------------------
# TABLE
# -----------------------------
df = pd.DataFrame({
    "Cost": cost_p,
    "Risk": risk_p
})

st.subheader("Pareto Data")
st.dataframe(df)

# -----------------------------
# RESULTS
# -----------------------------
st.header("Results")

st.subheader("Allocation")
for i, s in enumerate(suppliers):
    st.write(f"Supplier {s}: {x[i]:.2f}")

st.subheader("Metrics")
st.write(f"Total Cost: {C_star:.2f}")
st.write(f"Average Risk: {R_star:.4f}")

# -----------------------------
# INSIGHT PANEL
# -----------------------------
st.header("AI Recommendation")

st.write(f"Recommended Cost: {C_best:.2f}")
st.write(f"Recommended Risk: {R_best:.4f}")

if lam > 0.7:
    st.success("You are prioritizing cost heavily → Cheapest solution recommended.")
elif lam < 0.3:
    st.success("You are prioritizing risk → Safer suppliers preferred.")
else:
    st.success("Balanced strategy → Optimal cost-risk mix.")
