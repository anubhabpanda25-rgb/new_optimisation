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
# LAMBDA (RISK CONTROL)
# -----------------------------
st.header("Risk Preference (λ)")

lam = st.slider("Risk Tolerance (λ)", 0.0, 1.0, 0.5)

# Map λ → Risk Limit
R_max = 0.1 + 0.8 * lam

st.write(f"Allowed Risk Level: {R_max:.2f}")

# -----------------------------
# SOLVER FUNCTION (COST MIN)
# -----------------------------
def solve_model(R_limit):

    A_ub = [
        [-1, -1, -1],  # Demand
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        risk.tolist()  # Risk constraint
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
    st.error("No feasible solution for this risk level")
    st.stop()

# -----------------------------
# GENERATE TRADE-OFF CURVE
# -----------------------------
cost_list = []
risk_list = []

risk_levels = np.linspace(0.05, 0.9, 60)

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
# GRAPH
# -----------------------------
st.header("Cost vs Risk Trade-off")

fig, ax = plt.subplots()

# All solutions
ax.scatter(cost_list, risk_list, alpha=0.3, label="All Solutions")

# Pareto curve
ax.plot(cost_p, risk_p, color='blue', marker='o', linewidth=2, label="Pareto Frontier")

# Current solution
ax.scatter(C_star, R_star, color='red', s=120, label="Selected (λ-based)")

ax.set_xlabel("Cost")
ax.set_ylabel("Risk")
ax.set_title("Cost vs Risk Trade-off")
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
st.header("Insights")

if lam < 0.3:
    st.success("Low risk preference → safer suppliers selected, higher cost")
elif lam > 0.7:
    st.success("High risk tolerance → cheaper suppliers selected")
else:
    st.success("Balanced approach between cost and risk")
