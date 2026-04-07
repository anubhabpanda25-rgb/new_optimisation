import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog
import pandas as pd

st.set_page_config(layout="wide")

st.title("📊 Cost Minimization with Risk Constraint")

# -----------------------------
# DEMAND
# -----------------------------
st.header("Demand Input")
D = st.number_input("Enter Total Demand", value=1000)

# -----------------------------
# SUPPLIER INPUT (COLUMNS)
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
# SOLVER FUNCTION
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

    res = linprog(c=cost, A_ub=A_ub, b_ub=b_ub, bounds=bounds)

    return res

# -----------------------------
# CURRENT SOLUTION
# -----------------------------
res = solve_model(R_max)

if res.success:
    x = res.x
    C_star = np.dot(cost, x)
    R_star = np.dot(risk, x) / D
else:
    st.error("No feasible solution for selected risk level")
    st.stop()

# -----------------------------
# GENERATE TRADE-OFF DATA
# -----------------------------
cost_list = []
risk_list = []

risk_levels = np.linspace(0.1, 0.8, 30)

for r_lim in risk_levels:
    res_temp = solve_model(r_lim)

    if res_temp.success:
        x_temp = res_temp.x
        cost_val = np.dot(cost, x_temp)
        risk_val = np.dot(risk, x_temp) / D

        cost_list.append(cost_val)
        risk_list.append(risk_val)

# -----------------------------
# CLEAN + PARETO FILTERING
# -----------------------------
points = list(set(zip(cost_list, risk_list)))  # remove duplicates
points = sorted(points)  # sort by cost

pareto = []
for c, r in points:
    if not pareto or r < pareto[-1][1]:
        pareto.append((c, r))

if len(pareto) > 1:
    cost_p, risk_p = zip(*pareto)
else:
    cost_p, risk_p = cost_list, risk_list

# -----------------------------
# GRAPH
# -----------------------------
st.header("Cost vs Risk Pareto Curve")

fig, ax = plt.subplots()

# Pareto curve
ax.plot(cost_p, risk_p, marker='o', label="Pareto Curve")

# Selected solution
ax.scatter(C_star, R_star, color='red', s=120, label="Selected Solution")

ax.set_xlabel("Cost")
ax.set_ylabel("Risk")
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
# OUTPUT
# -----------------------------
st.header("Results")

st.subheader("Allocation")
for i, s in enumerate(suppliers):
    st.write(f"Supplier {s}: {x[i]:.2f}")

st.subheader("Metrics")
st.write(f"Total Cost: {C_star:.2f}")
st.write(f"Average Risk: {R_star:.4f}")
st.write(f"Risk Limit Used: {R_max}")

