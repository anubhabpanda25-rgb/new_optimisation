import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog
import pandas as pd

st.set_page_config(layout="wide")

st.title("📊 Cost Optimization with Risk Sensitivity")

# -----------------------------
# STEP 1: SUPPLIER SELECTION
# -----------------------------
st.header("Step 1: Select Suppliers")

suppliers = ["A", "B", "C"]
selected = []

for s in suppliers:
    val = st.checkbox(f"Use Supplier {s}", value=True)
    selected.append(1 if val else 0)

# -----------------------------
# STEP 2: INPUT DATA
# -----------------------------
st.header("Step 2: Supplier Inputs")

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

# -----------------------------
# DEMAND
# -----------------------------
st.header("Demand")
D = st.number_input("Total Demand", value=1000)

# -----------------------------
# STEP 3: LAMBDA INPUT
# -----------------------------
st.header("Step 3: Risk Preference")

lam = st.slider("Risk Tolerance (λ)", 0.0, 1.0, 0.5)

# Convert λ → risk limit
def lambda_to_risk(l):
    return 0.1 + 0.8 * l

R_max = lambda_to_risk(lam)

st.write(f"Allowed Risk Level: {R_max:.2f}")

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
# MAIN SOLUTION
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
# RESULTS
# -----------------------------
st.header("Optimal Solution")

st.subheader("Allocation")
for i, s in enumerate(suppliers):
    st.write(f"Supplier {s}: {x[i]:.2f}")

st.subheader("Metrics")
st.write(f"Cost: {C_star:.2f}")
st.write(f"Risk: {R_star:.4f}")

# -----------------------------
# SENSITIVITY ANALYSIS
# -----------------------------
st.header("Sensitivity Analysis (λ ± 0.10)")

lambda_values = [
    max(0, lam - 0.10),
    lam,
    min(1, lam + 0.10)
]

results = []

for l in lambda_values:
    R_test = lambda_to_risk(l)
    res_temp = solve_model(R_test)

    if res_temp.success:
        x_temp = res_temp.x
        cost_val = np.dot(cost, x_temp)
        risk_val = np.dot(risk, x_temp) / D
        status = "Feasible"
    else:
        cost_val = None
        risk_val = None
        status = "Infeasible"

    results.append({
        "Lambda": round(l, 2),
        "Risk Limit": round(R_test, 2),
        "Cost": cost_val,
        "Risk": risk_val,
        "Status": status
    })

df = pd.DataFrame(results)

st.dataframe(df)
