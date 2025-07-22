
import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm, t
import math

st.set_page_config(page_title="VaR & CVaR with Stress Tests", layout="wide")
st.title("🧪 Stress-Enhanced VaR & CVaR Risk Analyzer")

# --- Sidebar Options ---
st.sidebar.header("Upload CombinedPrices.csv")
uploaded_file = st.sidebar.file_uploader("CSV file with Date, Stocks, FVX, SP500", type="csv")

confLevel = st.sidebar.slider("Confidence Level", min_value=0.90, max_value=0.99, value=0.99, step=0.01)
use_gbm = st.sidebar.checkbox("Use GBM Monte Carlo Simulation", value=True)
fat_tails = st.sidebar.checkbox("Use t-distribution for Fat Tails", value=False)

st.sidebar.subheader("💥 Stress Test Shocks")
shock_fvx = st.sidebar.slider("Shock to FVX (%)", min_value=-5.0, max_value=5.0, value=0.0, step=0.25)
shock_sp500 = st.sidebar.slider("Shock to SP500 (%)", min_value=-5.0, max_value=5.0, value=0.0, step=0.25)
shock_scenarios = {
    "FVX": shock_fvx / 100,
    "SP500": shock_sp500 / 100
}

# --- Functions ---
def calculateVaR(risk, confLevel, principal=1, numMonths=1):
    vol = math.sqrt(risk)
    return abs(principal * norm.ppf(1 - confLevel) * vol * math.sqrt(numMonths))

def calculateCVaR(port_returns, confLevel):
    var_threshold = np.percentile(port_returns, (1 - confLevel) * 100)
    return abs(port_returns[port_returns <= var_threshold].mean())

def apply_stress(factor_df, shocks):
    df = factor_df.copy()
    for factor, shock in shocks.items():
        if factor in df.columns:
            df[factor] += shock
    return df

def gbm_simulation(S0, mu, sigma, days, n_simulations=10000, non_normal=False):
    dt = 1 / 252
    if non_normal:
        Z = t.rvs(df=5, size=(n_simulations, days))
    else:
        Z = np.random.normal(0, 1, size=(n_simulations, days))
    return S0 * np.exp(np.cumsum((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z, axis=1))

# --- Main Workflow ---
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, sep=";")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date")

    for col in df.columns[1:]:
        df[col] = df[col].astype(str).str.replace(",", ".").str.replace(" ", "").str.replace(u'\xa0', '').astype(float)

    factorNames = st.sidebar.multiselect("Select factorNames", [col for col in df.columns if col not in ["Date"]], default=["FVX", "SP500"])
    factorNames.append("Intercept")
    stockNames = [col for col in df.columns if col not in ["Date"] + factorNames]

    returns = df.drop(columns=["Date"]).pct_change().dropna()
    returns["Intercept"] = 1
    stockReturns = returns[stockNames]
    factorReturns = returns[factorNames]

    weights = np.array([1.0 / len(stockNames)] * len(stockNames))
    port_returns = stockReturns.dot(weights)

    hist_var = abs(np.percentile(port_returns, (1 - confLevel) * 100))
    hist_cvar = calculateCVaR(port_returns, confLevel)

    vc_risk = np.dot(np.dot(weights, stockReturns.cov()), weights.T)
    vc_var = calculateVaR(vc_risk, confLevel)

    if use_gbm:
        mu = stockReturns.mean()
        sigma = stockReturns.std()
        sim = gbm_simulation(1, mu.mean(), sigma.mean(), days=252, non_normal=fat_tails)
        sim_port = sim[:, -1] - 1  # return over 1 year
        mc_var = abs(np.percentile(sim_port, (1 - confLevel) * 100))
        mc_cvar = calculateCVaR(sim_port, confLevel)
    else:
        mc_var = "-"
        mc_cvar = "-"

    xData = factorReturns
    modelCoeffs = []
    for oneStock in stockNames:
        yData = stockReturns[oneStock]
        model = sm.OLS(yData, xData).fit()
        coeffs = list(model.params)
        coeffs.append(np.std(model.resid, ddof=1))
        modelCoeffs.append(coeffs)

    modelCoeffs = pd.DataFrame(modelCoeffs, columns=factorNames + ["ResidVol"])
    modelCoeffs["Names"] = stockNames

    factorCov = factorReturns[[col for col in factorNames if col != "Intercept"]].cov()
    B_factors = modelCoeffs[[col for col in factorNames if col != "Intercept"]]
    reconstructedCov = np.dot(np.dot(B_factors, factorCov), B_factors.T)
    systemicRisk = np.dot(np.dot(weights, reconstructedCov), weights.T)
    idiosyncraticRisk = sum(modelCoeffs["ResidVol"] ** 2 * weights ** 2)
    factor_risk = systemicRisk + idiosyncraticRisk
    factor_var = calculateVaR(factor_risk, confLevel)

    results_df = pd.DataFrame([
        ["Historical", hist_var, hist_cvar],
        ["Variance-Covariance", vc_var, "-"],
        ["Monte Carlo", mc_var, mc_cvar],
        ["Factor Model", factor_var, "-"]
    ], columns=["Method", "VaR (%)", "CVaR (%)"])

    st.subheader("📊 Risk Summary Table")
    st.dataframe(results_df)

    st.download_button("⬇️ Download Results", data=results_df.to_csv(index=False).encode("utf-8"),
                       file_name="var_cvar_summary.csv")

    # Stress Test Visual
    st.subheader("📉 Stress Test Impact on Factor Distributions")
    stressed = apply_stress(factorReturns[[col for col in ["FVX", "SP500"] if col in factorReturns.columns]], shock_scenarios)

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for i, factor in enumerate(stressed.columns):
        sns.kdeplot(factorReturns[factor], ax=ax[i], label="Original", linewidth=2)
        sns.kdeplot(stressed[factor], ax=ax[i], label="Stressed", linestyle="--", linewidth=2)
        ax[i].set_title(f"{factor} Distribution")
        ax[i].legend()
    st.pyplot(fig)

else:
    st.info("⬅️ Upload CombinedPrices.csv and select options.")
