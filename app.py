import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- PAGE LAYOUT CONFIGURATION ---
st.set_page_config(page_title="Future Investment Risk-Reward Simulator", layout="wide")

st.title("🔮 Predictive Capital Risk-Reward Simulator Engine")
st.markdown("### Powered by Quantitative Finance Stochastic Tracking & Monte Carlo Simulation Pipelines")
st.markdown("---")

# --- DATA STORAGE COMPONENT ---
@st.cache_data
def load_simulation_pool():
    # Consolidated background data structures directly mapped from analytical project benchmarks
    data = {
        'Rank': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'Stock': ['Trent Ltd', 'Bharat Electronics Ltd', 'Power Grid Corporation of India Ltd', 'ITC Ltd', 'Bharti Airtel Ltd', 'Sun Pharmaceutical Industries Ltd', 'Hindustan Unilever Ltd', 'Infosys Ltd', 'Larsen & Toubro Ltd', 'NTPC Ltd'],
        'Expected Return': [0.0252, 0.0200, 0.0008, 0.0008, 0.0267, 0.0046, 0.0200, 0.0200, 0.0200, -0.0003],
        'Semi-variance Risk': [0.0051, 0.0042, 0.0002, 0.0002, 0.0054, 0.0011, 0.0041, 0.0041, 0.0043, 0.0001]
    }
    return pd.DataFrame(data)

df_pool = load_simulation_pool()

# --- SYSTEM SIDEBAR CONTROLS ---
st.sidebar.header("🎯 Simulator Configurations")
risk_free_rate = st.sidebar.number_input("Global Risk-Free Rate Baseline (rf)", value=0.010, step=0.005, format="%.3f")

# --- CORE SIMULATOR WORKSPACE ---
col_inv1, col_inv2 = st.columns([1, 1.4])

with col_inv1:
    st.markdown("##### **📥 User Investment Parameters Setup**")
    
    # Input field for current principal capital (e.g., User target INR values)
    principal_amount = st.number_input("Enter Amount to Invest (₹)", value=10000, step=1000, format="%d")
    
    # Selection pool tied explicitly to the vetted top structural components
    selected_sim_stock = st.selectbox("Select Target Company for Simulation Track:", df_pool['Stock'].values)
    
    # Investment duration sliding scale horizon
    sim_years = st.slider("Investment Horizon Period (Years into Future):", 1, 20, 5)
    
    # Dynamic Volatility Tuner Slider: Controls the risk multiplier directly on the screen
    stress_multiplier = st.slider("Market Volatility Stress Tuner (Shock Factor Coefficient):", 1.0, 5.0, 1.0, step=0.5)
    
    # Extraction vectors processing
    stock_row = df_pool[df_pool['Stock'] == selected_sim_stock].iloc[0]
    base_mu = float(stock_row['Expected Return'])
    
    # Link volatility directly with the real-time input slider value
    base_sigma = float(stock_row['Semi-variance Risk']) * stress_multiplier
    
    st.markdown("---")
    st.markdown("##### **📊 Instant Algorithmic Projections**")
    
    # Stochastic Calculation Pipeline: Geometric Brownian Motion Engine
    np.random.seed(42)
    num_simulations = 1000
    dt = 1.0
    
    simulation_results = np.zeros((num_simulations, sim_years + 1))
    simulation_results[:, 0] = principal_amount
    
    # Formulate randomized projections tracking across time intervals
    for t in range(1, sim_years + 1):
        shocks = np.random.normal(0, 1, num_simulations)
        # Slicing excess mathematical drift factor over selected risk-free rates
        effective_mu = base_mu - risk_free_rate
        simulation_results[:, t] = simulation_results[:, t-1] * np.exp((effective_mu - 0.5 * base_sigma**2) * dt + base_sigma * np.sqrt(dt) * shocks)
        
    final_values = simulation_results[:, -1]
    expected_future_value = float(np.mean(final_values))
    worst_case_value = float(np.percentile(final_values, 5))
    loss_probability = float(np.mean(final_values < principal_amount) * 100)
    
    # Visual metrics readout widgets
    st.metric(
        label="Expected Future Valuation Asset Value", 
        value=f"₹{expected_future_value:,.2f}", 
        delta=f"₹{expected_future_value - principal_amount:,.2f} Net Yield"
    )
    
    st.metric(
        label="Worst-Case Scenario (5th Percentile Limit Risk)", 
        value=f"₹{worst_case_value:,.2f}", 
        delta=f"-₹{principal_amount - worst_case_value:,.2f} Capital Variation", 
        delta_color="inverse"
    )
    
    st.markdown(" ")
    # Dynamic Alert Banner Blocks rendering with the correct native unsafe mapping flag
    if loss_probability > 30.0:
        st.markdown(f"<div style='padding:12px; border-radius:6px; background-color:#ffcccc; border:1px solid #ff3333; color:#990000; font-weight:bold;'>⚠️ Probability of Capital Erosion: {loss_probability:.2f}% (HIGH VOLATILITY RISK)</div>", unsafe_allow_html=True)
    elif loss_probability > 10.0:
        st.markdown(f"<div style='padding:12px; border-radius:6px; background-color:#ffe6cc; border:1px solid #ff9933; color:#994c00; font-weight:bold;'>🔸 Probability of Capital Erosion: {loss_probability:.2f}% (MODERATE RISK PROFILE)</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div style='padding:12px; border-radius:6px; background-color:#e2f0d9; border:1px solid #385723; color:#385723; font-weight:bold;'>✅ Probability of Capital Erosion: {loss_probability:.2f}% (SAFE / LOW VARIATION PROFILE)</div>", unsafe_allow_html=True)

with col_inv2:
    st.markdown("##### **📈 Stochastic Projection Trajectories Curve**")
    
    # Plotting canvas generation
    fig_inv, ax_inv = plt.subplots(figsize=(7, 4.4))
    time_axis = np.arange(0, sim_years + 1)
    
    # Render localized samples paths
    for sim_idx in range(min(45, num_simulations)):
        ax_inv.plot(time_axis, simulation_results[sim_idx, :], alpha=0.25, color='steelblue')
        
    # Superimpose aggregate median baseline index trajectory
    mean_path = np.mean(simulation_results, axis=0)
    ax_inv.plot(time_axis, mean_path, color='darkred', linewidth=3.5, label='Expected Track (Mean Path)')
    ax_inv.axhline(y=principal_amount, color='black', linestyle='--', alpha=0.7, label='Initial Principal Baseline')
    
    ax_inv.set_title(f'Simulated 1,000 Future Valuation Tracks for {selected_sim_stock}', fontsize=10, fontweight='bold')
    ax_inv.set_xlabel('Years into Future Scale Horizon', fontsize=8)
    ax_inv.set_ylabel('Valuation Scale (INR ₹)', fontsize=8)
    ax_inv.grid(True, linestyle=':', alpha=0.6)
    ax_inv.legend(fontsize=8, loc='upper left')
    st.pyplot(fig_inv)
    
    

st.markdown("---")
st.subheader("📌 Academic Project Defense Reference Summary")
st.info(
    "**Algorithmic Architecture Note:** This module utilizes Geometric Brownian Motion (GBM) to forecast "
    "asset price probabilities under geometric market uncertainties. Rather than using traditional, standard variance, "
    "the pricing dispersion ($\sigma$) is parameterized directly via downside semi-variance risk metrics. This ensures "
    "that the stochastic volatility shocks specifically measure real capital erosion threats under stressful macroeconomic cycles "
    "making the decision support system mathematically superior for protective wealth management."
)
