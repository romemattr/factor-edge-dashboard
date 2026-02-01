import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
import statsmodels.api as sm
from datetime import datetime, timedelta

# ============================================================================
# APP CONFIGURATION & STATE
# ============================================================================
st.set_page_config(layout="wide", page_title="Factor Edge Dashboard")

if 'clusters' not in st.session_state:
    st.session_state.clusters = {
    'AI Compute': ['NVDA', 'AVGO', 'AMD', 'TSM', 'ARM', "MU", "COHR", "ASML"],
    'SMH': ['SMH'],
    'AI Power':['GEV', "OKLO", "CEG", "VST", "EQT"],
    'AI Infrastructure': ['ANET', 'CRDO', 'LITE', 'APH', 'CIEN', 'AMAT'],
    'Mag 8': ['MSFT', 'GOOGL', 'AMZN', 'META', 'AAPL', 'TSLA', "AVGO", "NVDA"],
    'SaaS': ['CRM', 'NOW', 'MSFT', 'DDOG', 'SNOW', "IGV", 'WDAY', 'ADBE', 'CNSWF'],
    'HPC': ['CIFR', 'WULF', 'CORZ', 'IREN', 'HUT', "APLD"],
    'Copper Miners': ['COPX', 'ICOP'],
    'Dr. Copper': ['CPER'],
    'XME': ['HL', 'UEC', 'CDE', 'AA', 'RGLD', 'FCX','BTU', 'NEM', 'HCC'],
    'Cyclical Materials': ['XLB'],
    'REMX':['ALB', 'MP', 'USAR'],
    'Gold':['GLD'],
    'Gold Miners':['GDX', 'GDXJ'],
    'Silver':['SLV'],
    'Silver Miners':["SIL", 'SILJ'],
    'XLE': ['XLE', 'OIH', 'XOP', 'SLB'],
    'XLF': ['JPM', 'BAC', 'GS', 'MS', 'XLF'],
    'Defensives': ['XLP', 'XLU', 'WMT', 'BRK-B', "COST"],
    'Rates': ['TLT', 'TYA'],
    'BTC': ['IBIT', 'ETHA'],
    'IBB': ["BBC", "XBI", 'IBB', 'XLV'],
    'PE Mngrs': ['BX', 'BN', 'KKR', 'CG', 'TPG'],
    'EFA':['EFA', 'VEA', 'EWJ', "EWY"],
    'EEM':['EEM', 'VWO', "EPOL", "EWZ", 'KWEB', "FXI", 'CQQQ'],
    'Beta':["QQQ", "XLK"],
    'Small':['IWM'],
    'US Value': ['VTV'],
    'US Growth': ['VUG'],
    'High DIV': ['LYB', "ARE", "CAG","MO", "PFE", "VZ", "KHC"],
    'Aerospace':["ITA", "HEI", "TDG", "GE", 'GD', "HII", "RTX", "LMT"],
    'COAL':['HCC', 'AMR', 'BTU', 'METC', 'NRP', 'METC', 'GLNCY'],
    'RiskParity':['ALLW']
}

# ============================================================================
# SIDEBAR: DATA CONTROLS
# ============================================================================
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Factor Management
    with st.expander("Edit Factor Clusters"):
        for cluster, tickers in st.session_state.clusters.items():
            new_list = st.text_input(f"Edit {cluster}", value=", ".join(tickers))
            st.session_state.clusters[cluster] = [t.strip() for t in new_list.split(",")]
        
        new_f_name = st.text_input("New Factor Name")
        new_f_ticks = st.text_input("Tickers (comma separated)")
        if st.button("Add Factor"):
            st.session_state.clusters[new_f_name] = [t.strip() for t in new_f_ticks.split(",")]

    # Global Data Fetch
    all_tickers = list(set([t for sub in st.session_state.clusters.values() for t in sub]))
    all_tickers.extend(['SPY', '^VIX'])
    
    # Change this line in your sidebar data fetch:
    data = yf.download(all_tickers, period='2y', progress=False)['Close']

    # Add this line immediately after to ensure no multi-index issues:
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(-1)

# ============================================================================
# CALCULATIONS
# ============================================================================
def get_factor_ts(data, clusters):
    ts = pd.DataFrame()
    for name, tickers in clusters.items():
        # Validating tickers exist in downloaded data
        valid_ticks = [t for t in tickers if t in data.columns]
        ts[name] = data[valid_ticks].pct_change().mean(axis=1)
    return ts

factor_returns = get_factor_ts(data, st.session_state.clusters)

# ============================================================================
# TABS
# ============================================================================
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📈 Performance", 
    "🔥 Correlation", 
    "🌋 Macro (VIX)", 
    "🔄 Rotation", 
    "⚖️ Long-Short",
    "🧪 Backtester",
    "📊 Comparison"
])

# --- TAB 1: PERFORMANCE & REGIME ---
with tab1:
    # 1. Calculate Core Metrics for Display
    # Using 20D as the primary trend window
    rets_20d = factor_returns.tail(20).sum() * 100
    rets_5d = factor_returns.tail(5).sum() * 100
    rets_1d = factor_returns.tail(1).sum() * 100
    # Calculate Z-Scores for the Regime Logic
    # (Current 20D return vs historical 20D rolling mean)
    roll_mean = factor_returns.rolling(20).sum().mean()
    roll_std = factor_returns.rolling(20).sum().std()
    z_scores = (rets_20d/100 - roll_mean) / roll_std

    # 2. Simple Regime Logic (Adjusted for your factors)
    current_regime = "MIXED / TRANSITION"
    regime_color = "#FFFF00"
    
    # Check for AI Tech Momentum
    if "AI Tech" in z_scores and z_scores["AI Tech"] > 1.0:
        current_regime = "AI TECH MOMENTUM"
        regime_color = "#00FF00"
    # Check for Defensive flight
    elif "Defensives" in z_scores and z_scores["Defensives"] > 0.5:
        current_regime = "RISK-OFF / DEFENSIVE"
        regime_color = "#FF0000"

    # Header Display
    st.markdown(f"""
        <div style="background-color:#1e293b; padding:20px; border-radius:10px; border-left: 10px solid {regime_color};">
            <h2 style="margin:0;">Current Regime: {current_regime}</h2>
            <p style="margin:0; color:#cbd5e1;">Updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
        </div>
    """, unsafe_allow_html=True)

    st.write("") # Spacer

    # 3. Layout: Heatmap and Stats Table
    col_p1, col_p2 = st.columns([2, 1])

    with col_p1:
        st.subheader("Factor Performance Heatmap (20D)")
        # Sort for the chart
        sorted_rets = rets_20d.sort_values(ascending=True)
        colors = ['#22c55e' if x > 0 else '#ef4444' for x in sorted_rets.values]
        
        fig_perf = go.Figure(go.Bar(
            x=sorted_rets.values,
            y=sorted_rets.index,
            orientation='h',
            marker_color=colors,
            text=[f"{x:.1f}%" for x in sorted_rets.values],
            textposition='outside'
        ))
        
        fig_perf.update_layout(
            template="plotly_dark",
            height=700,
            xaxis_title="Cumulative 20D Return (%)",
            yaxis_title="Factor Cluster",
            margin=dict(l=20, r=20, t=20, b=20)
        )
        st.plotly_chart(fig_perf, use_container_width=True)

    with col_p2:
        st.subheader("Momentum Scores")
        # Create a clean table for the sidebar of the tab
        # Enhanced Table with 1D%
        perf_df = pd.DataFrame({
        "Factor": rets_20d.index,
        "1D %": rets_1d.values.round(2),
        "5D %": rets_1d.values.round(2),
        "20D %": rets_20d.values.round(2),
        "Z-Score": z_scores.values.round(2)
        }).sort_values("20D %", ascending=False)

        # Apply styling to the dataframe
        st.dataframe(
            perf_df.style.background_gradient(subset=["Z-Score"], cmap="RdYlGn"),
            height=700,
            use_container_width=True
        )
    

# --- TAB 2: CORRELATION HEATMAP ---
with tab2:
    st.subheader("Factor Correlation Matrix")
    period_map = {"1M": 21, "3M": 63, "6M": 126, "1Y": 252}
    lookback = st.select_slider("Lookback Period", options=["1M", "3M", "6M", "1Y"])
    
    corr_df = factor_returns.tail(period_map[lookback]).corr()
    
    fig_corr = px.imshow(
        corr_df,
        text_auto=".2f",
        color_continuous_scale="RdYlGn",
        aspect="auto",
        title=f"Factor Correlations ({lookback})"
    )
    st.plotly_chart(fig_corr, use_container_width=True)

# --- TAB 3: VIX OVERLAY ---
with tab3:
    st.subheader("S&P 500 vs VIX Fear Index")
    fig_vix = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig_vix.add_trace(go.Scatter(x=data.index, y=data['SPY'], name="S&P 500"), secondary_y=False)
    fig_vix.add_trace(go.Scatter(x=data.index, y=data['^VIX'], name="VIX", opacity=0.3, fill='tozeroy'), secondary_y=True)
    
    fig_vix.update_layout(template="plotly_dark")
    st.plotly_chart(fig_vix, use_container_width=True)

# --- TAB 4: FACTOR ROTATION QUADRANT ---
with tab4:
    st.subheader("Factor Rotation Quadrant")
    st.markdown("Visualize factor momentum and identify rotation opportunities")

    # 1. Selection for timeframes (as seen in your reference image)
    col_t1, col_t2 = st.columns(2)
    x_axis_term = col_t1.selectbox("X-Axis (Short-term):", ["1D", "5D", "20D"], index=1)
    y_axis_term = col_t2.selectbox("Y-Axis (Medium-term):", ["5D", "20D", "60D"], index=1)

    # 2. Map data based on selection
    # Assuming factor_returns is daily pct change
    term_map = {"1D": 1, "5D": 5, "20D": 20, "60D": 60}
    
    x_data = factor_returns.tail(term_map[x_axis_term]).sum() * 100
    y_data = factor_returns.tail(term_map[y_axis_term]).sum() * 100

    quad_df = pd.DataFrame({
        'Factor': x_data.index,
        'X': x_data.values,
        'Y': y_data.values
    })

    # 3. Define Quadrants
    def get_quadrant(row):
        if row['X'] > 0 and row['Y'] > 0: return 'Leaders'
        if row['X'] < 0 and row['Y'] > 0: return 'Fading'
        if row['X'] < 0 and row['Y'] < 0: return 'Laggards'
        return 'Recovering'

    quad_df['Status'] = quad_df.apply(get_quadrant, axis=1)

    # 4. Quadrant Summary Metrics (The Top Row)
    q_counts = quad_df['Status'].value_counts()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🚀 Leaders", q_counts.get('Leaders', 0))
    c2.metric("📉 Fading", q_counts.get('Fading', 0))
    c3.metric("🔄 Recovering", q_counts.get('Recovering', 0))
    c4.metric("⚠️ Laggards", q_counts.get('Laggards', 0))

    # 5. The Main Scatter Plot
    # Color mapping to match your UI reference
    color_map = {
        'Leaders': '#22c55e',    # Green
        'Fading': '#f59e0b',     # Orange
        'Recovering': '#3b82f6', # Blue
        'Laggards': '#ef4444'    # Red
    }

    fig_rot = px.scatter(
        quad_df, x='X', y='Y', text='Factor', color='Status',
        color_discrete_map=color_map,
        labels={'X': f'{x_axis_term} Return (%)', 'Y': f'{y_axis_term} Return (%)'}
    )

    # Styling to match the "Liquidation Nation" dark UI
    fig_rot.update_traces(marker=dict(size=15), textposition='top center')
    fig_rot.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.3)
    fig_rot.add_vline(x=0, line_dash="dash", line_color="white", opacity=0.3)
    
    # Add Quadrant Labels
    fig_rot.add_annotation(x=0.9, y=0.9, text="LEADERS", showarrow=False, xref="paper", yref="paper", font=dict(color="#22c55e", size=16))
    fig_rot.add_annotation(x=0.1, y=0.9, text="FADING", showarrow=False, xref="paper", yref="paper", font=dict(color="#f59e0b", size=16))
    fig_rot.add_annotation(x=0.1, y=0.1, text="LAGGARDS", showarrow=False, xref="paper", yref="paper", font=dict(color="#ef4444", size=16))
    fig_rot.add_annotation(x=0.9, y=0.1, text="RECOVERING", showarrow=False, xref="paper", yref="paper", font=dict(color="#3b82f6", size=16))

    fig_rot.update_layout(
        template="plotly_dark", height=600,
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        showlegend=False
    )
    st.plotly_chart(fig_rot, use_container_width=True)

    # 6. Bottom Summary Cards (The List View)
    st.write("---")
    sc1, sc2, sc3, sc4 = st.columns(4)
    
    for status, col in zip(['Leaders', 'Fading', 'Recovering', 'Laggards'], [sc1, sc2, sc3, sc4]):
        factors = quad_df[quad_df['Status'] == status]['Factor'].tolist()
        with col:
            st.markdown(f"**{status}**")
            if factors:
                for f in factors:
                    st.caption(f"• {f}")
            else:
                st.caption("None")

# --- TAB 5: ROLLING REGRESSION & RESIDUAL ANALYSIS ---
with tab5:
    st.subheader("Statistical Spread Analysis (Rolling Regression)")
    
    # 1. Selection & Data Prep
    all_options = list(st.session_state.clusters.keys()) + list(data.columns)
    c1, c2 = st.columns(2)
    f1 = c1.selectbox("Asset A (Dependent/Long)", all_options, index=0)
    f2 = c2.selectbox("Asset B (Independent/Short)", all_options, index=1)
    
    lookback = st.slider("Hedge Ratio Lookback (Days)", 20, 252, 60)
    
    # Helper function to get prices (averages the group if it's a cluster)
    def get_prices(name):
        if name in st.session_state.clusters:
            # We use prices for the regression to find the hedge ratio
            return data[st.session_state.clusters[name]].mean(axis=1)
        return data[name]

    price_a = get_prices(f1).dropna()
    price_b = get_prices(f2).dropna()
    df_reg = pd.concat([price_a, price_b], axis=1).dropna()
    df_reg.columns = ['y', 'x']

    # 2. Rolling OLS Regression
    # This finds the 'Beta' or Hedge Ratio dynamically over time
    def rolling_regression(df, window):
        residuals = []
        betas = []
        indices = df.index[window:]
        
        for i in range(window, len(df)):
            y_sub = df['y'].iloc[i-window:i]
            x_sub = sm.add_constant(df['x'].iloc[i-window:i])
            model = sm.OLS(y_sub, x_sub).fit()
            
            # Predict current y based on current x relationship
            current_x = [1, df['x'].iloc[i]]
            y_pred = model.predict(current_x)[0]
            residuals.append(df['y'].iloc[i] - y_pred)
            betas.append(model.params[1])
            
        return pd.Series(residuals, index=indices), pd.Series(betas, index=indices)

    resids, rolling_betas = rolling_regression(df_reg, lookback)

    # 3. Calculate Asymmetric Sigma Bands (+2 / -1 Sigma)
    # We use a rolling standard deviation of the residuals to create dynamic bands
    rolling_std = resids.rolling(window=lookback).std()
    upper_band = 2 * rolling_std
    lower_band = -1 * rolling_std

    # 4. Visualization
    fig_reg = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                            subplot_titles=("Rolling Hedge Ratio (Beta)", "Residuals with +2/-1 Sigma Bands"),
                            row_heights=[0.3, 0.7], vertical_spacing=0.1)

    # Top: Beta Chart (Shows how much of B you need to hedge A)
    fig_reg.add_trace(go.Scatter(x=rolling_betas.index, y=rolling_betas, name="Hedge Ratio (Beta)", line=dict(color='yellow')), 1, 1)

    # Bottom: Residuals with Sigma Bands
    fig_reg.add_trace(go.Scatter(x=resids.index, y=resids, name="Residual", line=dict(color='white', width=2)), 2, 1)
    fig_reg.add_trace(go.Scatter(x=upper_band.index, y=upper_band, name="+2 Sigma (Rich)", line=dict(color='#ef4444', dash='dash')), 2, 1)
    fig_reg.add_trace(go.Scatter(x=lower_band.index, y=lower_band, name="-1 Sigma (Cheap)", line=dict(color='#22c55e', dash='dash')), 2, 1)

    fig_reg.update_layout(height=800, template="plotly_dark", hovermode="x unified")
    st.plotly_chart(fig_reg, use_container_width=True)

    # 5. Live Signals logic
    latest_res = resids.iloc[-1]
    if latest_res > upper_band.iloc[-1]:
        st.warning("⚠️ **Statistically Rich:** Asset A is overextended relative to Asset B. Potential Mean Reversion Short.")
    elif latest_res < lower_band.iloc[-1]:
        st.success("✅ **Statistically Cheap:** Asset A is lagging Asset B. Potential Mean Reversion Long.")
    else:
        st.info("⚖️ **Fair Value:** The relationship is within normal historical bounds.")
# --- TAB 6: BACKTESTING ENGINE ---
with tab6:
    st.subheader("🧪 Factor Momentum Backtester")
    st.markdown("""
    This engine simulates a **Momentum Rotation Strategy**. Every rebalance period, it ranks your factor clusters 
    by their cumulative return over a lookback window and "buys" the top performers.
    """)
    
    # 1. User Inputs for Strategy Rules
    col_bt1, col_bt2, col_bt3 = st.columns(3)
    lookback_window = col_bt1.slider("Selection Window (Days)", 5, 60, 20, 
                                     help="The number of past trading days used to rank factor performance.")
    top_n = col_bt2.slider("Number of Factors to Hold", 1, 5, 3, 
                           help="The number of top-ranked factors to include in the portfolio.")
    rebalance_freq = col_bt3.selectbox("Rebalance Frequency", ["Monthly", "Weekly"])

    # 2. Prepare Signals (Rolling Returns)
    # We use cumulative returns over the lookback window to rank factors
    signal_df = factor_returns.rolling(window=lookback_window).sum()
    
    # Generate calendar rebalance dates
    if rebalance_freq == "Monthly":
        rebal_dates = factor_returns.resample('ME').last().index
    else:
        rebal_dates = factor_returns.resample('W').last().index

    portfolio_returns = []
    
    # 3. Backtest Loop with Date Alignment Fix
    for i in range(len(rebal_dates)-1):
        try:
            # Find the actual closest trading days to avoid KeyErrors on weekends/holidays
            # .get_indexer with method='ffill' finds the last valid trading day ON or BEFORE the calendar date
            idx_start = factor_returns.index.get_indexer([rebal_dates[i]], method='ffill')[0]
            idx_end = factor_returns.index.get_indexer([rebal_dates[i+1]], method='ffill')[0]
            
            actual_start_date = factor_returns.index[idx_start]
            actual_end_date = factor_returns.index[idx_end]
            
            # RANKING: Get top factors based on signal available at the start date
            current_signals = signal_df.loc[actual_start_date].dropna().sort_values(ascending=False)
            
            if not current_signals.empty:
                selected_factors = current_signals.head(top_n).index.tolist()
                
                # PERFORMANCE: Calculate mean return of selected factors for the forward period
                # We slice from start+1 to end to avoid 'buying' at the price we used to rank
                period_rets = factor_returns.loc[actual_start_date:actual_end_date, selected_factors].mean(axis=1)
                
                if not portfolio_returns:
                    portfolio_returns.append(period_rets)
                else:
                    # Avoid duplicating the return of the rebalance day
                    portfolio_returns.append(period_rets.iloc[1:])
        except Exception as e:
            continue

    # 4. Results Visualization
    if portfolio_returns:
        # Concatenate all periods into a single series
        strat_rets = pd.concat(portfolio_returns)
        strat_rets = strat_rets[~strat_rets.index.duplicated()]
        
        # Equity Curves
        cum_strat = (1 + strat_rets).cumprod()
        spy_bench_rets = data['SPY'].pct_change().loc[cum_strat.index[0]:cum_strat.index[-1]]
        cum_spy = (1 + spy_bench_rets).cumprod()

        # Drawdown Calculation (Peak to Trough)
        rolling_max = cum_strat.cummax()
        drawdown = (cum_strat - rolling_max) / rolling_max
        
        # Charts
        fig_bt = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                               vertical_spacing=0.05, 
                               subplot_titles=("Cumulative Returns", "Drawdown (%)"),
                               row_heights=[0.7, 0.3])

        # Equity Curve Trace
        fig_bt.add_trace(go.Scatter(x=cum_strat.index, y=cum_strat, name="Strategy", line=dict(color='#22c55e', width=3)), row=1, col=1)
        fig_bt.add_trace(go.Scatter(x=cum_spy.index, y=cum_spy, name="S&P 500", line=dict(color='#64748b', dash='dot')), row=1, col=1)
        
        # Drawdown Trace
        fig_bt.add_trace(go.Scatter(x=drawdown.index, y=drawdown*100, name="Drawdown", fill='tozeroy', line=dict(color='#ef4444')), row=2, col=1)

        fig_bt.update_layout(height=700, template="plotly_dark", hovermode="x unified", showlegend=True)
        st.plotly_chart(fig_bt, use_container_width=True)

        # 5. Summary Metrics Table
        total_ret = (cum_strat.iloc[-1] - 1) * 100
        ann_vol = strat_rets.std() * np.sqrt(252) * 100
        sharpe = (strat_rets.mean() / strat_rets.std()) * np.sqrt(252) if strat_rets.std() != 0 else 0
        max_dd = drawdown.min() * 100

        st.markdown("### 📊 Strategy Performance Summary")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Return", f"{total_ret:.1f}%")
        m2.metric("Annualized Vol", f"{ann_vol:.1f}%")
        m3.metric("Sharpe Ratio", f"{sharpe:.2f}")
        m4.metric("Max Drawdown", f"{max_dd:.1f}%")

        # 6. Current Strategy Signals (LIVE)
        st.divider()
        st.markdown("### 🎯 Current Strategy Signals")
        st.caption(f"Based on latest closing data and a {lookback_window}-day lookback.")
        
        # Get the very last row of momentum signals
        latest_signals = signal_df.iloc[-1].dropna().sort_values(ascending=False)
        
        sig_col1, sig_col2 = st.columns(2)
        
        with sig_col1:
            st.success(f"**LONG (Top {top_n})**")
            current_longs = latest_signals.head(top_n).index.tolist()
            for factor in current_longs:
                st.write(f"✅ {factor}")
                
        with sig_col2:
            st.error(f"**SHORT (Bottom {top_n})**")
            current_shorts = latest_signals.tail(top_n).index.tolist()
            for factor in current_shorts:
                st.write(f"🔻 {factor}")
        
    else:
        st.error("Could not generate backtest. Ensure you have enough historical data for the selected lookback window.")
# --- TAB 7: FACTOR COMPARISON ---
with tab7:
    st.subheader("Factor & Ticker Comparison")
    st.markdown("Compare cumulative returns across multiple factors or tickers over a custom window.")

    # 1. Date and Selection Controls
    c1, c2 = st.columns(2)
    start_date = c1.date_input("Start Date", value=datetime.now() - timedelta(days=90))
    end_date = c2.date_input("End Date", value=datetime.now())

    # Combine factors and individual tickers for selection
    comparison_options = list(st.session_state.clusters.keys()) + all_tickers
    selected_items = st.multiselect("Select Factors or Tickers (Max 5):", 
                                    options=comparison_options, 
                                    default=[list(st.session_state.clusters.keys())[0]])

    if selected_items:
        # 2. Data Preparation
        # Create a dataframe for comparison
        comp_df = pd.DataFrame()
        
        for item in selected_items:
            if item in st.session_state.clusters:
                # It's a factor cluster
                tickers = st.session_state.clusters[item]
                valid_t = [t for t in tickers if t in data.columns]
                # Calculate mean return of cluster and then cumulative growth
                daily_rets = data[valid_t].pct_change().mean(axis=1)
            else:
                # It's an individual ticker
                daily_rets = data[item].pct_change()
            
            # Filter by date and calculate cumulative return from 0%
            filtered_rets = daily_rets.loc[start_date:end_date]
            comp_df[item] = (1 + filtered_rets).cumprod() - 1

        # 3. Visualization
        fig_comp = go.Figure()
        
        for col in comp_df.columns:
            fig_comp.add_trace(go.Scatter(
                x=comp_df.index, 
                y=comp_df[col] * 100, 
                name=col,
                mode='lines',
                line=dict(width=3)
            ))

        # Add zero baseline
        fig_comp.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.5)

        fig_comp.update_layout(
            template="plotly_dark",
            height=500,
            yaxis_title="Cumulative Return (%)",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_comp, use_container_width=True)

        # 4. Summary Metric Cards (Matches your screenshot)
        st.write("---")
        m_cols = st.columns(len(selected_items))
        for i, item in enumerate(selected_items):
            total_perf = comp_df[item].iloc[-1] * 100
            color = "#22c55e" if total_perf > 0 else "#ef4444"
            m_cols[i].markdown(f"""
                <div style="background-color:#1e293b; padding:15px; border-radius:10px; text-align:center; border-bottom: 4px solid {color};">
                    <p style="margin:0; font-size:14px; color:#cbd5e1;">{item}</p>
                    <h3 style="margin:0; color:{color};">{total_perf:+.2f}%</h3>
                </div>
            """, unsafe_allow_html=True)

        # 5. Spread Logic (If 2 items selected)
        if len(selected_items) == 2:
            spread = (comp_df[selected_items[0]] - comp_df[selected_items[1]]) * 100
            st.write("")
            st.info(f"**Spread Analysis:** {selected_items[0]} is outperforming {selected_items[1]} by **{spread.iloc[-1]:.2f}%** over this period.")