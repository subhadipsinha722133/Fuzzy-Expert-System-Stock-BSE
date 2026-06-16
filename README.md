## 📈 Fuzzy Expert System for Stock Portfolio Selection 🤖
- 🌟 Overview
  - This project implements a novel Fuzzy Expert System 🧠 designed for optimal stock portfolio selection. The system leverages fuzzy logic and Dempster-Shafer (DS) theory 📊 for rule construction, along with Ant Colony Optimization (ACO) 🐜 for model optimization. The primary goal is to allocate investor assets among various stocks to maximize returns 📈 while minimizing risk 📉.
    
---

## 🌐Live Demo 
https://fuzzy-expert-system-stock-bse-jrvxwcszcx8mkawrk4rs5l.streamlit.app/

---
## 🎯 Key Features
- 🧩 Fuzzy Rule Construction using DS-Theory: Implements Dempster-Shafer theory to handle uncertainty and combine evidence from multiple factors efficiently, reducing implementation time and cost ⏱️💰.
- 🎯 Defuzzification: Converts fuzzy output into crisp values for actionable investment decisions ✅.
- 🚀 Optimization using ACO: Utilizes Ant Colony Optimization to maximize the objective function: (fuzzy portfolio return - risk free return) / weighted mean semi-variance 📊➗.
  
---

## 📋 Factors Considered
The system evaluates stocks based on the following factors, each with its geometric mean as derived from historical data:

- SI No.	Factor	Geometric Mean (M_F)
  - 1️⃣	Earn per share	7.17
  - 2️⃣	Payout ratio	7.00
  - 3️⃣	Price to earning ratio	8.43
  - 4️⃣	Price to sales ratio	8.35
  - 5️⃣	Current ratio	6.02
  - 6️⃣	Price to book value ratio	8.26
  - 7️⃣	Price to cash flow ratio	6.59
  - 8️⃣	Profit margin	6.52
  - 9️⃣	Long term debt to equity ratio	8.20
  - 🔟	Accounts receivable turnover	5.81
---

## ⚙️ How It Works
- 📥 Input Processing: The system takes stock data and applies fuzzy logic to handle imprecise or uncertain information.
- 🧠 Rule Construction: Using Dempster-Shafer theory, rules are constructed to combine evidence from the listed factors.
- 🎯 Defuzzification: The fuzzy outputs are converted into crisp values to determine the optimal stock allocation.
- 🚀 Optimization: ACO is applied to refine the model, ensuring maximum return with minimal risk.
---

## 🌍 Global Applicability 🌎
As the outcome of this model found to be satisfactory ✅, this can be implemented for any stock exchanges around the world 🌐 but the selection of critical factors may vary over different stock exchanges. This fuzzy expert system model can be used to rank any set of alternatives based on the factors influencing them.

---

## 🔮 Future Enhancements & Research Directions
- 🌐 Integration of expert system on integrated data of assets from different stock exchanges and mutual funds 🤝.
- 🔄 Alternative Optimization Algorithms: Researchers can use other meta-heuristic algorithms such as:
- Simulated Annealing 🔥
- Tabu Search 🚫
- Particle Swarm Optimization 🐦‍🔥
- Genetic Algorithm 🧬
- 📊 Enhanced Visualization tools for results interpretation 📉➡️📈.
- ⏰ Real-time Data Feeds integration for live analysis 📡.

---
# 📊 Dynamic Stock Portfolio Selection & Optimization System

### 🚀 Powered by Dempster-Shafer Fuzzy Expert System (DS-FES) & Ant Colony Optimization (ACO)

---

## 📝 Project Overview
This repository contains an industry-grade **Decision Support System (DSS)** for automated stock ranking and portfolio optimization within the Bombay Stock Exchange (BSE). By replacing traditional, subjective expert consensus methods (like the Fuzzy Delphi Method) with data-driven **Grey Relational Analysis (GRA)**, this system processes a decade-long financial dataset dynamically to identify high-return, low-risk investment configurations under real-world market uncertainties.

---

## 🛠️ System Architecture & Framework
The application architecture is divided into four main soft computing pipelines:
1. **Feature Selection (GRA Framework):** Evaluates multi-dimensional financial parameters (such as Diluted EPS, Profit Margin, Price-to-Book Value, and Debt-to-Equity Ratio) to filter out information redundancies using a correlation mask.
2. **Fuzzification Module:** Maps crisp normalized corporate indicators to linguistic categories (*Low, Standard, High*) utilizing statistical percentile bounds through custom Trapezoidal Membership Functions.
3. **Automated Synthesis Engine (DS Theory):** Automatically generates a comprehensive base of 81 rules using Dempster-Shafer Rule of Combination, eliminating manual rule-writing bottlenecks.
4. **Portfolio Optimizer (Continuous ACO Solver):** Maximizes the Modified Sharpe Ratio against downside semi-variance variations using an interactive Ant Colony Heuristic to output perfect capital weight proportions.

---

## 📑 Core Application Features & Tab Structures

### 📅 Tab 1: Historical Optimization Dashboard
* **Dynamic Loading:** Connects directly with operational matrix files (`6_Final_Ranking.csv`, `Avg_Return_Matrix.csv`).
* **Financial Year Filter:** Tracks actual asset trajectories across historical backtesting cycles (FY13 to FY25) instantaneously.
* **Risk-Return Trade-offs:** Visualizes learning convergence tracks and downside variation coordinate spreads via active Matplotlib charts.

### ⚡ Tab 2: Real-Time Simulation Panel
* **Run-Time Parameter Modulation:** Provides a direct testing workspace where users can manually input present-time custom return and risk configurations.
* **Instant Re-weighting Solver:** Automatically triggers the background ACO array script upon value changes to show immediate budget redirection behavior.

### 🎯 Tab 3: AI Stock Selection Matrix
* **Risk-Reward Appetite Sliders:** Allows users to set minimum acceptable return thresholds and maximum risk constraints.
* **Smart Filtering Recommendations:** Displays an explicit filtered recommendation pool showing the best specific company names matching the given criteria.

### 📈 Tab 4: Predictive Success Analytics
* **Empirical Backtesting Track:** Displays predictive hit-ratio validation trend line scores across several financial years (FY23 to FY26).
* **soft Computing Evaluation Matrix:** Validates structural performance and parameter efficiency logs against real historical market cycles.

---

## 🚀 How to Run the Workspace

### 🔧 1. Prerequisites Setup
Ensure you have Python installed on your local workstation environment, then run the terminal installation command:
```bash
pip install -r requirements.txt
## 🚀 Usage
To use this system, provide stock data corresponding to the factors listed above. The model will process the inputs and output a recommended portfolio allocation.

---

## 📄 License
This project is licensed under the MIT License. See the LICENSE file for details.
