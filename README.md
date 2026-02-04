# Cloud Resource Optimization Using Machine Learning
**Predictive Auto-Scaling System for Cost Reduction**

**Author:** Rebecca Sherwood  
**Project Type:** MSIT Capstone Project  
**Date:** February 2026

---

## Project Overview

This project implements an AI-powered system that uses machine learning to predict cloud resource demand and automatically optimize infrastructure allocation. The system reduces cloud costs by 52% while maintaining performance stability through predictive scaling.

### Key Results
- **Model Accuracy:** 19.2% MAPE (Good prediction accuracy)
- **Cost Savings:** 52.1% reduction in cloud spending
- **Automation:** 428 scaling decisions made automatically
- **Performance:** Maintained stable performance across 90 days of workload data

---

## Problem Statement

Organizations waste 30-40% of cloud expenditures due to:
- Over-provisioning of resources
- Reactive (not predictive) scaling policies
- Inability to anticipate workload fluctuations
- Manual configuration management

This project addresses these challenges through machine learning-based predictive auto-scaling.

---

## Solution Components

### Component A: Data Pipeline
- Ingests cloud usage metrics (CPU, memory, network)
- Cleans and normalizes data
- Engineers time-based features
- Handles missing values

### Component B: ML Forecasting Engine
- Uses Facebook Prophet for time series prediction
- Identifies daily and weekly patterns
- Generates hourly forecasts with confidence intervals
- Achieves 19.2% MAPE accuracy

### Component C: Automated Scaling Engine
- Translates predictions into scaling decisions
- Applies threshold-based logic (70% scale up, 30% scale down)
- Simulates cloud provider API calls
- Calculates cost savings in real-time

---

## Project Structure

```
CloudOptimization/
├── capstone_rebecca_sherwood_FIXED.py    # Main analysis code
├── dashboard_app.py                       # Interactive Streamlit dashboard
├── cloud_usage_data.csv                   # 90 days of cloud usage data
├── architecture_diagram.png               # System architecture
├── eda_time_series.png                    # Time series visualization
├── eda_patterns.png                       # Usage pattern analysis
├── model_predictions.png                  # Actual vs predicted results
├── model_components.png                   # Model seasonality analysis
├── scaling_decisions.png                  # Scaling decision visualization
└── README.md                              # This file
```

---

## How to Run This Project

### Prerequisites
- **Python 3.8 or higher**
- **Windows, Mac, or Linux** operating system
- **Internet connection** (for library installation)

---

## OPTION 1: Run the Main Analysis (Jupyter Notebook)

### Step 1: Install Python
If you don't have Python installed:
1. Download from: https://www.python.org/downloads/
2. **IMPORTANT:** Check "Add Python to PATH" during installation
3. Complete installation

### Step 2: Install Required Libraries
Open Command Prompt (Windows) or Terminal (Mac/Linux) and run:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn prophet jupyter
```

*Note: Installation takes 3-5 minutes*

### Step 3: Navigate to Project Folder
```bash
cd path/to/CloudOptimization
```
*(Replace `path/to/CloudOptimization` with your actual folder path)*

### Step 4: Launch Jupyter Notebook
```bash
jupyter notebook
```

Your browser will open automatically showing the project files.

### Step 5: Run the Analysis
1. Click **"New"** → **"Python 3"**
2. Open `capstone_rebecca_sherwood_FIXED.py` in a text editor
3. Copy all code (Ctrl+A, Ctrl+C)
4. Paste into Jupyter cell (Ctrl+V)
5. Click **"Cell"** → **"Run All"**

### Step 6: View Results
- Results appear below each code section
- Graphs display inline
- Runtime: approximately 1-2 minutes
- Final message: "CAPSTONE PROJECT COMPLETE!"

### Generated Output Files:
- `eda_time_series.png` - Usage trends over time
- `eda_patterns.png` - Daily and weekly patterns
- `model_predictions.png` - Prediction accuracy visualization
- `model_components.png` - Seasonality breakdown
- `scaling_decisions.png` - Auto-scaling actions

---

## OPTION 2: Run the Interactive Dashboard

### Step 1: Install Streamlit
```bash
pip install streamlit
```

### Step 2: Launch Dashboard
```bash
streamlit run dashboard_app.py
```

Or if that doesn't work:
```bash
python -m streamlit run dashboard_app.py
```

### Step 3: Explore Dashboard
Your browser opens automatically at `http://localhost:8501`

**Dashboard Pages:**
1. **Overview** - Project summary and key metrics
2. **Data Analysis** - Interactive data exploration
3. **ML Predictions** - Model performance and forecasts
4. **Auto-Scaling** - Scaling decisions with adjustable thresholds
5. **Cost Savings** - Financial impact and ROI analysis

**Interactive Features:**
- Adjust scaling thresholds with sliders
- Change cost parameters
- View real-time cost calculations
- Explore different metrics

---

## Key Results & Findings

### Model Performance
| Metric | Value | Interpretation |
|--------|-------|----------------|
| MAE | 7.92% | Mean Absolute Error |
| RMSE | 10.81% | Root Mean Squared Error |
| MAPE | 19.19% | Good prediction accuracy |

### Cost Analysis (Test Period: 428 hours)
| Metric | Value |
|--------|-------|
| Actual Cost (with auto-scaling) | $307.50 |
| Baseline Cost (without auto-scaling) | $642.00 |
| **Total Savings** | **$334.50 (52.1%)** |

### Scaling Actions
- **HOLD:** 426 decisions (no action needed)
- **SCALE UP:** 0 actions (resources sufficient)
- **SCALE DOWN:** 2 actions (reduced unnecessary capacity)

### Annual Projection
- **Annual Savings:** ~$2,570
- **5-Year ROI:** ~$12,850 (assuming $7,000 implementation cost)

---

## Technologies Used

- **Python 3.11** - Programming language
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing
- **Matplotlib/Seaborn** - Data visualization
- **Scikit-learn** - Machine learning metrics
- **Prophet** - Time series forecasting (by Facebook)
- **Jupyter** - Interactive development environment
- **Streamlit** - Interactive dashboard framework

---

## Dataset Information

**File:** `cloud_usage_data.csv`

**Size:** 2,160 records (90 days × 24 hours)

**Metrics Collected:**
- `timestamp` - Date and time of measurement
- `cpu_percent` - CPU utilization (0-100%)
- `memory_percent` - Memory utilization (0-100%)
- `network_mbps` - Network throughput (Mbps)
- `hour` - Hour of day (0-23)
- `day_of_week` - Day of week (0=Monday, 6=Sunday)
- `is_weekend` - Weekend flag (0=weekday, 1=weekend)
- `is_business_hours` - Business hours flag (1=9am-5pm)

**Data Characteristics:**
- Realistic daily patterns (higher usage during business hours)
- Weekly seasonality (lower usage on weekends)
- Random spikes simulating unexpected demand
- Missing values (~2%) to simulate real-world data quality issues

---

## Academic Context

**Program:** Master of Science in Information Technology (MSIT)  
**Focus Areas:**
- Cloud computing and infrastructure optimization
- Machine learning and predictive analytics
- Data-driven decision making
- Automated system design

**Learning Outcomes Demonstrated:**
1. Applied machine learning to solve real-world IT challenges
2. Designed and implemented end-to-end data pipeline
3. Evaluated model performance using industry-standard metrics
4. Demonstrated cost-benefit analysis and ROI calculation
5. Created professional documentation and visualizations

---

## Future Enhancements

If this project were extended beyond the capstone scope, potential enhancements include:

1. **Real Cloud Integration**
   - Connect to AWS Auto Scaling API
   - Integrate with Azure Virtual Machine Scale Sets
   - Support Google Cloud Platform auto-scaling

2. **Advanced ML Models**
   - Compare LSTM vs Prophet vs ARIMA
   - Ensemble methods for improved accuracy
   - Multi-metric prediction (CPU + Memory + Network simultaneously)

3. **Real-Time Monitoring**
   - Live dashboard with streaming data
   - Anomaly detection for unusual patterns
   - Alert system for critical events

4. **Multi-Cloud Support**
   - Unified interface for AWS, Azure, GCP
   - Cross-cloud cost optimization
   - Vendor-agnostic architecture

5. **Enhanced Features**
   - User authentication and role-based access
   - Historical decision audit log
   - A/B testing for threshold optimization
   - Integration with monitoring tools (Datadog, New Relic)

---

## References

- 2025 State of the Cloud Report. (2025). Cloud waste and optimization trends.
- Keen, M. (2025). Cloud optimization as a competitive advantage.
- Li, H., et al. (2025). Machine learning for cloud resource management.
- Wang, L., & Xing, H. (n.d.). Predictive scaling in cloud environments.

---

## Contact

**Rebecca Sherwood**  
MSIT Candidate  
Email: sherwood.r3@icloud.com  
GitHub: BeccaSB  


---

## License

This project is submitted as academic work for the Master of Science in Information Technology program.

---

## Acknowledgments

- **Prophet Library** by Facebook Research for time series forecasting
- **Streamlit** for the interactive dashboard framework
- **MSIT Program Faculty** for guidance and support
- **Industry Practitioners** whose insights informed this research
- **My Husband & Son** for their support and encouragment

---

## Troubleshooting

### Issue: "python is not recognized"
**Solution:** Python wasn't added to PATH. Reinstall Python and check "Add Python to PATH"

### Issue: "No module named 'prophet'"
**Solution:** Run `pip install prophet`

### Issue: "FileNotFoundError: cloud_usage_data.csv"
**Solution:** Ensure `cloud_usage_data.csv` is in the same folder as the Python script

### Issue: Jupyter doesn't open
**Solution:** Try navigating to `http://localhost:8888` manually in your browser

### Issue: Streamlit won't run
**Solution:** Use `python -m streamlit run dashboard_app.py` instead

For additional support, please contact the author or refer to library documentation.

---

**Last Updated:** February 2026  
**Version:** 1.0  
**Status:** ✅ Complete and Tested
