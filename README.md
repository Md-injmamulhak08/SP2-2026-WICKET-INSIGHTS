# Sona Power Predict – 2026

**College Name:** Sona College of Technology  
**Team Name:** Wicket Insights  

### Team Members
* **Vibashini S** - [Year, 1st] - [Department, e.g., CSE]
* **[Teammate 2 Name]** - [Year] - [Department]
* **[Teammate 3 Name]** - [Year] - [Department]
*(Remove or add bullet points if you have more/fewer members)*

### Libraries Used
* **`pandas`** (Data manipulation and chronological window functions)
* **`numpy`** (Mathematical operations and failsafe prediction clipping)
* **`lightgbm`** (Gradient boosting tree architecture)
* **`scikit-learn`** (Label encoding for categorical variables)
* **`re`** (Regex for dynamic player ID extraction)

### Brief Explanation of Our Approach / Model

Team Wicket Insights built a highly defensive, dual-architecture machine learning pipeline designed specifically to handle dirty hackathon data and the distinct psychology of T20 cricket innings. 

Our methodology is built on four core pillars:

**1. Dual-Model Architecture**
We split our predictions into two distinct LightGBM Regressors (`model_inn1` and `model_inn2`). The 1st innings model learns pitch conditions and par scores, while the 2nd innings model includes `target_score` as a core feature to understand required run rates and scoreboard pressure.

**2. Chronological Sliding Windows (Zero Data Leakage)**
Instead of using lifetime career averages, we engineered 5-match rolling windows for Batter Strike Rate, Bowler Economy, and Venue Form. We strictly used `.shift(1)` combined with `.transform()` to ensure the model never looks at the current match's target variable, mathematically preventing data leakage.

**3. Dynamic Wicket Pressure**
To simulate the psychological impact of a batting collapse, we track the unique number of batters who face a delivery in the PowerPlay (`num_batters`). If 5 or more batters are required (indicating 3+ early wickets), the model applies an extreme pressure penalty multiplier (scaling down to `0.72`) to severely drop the predicted score, capturing the reality of teams going into survival mode. All final predictions are safely clipped between 25 and 115 runs.

**4. Defensive Failsafes (Crash-Proofing)**
Our pipeline includes a `_standardize_columns()` method that automatically fixes misnamed columns and drops accidental duplicates. Furthermore, we built an ultimate `try-except` failsafe: if an unseen categorical variable entirely breaks the test array, our script bypasses the tree and calculates a mathematically sound estimate using the training mean, adjusted by the wicket pressure multipliers mentioned above.

**LICENCE** : 
This project is licensed under the MIT License.
