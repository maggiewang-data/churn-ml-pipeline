# Monitoring report: 6 simulated months after launch

Launch PR-AUC: 0.661. PSI thresholds: watch > 0.1, act > 0.25. Investigate rules: {'max_feature_psi': 0.25, 'max_score_psi': 0.25}. Retrain rules: {'max_pr_auc_drop': 0.05, 'max_calibration_gap': 0.05}

| batch                 |    n | top_drift_feature   |   top_feature_psi |   score_psi |   pr_auc |   calibration_gap | action      | why                                                      |
|:----------------------|-----:|:--------------------|------------------:|------------:|---------:|------------------:|:------------|:---------------------------------------------------------|
| M1 normal             | 1500 | OnlineBackup        |             0.011 |       0.015 |    0.681 |             0.008 | OK          |                                                          |
| M2 normal             | 1500 | MonthlyCharges      |             0.017 |       0.016 |    0.651 |             0.011 | OK          |                                                          |
| M3 m2m promo          | 1500 | Contract            |             0.269 |       0.17  |    0.683 |             0.007 | INVESTIGATE | input drift in Contract (PSI 0.27)                       |
| M4 fiber price +15%   | 1500 | MonthlyCharges      |             0.24  |       0.026 |    0.743 |             0.056 | RETRAIN     | calibration gap 0.056 (predicted 28.8% vs. actual 34.5%) |
| M5 new payment method | 1500 | PaymentMethod       |             1.989 |       0.011 |    0.647 |             0.003 | INVESTIGATE | input drift in PaymentMethod (PSI 1.99)                  |
| M6 normal             | 1500 | MonthlyCharges      |             0.016 |       0.007 |    0.669 |             0.002 | OK          |                                                          |
