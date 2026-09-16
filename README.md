# Battery-Optimisation-SA
Built using GPT-6 Astra. Takes data of energy prices over 5 minute intervals from 2021-2025.

There are three strategies tested:
1. Threshold: Charge below $30/MWh, discharge above $150/MWh using the previous completed interval's price.
2. Forecast Optimisation: Dynamically programs the best possible future price backwards, enabling us to follow its forecast.
3. Perfect Forecast (not a proper strategy, of course): Uses future prices and optimises using them. Acts as a perfect benchmark.
<img width="489" height="348" alt="image" src="https://github.com/user-attachments/assets/0a34efd9-9d80-4a07-98b0-9e4a8b587e50" />

Configurations are arbitrary and depend on the situation.
