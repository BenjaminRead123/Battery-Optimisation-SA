# Battery-Optimisation-SA
Built using GPT-6 Astra. Takes data of energy prices over 5 minute intervals from 2021-2025.

There are three strategies tested:
1. Threshold: Charge below $30/MWh, discharge above $150/MWh using the previous completed interval's price.
2. Forecast Optimisation: Dynamically programs the best possible future price backwards, enabling us to follow its forecast.
3. Perfect Forecast (not a proper strategy, of course): Uses future prices and optimises using them. Acts as a perfect benchmark.

Configurations are arbitrary and depend on the situation.
