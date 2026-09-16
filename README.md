# Battery-Optimisation-SA
Built using GPT-6 Astra. Takes data of energy prices over 5 minute intervals from 2021-2025.

There are three strategies tested:
1. Threshold: Charge below $30/MWh, discharge above $150/MWh using the previous completed interval's price.
2. Forecast Optimisation: Dynamically programs the best possible future price backwards, enabling us to follow its forecast.
3. Perfect Forecast (not a proper strategy, of course): Uses future prices and optimises using them. Acts as a perfect benchmark.

With 100MW power rating, 200MWh energy capacity, 160MWh usable energy capacity:
<img width="489" height="348" alt="image" src="https://github.com/user-attachments/assets/0a34efd9-9d80-4a07-98b0-9e4a8b587e50" />

Configurations are arbitrary and depend on the situation. Changing one assumption:

<img width="487" height="219" alt="image" src="https://github.com/user-attachments/assets/741ce17e-ded3-4e83-950d-df50dc0076a5" />

Model selected on validation:

<img width="485" height="270" alt="image" src="https://github.com/user-attachments/assets/43b389b1-7f50-4763-8a98-fe62441bc2bf" />

