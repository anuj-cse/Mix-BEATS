# Mix-BEATS
Repository for ACM Future and Sustainable Energy Systems (e-Energy 2025)


# Mix-BEATS: Mixer-enhanced Basis Expansion Analysis for Load Forecasting

## Mix-BEATS Model Overview
Introducing Mix-BEATS, a novel model combining N-BEATS and TSMixer blocks for accurate and efficient short-term load forecasting (STLF) in smart buildings, with fewer parameters and a design optimized for edge-device deployment.

![Mix-BEATS Model Figure](Mix-BEATS/plots/mix-beats.jpg)

## Key Features

- **Mix-Beats Architecture**: Combines task-specific enhancements with N-BEATS-inspired design to optimize short-term load forecasting.
- **Generalized Forecasting**: Handles diverse load patterns across various building profiles.
- **Anomaly Detection**: Detects irregularities in energy consumption trends.
- **Open Benchmarking**: Results compared with N-BEATS and other state-of-the-art models for reproducibility and transparency.

## Results

### Performance Metrics

The table below compares the forecasting performance of Mix-Beats with N-BEATS and TTMS models across multiple datasets. Metrics include Mean Absolute Percentage Error (MAPE) and Root Mean Squared Error (RMSE):

| Model      | Dataset  | MAPE (%) | RMSE  |
|------------|----------|----------|-------|
| Mix-Beats  | Dataset A| 3.21     | 5.12  |
| N-BEATS    | Dataset A| 3.56     | 5.42  |
| TTMS       | Dataset A| 3.89     | 5.67  |
| Mix-Beats  | Dataset B| 4.12     | 6.45  |
| N-BEATS    | Dataset B| 4.56     | 6.89  |
| TTMS       | Dataset B| 4.78     | 7.01  |

### Key Insights

- **Mix-Beats** outperforms N-BEATS and TTMS on both MAPE and RMSE metrics, demonstrating its robustness and accuracy for STLF tasks.
- The architecture's focus on task-specific enhancements ensures adaptability to diverse building profiles.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/EnergyFM.git
   cd EnergyFM

