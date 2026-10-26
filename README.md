# Monte Carlo Uncertainty Analysis

## Description

The Monte Carlo Uncertainty Analysis is a Python program that studies how uncertain input parameters affect the output of a simple electrical system. It calculates the voltage across a resistor in a voltage divider, simulates many random variations of the supply voltage and resistor values, reports summary statistics for the output voltage, determines the probability of exceeding an engineering limit, examines how the results change as the number of simulations increases, and performs a sensitivity analysis on each input.

The example system is a hypothetical voltage divider with a 12 V supply, a 7000 ohm resistor R1, and a 5000 ohm resistor R2. The output voltage is calculated as `V_out = V_in * R2 / (R1 + R2)`, which gives a nominal output of 5.000 V. All data used by the program is synthetic and is not a measurement of a real circuit.

The program is organized as a set of functions, each responsible for one part of the analysis, with `main()` acting as the driver that coordinates them and passes their results to the report. The system equation is in `src/simulation.py` along with the sampling, statistics, convergence, and sensitivity functions. The plots are in `src/plots.py`, and the tests are in `tests/`.

## Requirements

* Python 3
* NumPy
* Matplotlib
* pytest (only needed to run the tests)

## Build and Run

Install the required libraries from the project directory using:

```
python -m pip install -r requirements.txt
```

Run the program from the project directory using:

```
python main.py
```

Depending on the Python installation, the following command may also be used:

```
python3 main.py
```

Run the tests from the project directory using:

```
python -m pytest
```

## Usage

Run the program to display the uncertainty analysis report. The program uses the input parameters and uncertainty distributions defined in the configuration section at the top of `main.py` and does not require user input. The random seed is fixed, so every run produces the same results.

For example:

```
SYNTHETIC DATA - hypothetical voltage divider, not a real measurement

Inputs
  v_in  nominal       12 V    (normal, std 0.2 V)
  r1    nominal     7000 ohm  (uniform, +/- 350 ohm)
  r2    nominal     5000 ohm  (normal, std 83.3333 ohm)

Nominal V_out (no uncertainty): 5.0000 V

Monte Carlo results  (n = 100,000, seed = 2026)
  mean                5.0017 V   (+/- 0.0004 V sampling error)
  median              5.0006 V
  standard deviation  0.1277 V
  min / max           4.4565 / 5.5142 V
  percentile P1       4.7176 V
  percentile P5       4.7944 V
  percentile P50      5.0006 V
  percentile P95      5.2133 V
  percentile P99      5.2939 V
  P(V_out > 5.2 V)    6.2520%   (+/- 0.0766% sampling error)
  mean - nominal      +0.00172 V

Convergence  (20 repeats per n; 'spread' = std. dev. across repeats)
         n   mean [V]    spread   std [V]  P(exceed)    spread
       100     5.0039   0.00946    0.1309     7.500%    2.947%
       300     5.0005   0.00579    0.1290     6.433%    1.027%
     1,000     5.0005   0.00439    0.1287     6.300%    0.868%
     3,000     5.0001   0.00271    0.1282     6.192%    0.309%
    10,000     5.0009   0.00136    0.1280     6.225%    0.251%
    30,000     5.0009   0.00074    0.1281     6.189%    0.141%
   100,000     5.0011   0.00040    0.1281     6.213%    0.065%

Sensitivity  (n = 50,000 per run; one input's uncertainty scaled, others at baseline)
  output std [mV]    x0     x0.5   x1     x1.5   x2
    v_in            97.3  105.9  128.2  158.4  193.0
    r1              96.3  105.2  128.2  159.3  194.9
    r2             118.8  121.1  128.2  139.1  153.2
  P(exceed) [%]      x0     x0.5   x1     x1.5   x2
    v_in            1.33   2.70   6.24  10.66  15.17
    r1              1.89   2.85   6.24  11.82  18.48
    r2              5.01   5.26   6.24   7.76   9.71

Figures written:
  outputs\output_distribution.png
  outputs\convergence.png
  outputs\sensitivity.png
```

The supply voltage follows a normal distribution with a standard deviation of 0.2 V. R1 follows a uniform distribution within 5 percent of its nominal value. R2 follows a normal distribution where 5 percent of its nominal value is three standard deviations. The program compares each simulated output against an engineering limit of 5.2 V. It reports the fraction of samples above that limit as the probability of exceeding it.

The convergence study runs the simulation 20 times with different seeds at each sample size. The spread column shows how much the estimate changes between those runs. The sensitivity study multiplies the uncertainty of one input at a time by 0, 0.5, 1, 1.5, and 2 while the other inputs stay at their baseline uncertainty.

The program saves three figures to the `outputs` folder. `output_distribution.png` shows a histogram of the simulated output voltage compared with the nominal value, the simulated mean, and the limit. `convergence.png` shows how the estimates and their spread change as the number of simulations increases. `sensitivity.png` shows how the output spread and the probability of exceeding the limit change as each input's uncertainty is scaled.

The tests check the voltage divider equation against hand-calculated examples, such as 9 V with a 2000 ohm R1 and a 1000 ohm R2 giving 3 V. They also check that the same seed reproduces the same samples and that the statistics match known values. Another test checks that the sampling error shrinks as the number of simulations increases.

## Input Uncertainty, Output Variability, and Sampling Error

Input uncertainty is how much the input parameters may vary from their nominal values. In this program, it is the spread of the supply voltage and resistor values defined in the configuration section. These distributions are assumptions chosen for the example, and the simulation cannot check whether they are correct.

Output variability is the spread in the output voltage that results from passing the input uncertainty through the system equation. In this program, the output has a standard deviation of about 0.128 V and a 6.25 percent chance of exceeding 5.2 V. Running more simulations does not reduce output variability. It can only be reduced by tightening the input tolerances or changing the design.

Numerical sampling error comes from estimating the output with a limited number of random samples. Running the simulation with a different seed gives slightly different results. This error shrinks as the number of simulations increases, at a rate of about 1 divided by the square root of the number of simulations. The program reports this error next to the mean and the probability of exceeding the limit. A fixed seed makes the results reproducible but does not make them more accurate.

## AI Disclosure

The data used in this project was generated using AI. The concepts used in this project were explained to me by AI.
