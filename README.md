# headrest-evaluation
Stimuli using pygame for headrest evaluation

## ERP Stimulus Requirements
* One yellow shape as the stimulus on a black background separated by black screens
    * Square as the standard stimulus
    * Circle as the target stimulus
* Stimulus duration of 500 ms
* Inter-trial-interval (rest period) of 500 ms
* Must timestamp beginning and end of each trial (stimulus) with clear labeling of which stimulus
* 100 total trials
    * Standard:target ratio of 80:20

## SSVEP Stimulus Requirements
* One flickering, yellow circle as the stimulus on a black background
    * Can flicker at frequencies of 7.5, 8.57, 10, and 12 Hz
    * Randomize ordering of frequencies
* Stimulus duration of 8 seconds per frequency
* Rest duration of 20 seconds before moving onto next stimulus frequency
* Must timestamp at the beginning and end of each stimulus period
* Total duration of ~1.5 min

## Executing Stimulus Program
To run the SSVEP stimulus:
```
python ssvep/stimulus.py
```

To run the ERP stimulus:
```
python erp/stimulus.py
```

## TiD Values
Within the code, we send commands to amplifer to timestamp specific events. Each event correlates to a `TiD` value.

Here we list the TiD values and the events that are correlated with it.

### SSVEP TiD
| TiD | Event |
| --- | --- |
| 99  | Beginning of SSVEP trials |
| 199 | End of SSVEP trials |
| 100 | Start/End of 7.5 Hz trial |
| 101 | Start/End of 8.57 Hz trial |
| 102 | Start/End of 10 Hz trial |
| 103 | Start/End of 12 Hz trial |

**Note:** In the timestamps, a trial for a frequency will begin with a TiD and end with the same TiD. For example, a trial for 7.5 Hz will timestamp the beginning of the trial with a 100 TiD. When it ends, it will timestamp the end of the trial with a 100 TiD as well.

### ERP TiD
| TiD | Event |
| --- | --- |
| 299 | Beginning of ERP trials |
| 399 | End of ERP trials |
| 300 | Standard stimulus (square) appears |
| 301 | Target stimulus (circle) appears |
| 302 | Key was pressed |