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
| 1   | Beginning of SSVEP trials |
| 1   | End of SSVEP trials |
| 10  | Start/End of 7.5 Hz trial |
| 11  | Start/End of 8.57 Hz trial |
| 12  | Start/End of 10 Hz trial |
| 13  | Start/End of 12 Hz trial |

**Note:** In the timestamps, a trial for a frequency will begin with a TiD and end with the same TiD. For example, a trial for 7.5 Hz will timestamp the beginning of the trial with a 100 TiD. When it ends, it will timestamp the end of the trial with a 100 TiD as well.

### ERP TiD
| TiD | Event |
| --- | --- |
| 1   | Beginning of ERP trials |
| 1   | End of ERP trials |
| 10  | Standard stimulus (square) appears |
| 20  | Target stimulus (circle) appears |
| 30  | Key was pressed |

## Executing Analysis Code
### Step 1: Install MNE
Install a specific version of MNE due to compatibility issues with the latest version.

```
pip install mne==1.6
```
### Step 2: Install Compatible Numpy Version
The latest MNE version installs a newer numpy version that may cause errors. Install a compatible version:

```
pip install numpy==1.23.5
```

### Step 3: Modify MNE to Handle GDF Format
MNE's handling of GDF files may cause errors due to lowpass filter settings. Perform a manual fix:

Locate edf.py in your MNE installation, typically found in mne/io/edf/edf.py.

Edit line 727: Change ```if info["highpass"] > info["lowpass"]:``` to ```if info["highpass"] >= info["lowpass"]:```

This ensures proper handling of lowpass filter settings for GDF files.
