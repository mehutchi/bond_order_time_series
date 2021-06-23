# bond_order_time_series

A time-saving method that uses bond order time series obtained from ab initio molecular dynamics to quickly and accurately predict temporal locations of reaction events. The method is tuned using two adjustable parameters to bring reaction event predictions into closer agreement with an ideal reference set of reaction events given infinite computing resources.

Publication of this method can be found here:

https://pubs.acs.org/doi/10.1021/acs.jctc.9b01039

Citation:

Bond-Order Time Series Analysis for Detecting Reaction Events in Ab Initio Molecular Dynamics Simulations.
Marshall Hutchings, Johnson Liu, Yudong Qiu, Chenchen Song, and Lee-Ping Wang.
Journal of Chemical Theory and Computation 2020 16 (3), 1606-1617
DOI: 10.1021/acs.jctc.9b01039

The BO_time_series_FINAL.py code requires 4 input files (samples provided). 

1. xyz coordinates (sample -> coors.xyz)
      
2. bond order list file (sample -> bond_order.list)

3. geometry optimized bond order list file (sample -> opt_bond_order.list)

4. cluster time series json file (sample -> timeseries.json)

Files 1 and 2 are easily obtained from an ab initio molecular dynamics simulation (this method is intended for a single reactive molecule, but the principles can carry over for more complex systems), but 3 and 4 require more effort. File 3 is a homemade file obtained by using each frame from file 1 as an input coordinate for a geometry optimization and creating a optimized bond order list file by appending the resulting bond order data from each simulation. The provided sample file 3 is the result of 2000 geometry optimizations. File 4 is a list of cluster indices according to how each optimized frame was assigned. This is obtained by creating a homemade optimized coordinates file (by appending the resulting geometries in order) and using a RMSD-esque bond order metric (described in the publication) to perform hierarchical clustering at a threshold of 1.0 (justified in the publication) to get the cluster assignments for each frame. File 4 is the resource-intensive (and impractical for everyday use) reference set of reaction events which helps parameterize this method.

Running the code with three sigma (smoothing parameter) inputs (start, stop, increment -> according to np.arange()) AND three threshold inputs creates a heat map of objective function scores. There are some options for the objective function, but we use the "bondwise" option (set as default) as explained in the paper.

Smoothing parameter and threshold heatmap for sample data set (heatmap pdf and additional diagnostic information file found under the /heat_map/ directory):

![image](https://user-images.githubusercontent.com/20996215/122833545-02e48e00-d2a2-11eb-8b26-f6589d86f669.png)

Objective function scores of zero and those below 0.5 are shown as whitespace and those from 0.5-1.0 according to the colorbar.  The heatmap helps the user calibrate the method for a given system type, in this example, an iron carbonyl system. Once the most optimally scoring region of the heatmap is identified the user can choose a parameter combination (in this example, we will select sigma = 50 and mu = 0.7\*sigma) setting to predict reaction events across large AIMD trajectories with high fidelity. 

Once the method settings have been identified, we can re-run the code with a single sigma and threshold as inputs to see the predictions. This option also produces five plots (which can be found along with additional diagnostic output files under the /single_point/ directory) to analyze quality of fit (showing sigma = 50 and mu = 0.7\*sigma):

1. Simplified reference vs predictions plot with corresponding AUC

![image](https://user-images.githubusercontent.com/20996215/122836227-b64f8180-d2a6-11eb-8fd0-5aa27b88bc50.png)

Left Panel: red - reference reaction locations, blue - predicted reaction locations, orange - expanding windows to measure distance from predicted to reference reactions (differs depending on objective function choice) halted once 100% of references detected

Right Panel: TPR vs FPR plot, similar to receiver operating characteristic. Dark orange - (TPR, FPR), light orange - area under the curve, gray - remainder of TPR vs FPR plot

2. Expanded reference vs predictions plot

![image](https://user-images.githubusercontent.com/20996215/122837230-99b44900-d2a8-11eb-9644-f087d2a99fa6.png)

This plot is similar to the left panel of #1, but is split into atom pairs to better showcase the "bondwise" objective function. Colored lines - reference reaction locations (colored according to distance from predictions, see colorbar on #3), Black - predicted reaction locations, orange - expanding windows to measure distance from predicted to reference reactions

3. Cluster transitions of reference reaction events

![image](https://user-images.githubusercontent.com/20996215/122839056-257ba480-d2ac-11eb-8616-e596d54b34dd.png)

The arrows are reference reaction events where the tail shows the initial cluster index (y-axis) and the head the final cluster index. Arrow color depends on proximity to a predicted reaction event. The dotted lines show time frames where the cluster index is stable and unchanging. The x-axis is not linear to exclude empty space.

4. Bond order, Fourier transform, 1st derivative column plot

![image](https://user-images.githubusercontent.com/20996215/122843443-5233ba00-d2b4-11eb-9195-59565d5f1415.png)

Plot that depicts how the predicted reaction events arise from the bond order time series. Each row is an atom pair. The leftmost column is the bond order time series of each atom pair in orange and the smoothed series in purple. The center column is the Fourier transform of the bond order time series with orange and purple same as the right. The black trace is the actual low-pass filter that is applied. The right column is the first time derivative (green) of the smoothed time series. The horizontal, dashed black lines are the applied thresholds and the extrema of the green trace beyond those thresholds (blue dots) are the locations of the predicted reaction events.

5. Reference reaction events, ranked by distance

![image](https://user-images.githubusercontent.com/20996215/122844176-eeaa8c00-d2b5-11eb-8e49-631831ca3ef3.png)

The x-axis is simply a count of sorted reference reaction events according to their proximity to predicted reaction events. This plot is intended to provide a quick snapshot of how much of the reference set is easily predicted by the method.
