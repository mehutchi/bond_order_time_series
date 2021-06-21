# bond_order_time_series

A time-saving method that uses bond order time series obtained from ab initio molecular dynamics to quickly and accurately predict temporal locations of reaction events. The method is tuned using two adjustable parameters to bring reaction event predictions into closer agreement with an ideal reference set of reaction events given infinite computing resources.

Publication of this method can be found here:

https://pubs.acs.org/doi/10.1021/acs.jctc.9b01039

Citation:

Bond-Order Time Series Analysis for Detecting Reaction Events in Ab Initio Molecular Dynamics Simulations
Marshall Hutchings, Johnson Liu, Yudong Qiu, Chenchen Song, and Lee-Ping Wang
Journal of Chemical Theory and Computation 2020 16 (3), 1606-1617
DOI: 10.1021/acs.jctc.9b01039

The BO_time_series_FINAL.py code requires 4 input files (samples provided). 

1. xyz coordinates (sample -> coors.xyz)
      
2. bond order list file (sample -> bond_order.list)

3. geometry optimized bond order list file (sample -> opt_bond_order.list)

4. cluster time series json file (sample -> timeseries.json)

Files 1 and 2 are easily obtained from an ab initio molecular dynamics simulation (this method is intended for a single reactive molecule, but the principles can carry over for more complex systems), but 3 and 4 require more effort. File 3 is a homemade file obtained by using each frame from file 1 as an input coordinate for a geometry optimization and creating a optimized bond order list file by appending the resulting bond order data from each simulation. The provided sample file 3 is the result of 2000 geometry optimzations. File 4 is a list of cluster indices according to how each optimized frame was assigned. This is obtained by creating a homemade optimized coordinates file (by appending the resulting geometries in order) and using a RMSD-esque bond order metric (described in the publication) to perform hierarchical clustering at a threshold of 1.0 (justified in the publication) to get the cluster assignments for each frame. File 4 is the resource-intensive (and impractical for everyday use) reference set of reaction events which helps parameterize this method.

Running the code with three sigma (smoothing parameter) inputs (start, stop, increment -> according to np.arange()) AND three threshold inputs creates a heat map of objective function scores. There are some options for the objective function, but we use the "bondwise" option (set as defauilt) as explained in the paper.

Smoothing parameter and threshold heatmap for sample data set:

![image](https://user-images.githubusercontent.com/20996215/122833449-d892d080-d2a1-11eb-87ca-79bb4fedeb9e.png)

