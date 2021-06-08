#!/usr/bin/env python

#%matplotlib inline
import numpy as np
import pickle as pickle
import time
from forcebalance.molecule import Molecule
import argparse

from BOTS_scanner_class_FINAL import BOTS_scanner

# record program start time
t0 = time.time()
   
def main():
    # create arguments 
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--coordinates", default='coors.xyz', help='MD_coordinates')
    parser.add_argument("-b", "--raw_bond_order_list", default='bond_order.list', help='raw, unoptimized bond order information')
    parser.add_argument("-o", "--opt_bond_order_list", default='opt_bond_order.list', help='bond order information from the optimized trajectory frames')
    parser.add_argument("-r", "--reference_data", default='timeseries_1.0.data', help='cluster time series pickle \
    data from clustering method, used to parameterize time series method')
    parser.add_argument("-f", "--objective_function", choices=['bondwise', 'unified'], default='bondwise', help='type of objective\
    funtion used in scoring the BO and reference comparison. \'unified\' is traditional ROC and it \
    is found by calculating tpr/fpr for increasing window sizes and calculating the area under the curve of those points (plotted with tpr vs \
    fpr; \'bondwise\' is found similar to auc, but is scored on a bond-by-bond basis and is the fraction of reference transitions found)')
    parser.add_argument("-a", "--average_clusters", choices=[True, False], default=True, help='average the cluster bond order matrices \
    or take individual bond order matrices')
    parser.add_argument("-s", "--sigma_input", type=float, nargs="*", default=[40, 42.1, 1], help='accepts a single input for a single sigma \
    or accepts 3 inputs to create a range: starting sigma, ending sigma, increment value.')
    parser.add_argument("-t", "--threshold_input", type=float, nargs="*", default=[0.5, 0.71, 0.1], help='accepts a single input for \
    a single BO value or accepts 3 inputs to create a range: starting BO, ending BO, increment value.')
    args = parser.parse_args()
        
    # create the sigma_list
    if len(args.sigma_input) == 1:
        sigma_list = np.array(args.sigma_input)
    elif len(args.sigma_input) == 3:
        sigma_list = np.arange(*args.sigma_input)
    else:
        raise ValueError('incorrect number of sigma inputs')

    # create the BO_threshold_list
    if len(args.threshold_input) == 1:
        threshold_list = np.array(args.threshold_input)
    elif len(args.threshold_input) == 3:
        threshold_list = np.arange(*args.threshold_input)
    else:
        raise ValueError('incorrect number of threshold inputs')
    
    cluster_time_series = pickle.load(open(args.reference_data, 'rb'))
    # use the Forcebalance Molecule class on the .xyz data from the MD simulation
    m = Molecule(args.coordinates)
    # elements list (in index order) from the forcebalance Molecule class
    elements = m.elem

    sampling_rate = 10**15
    TEST_scanner = BOTS_scanner(elements=elements, cluster_time_series=cluster_time_series, bond_order_list_name=args.raw_bond_order_list, opt_bond_order_list_name=args.opt_bond_order_list, coors_list_name=args.coordinates, sampling_rate=sampling_rate, average_clusters=args.average_clusters, sigma_list=sigma_list, threshold_list=threshold_list)
    TEST_scanner.get_scores(args.objective_function)
    
    sig = 40.0
    thresh = 0.7

#    TEST_scanner.find_rxn_events(sig, thresh)
    TEST_scanner.make_heatmap(sigma_list, threshold_list)
    '''
    TEST_scanner.plot_regular_vs_FT_vs_deriv(sig, thresh)
    TEST_scanner.plot_bondwise_comparison(sig, thresh)
    TEST_scanner.plot_reference_distances(sig, thresh)
    TEST_scanner.plot_cluster_pairs_vs_bots(sig, thresh)
    TEST_scanner.plot_union_comparison_and_auc(sig, thresh)
    '''
    
    TEST_scanner.generate_all_plots(sig, thresh, args.objective_function)
    
    TEST_scanner.generate_output()

    
if __name__ == '__main__':
    main()


