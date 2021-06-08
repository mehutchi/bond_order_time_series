#!/usr/bin/env python
import numpy as np
import os
import pickle as pickle
from scipy.signal import butter
from scipy.signal import freqz
import operator
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.colors as colors
import math
from operator import itemgetter
import statistics

from matplotlib.font_manager import findfont, FontProperties
font = findfont(FontProperties(family=['sans-serif']))
print('font ', font)

from transition_class import transition


def extract_two_atom_info(all_BO_list, BO_matrix):
    """Takes in a split bond order trajectory and the two atoms of interest as input and extracts BO information
    from each frame, appending a zero when the atom pair is not listed. Returns the bond orders and the element names.
    
    Example split bond order trajectory:
    
          185
    frame_0000 bond order list
        0     0    4.628  Fe  Fe
        0     1    0.400  Fe  Fe
        0     2    0.549  Fe  Fe
        0     3    0.496  Fe   C
        0     4    0.831  Fe   C
        0     5    0.842  Fe   C
        0     6    0.073  Fe   O
        0     7    0.110  Fe   O
        0     8    0.101  Fe   O
        0     9    0.091  Fe   C
        .
        .
        .
        
    Parameters
    ----------
    all_BO_list : np.array()
        Split list like the above example.
    BO_matrix : np.array()
        L x n x n matrix of zeros, where n is the number of atoms and L is the trajectory length.

    Returns
    -------
    BO_matrix : np.array()
        Filled in L x n x n matrix.
    """
    num_BOs = 0
    displacement = 2
    # iterate over every entry in the list that has a length of one (each entry that contains the number of lines 
    # for that particular frame)
    frame = 0 #assuming that the frame count starts at zero
    for i in range(len(all_BO_list)):
        if len(all_BO_list[i]) == 1:
            # record the number of BOs in this frame
            num_BOs = int(all_BO_list[i][0])
            # check the next j entries, where j is the number of bond orders in this frame, to determine whether
            # atom_one and atom_two are included in the BOs of that frame
            for j in range(num_BOs):
                if int(all_BO_list[i+j+displacement][0]) != int(all_BO_list[i+j+displacement][1]):
                    BO_matrix[frame][int(all_BO_list[i+j+displacement][0])][int(all_BO_list[i+j+displacement][1])] = float(all_BO_list[i+j+displacement][2])                
            print("Processed the %d data"%frame)
            frame += 1
    return BO_matrix

def list_splitter(raw_list):
    ''' function for splitting raw line-lists into lines of separate list elements

    Parameters
    ----------
    raw_list : list
        List to be split by elements.
    
    Returns
    -------
    split_list : list
        Resulting list split by elements.
    '''
    split_list = []
    # split a list (by spaces) into separate elements 
    for i in range(len(raw_list)):
        split_list.append(raw_list[i].split())
    return split_list

def get_matrix(traj_length, num_atoms, bond_order_file_name, pickle_name):
    ''' create a list of all bond order matrices and save that into a pickle file
    if it does not already exist
    
    Parameters
    ----------
    traj_length : int
        Trajectory length.
    num_atoms : int
        Number of atoms.
    bond_order_file_name : string
        Bond order file name.
    pickle_name : string
        File name to create or open.

    Returns
    -------
    finished_BO_matrix : np.array()
        Filled in L x n x n matrix, where n is the number of atoms and L is the trajectory length.
    '''    
    # if the pickle data for the BO_matrix already exists, then load it
    if os.path.exists(pickle_name):
        finished_BO_matrix = pickle.load(open(pickle_name, 'rb'))
    # if the pickle data for the BO_matrix does not already exist, create it
    else:
        # import the BO list data
        with open(bond_order_file_name) as MD_BO_file:
            MD_all_BO_block = MD_BO_file.readlines()
        MD_all_BO = np.array(list_splitter(MD_all_BO_block))
        # create a matrix of zeros in the shape of the trajectory of BO matrices
        BO_matrix = np.zeros((traj_length, num_atoms, num_atoms))
        # create BO matrix
        finished_BO_matrix = extract_two_atom_info(MD_all_BO, BO_matrix)
        # save the numpy array using the pickle method
        pickle.dump(finished_BO_matrix, open(pickle_name, 'wb'))
    return finished_BO_matrix

def get_raw_time_series(BO_matrix):
    ''' Create list of all raw time series that are ever above zero
    
    Parameters
    ----------
    BO_matrix : np.array()
        L x n x n matrix, where n is the number of atoms and L is the trajectory length.

    Returns
    -------
    BO_matrix.T[raw_cols,raw_rows] :  np.array()
        L x m array where m is the number of BO time series (that are ever above 0.0) and L is the trajectory length.
    raw_rows : list
        List of m remaining row element indices.
    raw_cols : list
        List of m remaining column element indices.
    '''
    ### assumes input matrix is either strictly upper or strictly lower triangular (no duplicated entries and no diagonal)
    # set True for every value above zero
    # one consequence of using the line below is that time series that are never above zero
    # are excluded, however, this is acceptable because those do not contain any information
    gt = BO_matrix > 0.0
    # set True for every trajectory that ever is above 0.0
    want = np.any(gt,axis=0)
    # record the rows and columns that are ever above 0.0
    raw_rows, raw_cols = np.nonzero(want)
    return BO_matrix.T[raw_cols,raw_rows], raw_rows, raw_cols

def find_cluster_BO_matrices(traj_length, num_atoms, BO_matrix, cluster_time_series):
    ''' average the BO matrices that belong to the same cluster
    
    Parameters
    ----------
    traj_length : int
        Trajectory length.
    num_atoms : int
        Number of atoms.
    BO_matrix : np.array()
        L x n x n matrix, where n is the number of atoms and L is the trajectory length.
    cluster_time_series : np.array()
        Time series of cluster numbers.
        
    Returns
    -------
    np.asarray(cluster_BO_matrices) : np.array()
        c x n x n averaged-by-cluster matrices, where c is the number of clusters.
    cluster_no_and_frames : list
        c x 2 array, where each element is a list of the form [cluster number, number of frames in cluster]
    '''
    # create output file documenting standard deviation of cluster BO matrices as well
    # as max difference of any atom pair in the cluster
    with open('cluster_std_dev.txt', 'w') as std_dev_file:
        # average the BO matrices that belong to the same cluster
        num_clusters = np.amax(cluster_time_series)
        print("there are %i clusters"%num_clusters)
        cluster_BO_matrices = []
        cluster_no_and_frames = []
        keep_best = 1
        index_diff = []
        # iterate over the number of clusters
        for j in range(1, num_clusters+1):
            # create an n x n matrix of zeroes
            single_BO_matrix = np.zeros((num_atoms, num_atoms))
            # collection of all the matrices belonging to the cluster, for purpose of finding standard deviation
            cluster_matrix_collection = []
            num_in_cluster = 0
            sd_list = []
            # iterate over every MD step (labeled with cluster number in cluster time series)
            for i in range(len(cluster_time_series)):
                # if the MD step's cluster number matches j, add the matrices
                if cluster_time_series[i] == j:
                    # add matrix belonging to cluster j to the others also belonging
                    single_BO_matrix = np.add(single_BO_matrix, BO_matrix[i])
                    # append matrix belonging to cluster j to a list for cluster j
                    cluster_matrix_collection.append(BO_matrix[i])
                    num_in_cluster += 1
            std_dev_file.write('Cluster # %i'%j)
            std_dev_file.write('\nAll standard deviations (of the BO matrix elements) within 50% of maximum standard deviation\n')
            # find standard deviation of the BO matrix
            std_of_cluster = np.asarray(cluster_matrix_collection).std(0)
            indices = (-std_of_cluster).argpartition(keep_best, axis=None)[:keep_best]
            x, y = np.unravel_index(indices, std_of_cluster.shape)
            # find maximum standard deviation
            matrix_max = std_of_cluster[x, y]
            # find the indices of all elements that are greater than 50% of the max value
            coords_of_greatest = [index for index, BO in np.ndenumerate(std_of_cluster) if BO > (matrix_max * 0.5)]
            # n x n matrix of max values of atom pairs in the list of BO matrices
            max_list = np.amax(cluster_matrix_collection, axis=0)
            # n x n matrix of min values of atom pairs in the list of BO matrices
            min_list = np.amin(cluster_matrix_collection, axis=0)
            diff = 0
            max_diff = 0
            for row in range(num_atoms):
                for col in range(row+1, num_atoms):
                    diff = abs(max_list[row][col] - min_list[row][col])
                    # record max difference of any atom pair between any two frames belonging to cluster j
                    if diff > max_diff:
                        max_diff = diff
                        index_diff = [row, col]
            std_dev_file.write('Cluster # %i maximum (BO matrix element) difference at (%i, %i) = %f\n'%(j, index_diff[0]+1, index_diff[1]+1, max_diff))
            # create list of highest crontributing standard deviations and their corresponding atom pairs
            for coord in coords_of_greatest:
                # adding one to each coord to be consistent in starting at one
                sd_list.append([coord[0]+1, coord[1]+1, std_of_cluster[coord[0], coord[1]]])
            # sort list by descending standard deviation
            sorted_sd = sorted(sd_list, key=itemgetter(2), reverse=True)
            counter = 0
            for item in sorted_sd:
                counter +=1
                # print atom pair then standard deviation
                std_dev_file.write('(%i, %i), %f'%(item[0], item[1], item[2]))
                if counter != len(sorted_sd):
                    std_dev_file.write('\n')
                elif counter == len(sorted_sd):
                    std_dev_file.write('\n\n')
            # append the cluster BO matrix to the list and divide by the appropriate number to complete the averaging process
            cluster_BO_matrices.append(np.divide(single_BO_matrix, float(num_in_cluster)))
            # create a list that keeps track of how many frames belong to each cluster
            cluster_no_and_frames.append([j, num_in_cluster])
    return np.asarray(cluster_BO_matrices), cluster_no_and_frames

def print_list(list_file):
    '''Convert a list to a string format, so lists can be included in the output file
    
    Parameters
    ----------
    list_file : list
        List to be converted into a string.
    Returns
    -------
    list_str : string
        Input list as a string (separated by commas).
    '''
    list_len = len(list_file)
    list_str = '[ '
    for i in range(list_len):
        list_str += str(list_file[i])
        # only add a comma and a space if not the last entry
        if i < (len(list_file) - 1):
            list_str += ', '
        if i == (len(list_file) - 1):
            list_str += ' ]'
    return list_str

def cluster_differences(cluster_BO_matrices):
    ''' create a dictionary where the key is a pair of clusters and the entry is the list
    of atom pair indices that contribute most to a change in bond order between the frames
    
    Parameters
    ----------
    cluster_BO_matrices : np.array()
        First output of find_cluster_BO_matrices(), c x n x n averaged-by-cluster matrices, where c is the number of clusters.
        
    Returns
    -------
    top_contributions : dict
        A dictionary of top contributions where the key is the pair of clusters and the entry is the list of pairs 
        of atom indices that contribute most to a change in bond order between the frames.
    '''
    num_clusters = len(cluster_BO_matrices)
    keep_best = 1
    top_contributions = {}
    # iterate over every cluster
    for i in range(num_clusters-1):
        # iterate over every j greater than i
        for j in range(i+1, num_clusters):
            # difference matrix
            diff_matrix = np.absolute(cluster_BO_matrices[i] - cluster_BO_matrices[j])
            # indices of maximum magnitude value
            indices = (-diff_matrix).argpartition(keep_best, axis=None)[:keep_best]
            x, y = np.unravel_index(indices, diff_matrix.shape)
            matrix_max = diff_matrix[x, y]
            # find the indices of all elements that are greater than 50% of the max value
            coords_of_greatest = [index for index, BO in np.ndenumerate(diff_matrix) if BO > (matrix_max * 0.5)]
            temp = []
            for coord in coords_of_greatest:
                temp.append(diff_matrix[coord[0], coord[1]])
            # increase atom indices by one (so that they start from one)
            adj_coords_of_greatest = [(index[0]+1, index[1]+1) for index in coords_of_greatest]
            top_contributions.update({(i+1, j+1) : adj_coords_of_greatest})
    return top_contributions

def remaining(relevant, r, c, threshold):
    '''Function that processes trajectories after smoothing. The list of
    relevant (reaction event-contributing) trajectories list is updated.
        
    Parameters
    ----------
    relevant : np.array()
        L x m array where m is the number of BO time series and L is the trajectory length.
    r : list
        List of m remaining row element indices.
    c : list
        List of m remaining column element indices.
    threshold : float
        Threshold.
        
    Returns
    -------
    relevant[rel_rows] : np.array()
        L x o array where o is the number of BO time series (that cross the threshold) and L is the trajectory length.
    r[rel_rows] : list
        List of o remaining row element indices.
    c[rel_rows] : list
        List of o remaining column element indices.
    rel_rows : list
        List of BO time series that cross the threshold.
    '''
    # set True for every value above the threshold
    # absolute value checks the positive threshold and negative threshold
    gt = abs(relevant) > threshold
    # set True for every entry (atom pair) that ever is above the BO_threshold and not always above the BO_threshold
    want = np.logical_and(np.any(gt,axis=1), np.logical_not(np.all(gt,axis=1)))
    # find the indicies of the post-smoothing relevant trajectories
    rel_rows = np.nonzero(want)
    return relevant[rel_rows], r[rel_rows], c[rel_rows], rel_rows

def max_finder(smoothed, thresh):
    '''This function takes the matrix of derivatives of smoothed functions and produces a cumulative list
    containing all the peaks (transitions) as well as lists by row (to be turned into a 'transition_matrix'
    later on for plotting purposes)
    
    Parameters
    ----------
    smoothed : np.array()
        L x o array where o is the number of BO time series (that cross the threshold) and L is the trajectory length.
    thresh : float
        Threshold value.
        
    Returns
    -------
    combined_peaks : list
        List of all transitions (reaction events).
    np.asarray(peaks_by_row) : np.array()
        i x o array where o is the number of BO time series (that cross the threshold) and i is the number of 
        transitions (reaction events) for that atom pair.
    '''
    num_series = len(smoothed)
    if num_series == 0:
        raise ValueError('check parameters, no relevant series')
    traj_length = len(smoothed[0])
    combined_peaks = []
    peaks_by_row = []
    # iterate over each time series
    for i in range(num_series):
        time_series_i = smoothed[i]
        peak_row = []
        # iterate over the length of the series
        for j in range(traj_length):
            # if positive
            if time_series_i[j] >= 0:
                # skipping the endpoints
                if j != 0 and j != (traj_length - 1):
                    # if the value is greater than its neighbors
                    if time_series_i[j] > time_series_i[j - 1] and time_series_i[j] > time_series_i[j + 1]:
                        if time_series_i[j] > thresh:
                            peak_row.append(j)
                            if j not in combined_peaks:
                                combined_peaks.append(j)
            # if negative        
            else:
                # skipping the endpoints
                if j != 0 and j != (traj_length - 1):              
                    # if the value is less than its neighbors
                    if time_series_i[j] < time_series_i[j - 1] and time_series_i[j] < time_series_i[j + 1]:
                        if time_series_i[j] < -thresh:
                            peak_row.append(j)
                            if j not in combined_peaks:
                                combined_peaks.append(j)
        peaks_by_row.append(peak_row)
    return combined_peaks, np.asarray(peaks_by_row)

def make_series(peak_locations, traj_length):
    ''' This Function is used to convert a list of (reaction event) locations into a full time series 
    with ones at the locations and zeros everywhere else
    
    Parameters
    ----------
    peak_locations : np.array()
        Second output of the max_finder() function. i x o array where o is the number of BO time series 
        (that cross the threshold) and i is the number of transitions (reaction events) for that atom pair.
    traj_length : int
        Trajectory length.
        
    Returns
    -------
    peak_list : list
        List with ones at reaction event locations and zeros everywhere else.
    '''
    peak_list = []
    for y in range(traj_length):
        if y in peak_locations:
            peak_list.append(1)
        else:
            peak_list.append(0)
    return peak_list

def bondwise_scoring(ref_transitions_atom_pairs, transition_matrix, r, c, traj_length):
    ''' objective function that compares a BOTS set to a reference set using bondwise criterion
    The windows are combined to eventually produce (TPR, FPR)
    
    Parameters
    ----------
    ref_transitions_atom_pairs : dict
        Dictionary where the key is a reference reaction event (transition) location and the value
        is a transition class object.
    transition_matrix : np.array()
        L x o array with ones for reaction events and zeros everywhere else, where o is the number of 
        BO time series (that cross the threshold) and L is the trajectory length.
    r : list
        List of m remaining row element indices.
    c : list
        List of m remaining column element indices.
    traj_length : int
        Trajectory length.
        
    Returns
    -------
    tpr_fpr : list
        List containing each [tpr, fpr] as elements.
    window_size : int
        Window size (+/- around each predicted reaction event).
    found_by : dict
        Dictionary where the keys are reference reaction events and the values is the predicted
        reaction event that finds that reference reaction event.
    '''
    tm = transition_matrix
    num_all_ref = len(ref_transitions_atom_pairs)
    zip_rc = list(zip(r, c))
    tpr_fpr = []
    stop = False
    window_size = 0
    tpr = 0 # the same as recall
    fpr = 0
    found_by = {}
    while stop == False:
        # grow the windows if the window_size is greater than zero
        if window_size > 0:
            # delete the leftmost column of the matrix of transition series
            delete_left = np.delete(tm, 0, 1)
            # insert a column of zeros on the right side
            delete_left = np.insert(delete_left, -1, 0, axis=1)
            # delete the rightmost column of the matrix of transition series
            delete_right = np.delete(tm, -1, 1)
            # insert a column of zeros on the left side
            delete_right = np.insert(delete_right, 0, 0, axis=1)
            # add the modified matrices together for a resulting window size increased by one
            tm = np.add(np.add(delete_left, delete_right), transition_matrix)
        # if the tpr is not yet equal to 1.0
        if tpr != 1.0:
            # determine 'found' transitions
            # iterate over each transition object
            for key, t in ref_transitions_atom_pairs.items():
                has_key = []
                # iterate over atom pairs associated with that transition
                for atompair in t.atompairs:
                    # if the reference atompair is also in the BOTS
                    if atompair in zip_rc:
                        # get the index
                        index = zip_rc.index(atompair)
                        # check the atom pair (row) and the transition (column), if a transition appears there (greater than zero)
                        # (if nonzero means that either a BOTS transition occurs there OR the time window has reached that point)
                        if tm[index][key] != 0 and t.found == False: # and t.found == False: comment this part out to show "all findings"
                            # mark transition as found
                            t.found = True
                            # find the BOTS transitions for that atom pair
                            locs_for_pair = np.nonzero(transition_matrix[index])
                            # for each of those transitions
                            for loc in locs_for_pair[0]:
                                # if the BOTS transition is within +/- the window_size from the key, append it to the list
                                if loc + window_size >= key and loc - window_size <= key:
                                    has_key.append(loc)
                # record which reference events are found by which BOTS events
                if has_key != []:
                    found_by.update({key : has_key})
            num_found = 0
            f_num_found = 0      
            # determine the number of reference events found as well as the total number of instances of reference events 
            # being found (f_num_found)
            for key, t in ref_transitions_atom_pairs.items():
                if t.found == True:
                    num_found += 1
                    f_num_found += len(t.atompairs)
        # number of covered frames (any frame covered by a window in any series)
        num_covered = np.count_nonzero(np.sum(tm, axis=0))
        # fpr is the union of all the windowed areas divided by the (traj length - the reference transitions)
        total_fp_frames = float(traj_length - num_all_ref)
        total_fp_covered = float(num_covered - num_found) # this can be larger than total_fp_frames when some of the transitions are not found at all
        # set fpr to 1.0 if the entire total space (or more for those cases that don't find all the transitions) is covered
        if total_fp_covered >= total_fp_frames:
            fpr = 1.0
        else:
            fpr = total_fp_covered/total_fp_frames
        tpr = float(num_found)/float(num_all_ref) # also recall
        tpr_fpr.append([tpr, fpr])
        # shortcut to skip to the endpoint of the AUC plot
        # saves time so you don't have to calculate the potentially thousands of points before the endpoint, 
        # while not affecting the final score
        if tpr == 1.0:
            fpr = 1.0
            tpr_fpr.append([tpr, fpr])
        # exit loop once last point has been determined
        if fpr == 1.0:
            stop = True
        window_size += 1
    print('num_covered', num_covered)
    # reset the transition objects' 'found' attributes to False       
    for key, t in ref_transitions_atom_pairs.items():
        t.found = False
    return tpr_fpr, window_size, found_by

def unified_scoring(transition_matrix, binary_cluster_time_series):
    '''Function for traditional "unified" ROC (not "bondwise")
    
    Parameters
    ----------
    transition_matrix : np.array()
        L x o array with ones for reaction events and zeros everywhere else, where o is the number of 
        BO time series (that cross the threshold) and L is the trajectory length.    
    binary_cluster_time_series : np.array()
        Length L array with ones for reference reaction events (cluster transitions) and zeros everywhere else.
        
    Returns
    -------
    tpr_fpr : list
        List containing each [tpr, fpr] as elements.
    window_size : int
        Window size (+/- around each predicted reaction event).
    '''
    
    orig = np.any(transition_matrix, axis=0)
    btm = np.any(transition_matrix, axis=0)
    trajectory_len = len(transition_matrix[0])
    num_all_ref = np.count_nonzero(binary_cluster_time_series)
    print('num bots events', np.count_nonzero(transition_matrix))
    tpr_fpr = []
    stop = False
    window_size = 0
    tpr = 0
    fpr = 0
    while stop == False:
        if window_size > 0:
            # delete the leftmost column of the matrix of transition series
            delete_left = np.delete(btm, 0)
            # insert a column of zeros on the right side
            delete_left = np.insert(delete_left, -1, 0)
            # delete the rightmost column of the matrix of transition series
            delete_right = np.delete(btm, -1)
            # insert a column of zeros on the left side
            delete_right = np.insert(delete_right, 0, 0)
            # add the modified matrices together for a resulting window size increased by one
            btm = np.add(np.add(delete_left, delete_right), orig)
        # number of covered frames (any frame covered by a window in any series)
        num_covered = np.count_nonzero(btm)
        # create a set of reaction events 
        set_cluster_reaction_events = set(np.nonzero(binary_cluster_time_series)[0])
        # create a set containing the indices of frames covered by windows
        set_btm = set(np.nonzero(btm)[0])
        # number of found transitions is the intersection of the set of window-covered frames and the set of rxn events
        num_found = len(set_btm.intersection(set_cluster_reaction_events))
        # fpr is the union of all the windowed areas divided by the (traj length - the reference transitions)
        total_fp_frames = float(trajectory_len - num_all_ref)
        total_fp_covered = float(num_covered - num_found) # this can be larger than total_fp_frames when some of the
        # transitions are not found at all
        # set fpr to 1.0 if the entire total space (or more for those cases that don't find all the transitions) is covered
        if total_fp_covered >= total_fp_frames:
            fpr = 1.0
        else:
            fpr = total_fp_covered/total_fp_frames
        tpr = float(num_found)/float(num_all_ref)
        tpr_fpr.append([tpr, fpr])
        # shortcut to skip to the endpoint of the AUC plot
        # saves time so you don't have to calculate the potentially thousands of points before the endpoint, 
        # while not affecting the final score
        if tpr == 1.0:
            fpr = 1.0
            tpr_fpr.append([tpr, fpr])
        # exit loop once last point has been determined
        if fpr == 1.0:
            stop = True
        window_size += 1
    print('num_covered', num_covered)
    print("tpr ", tpr, " fpr ", fpr)
    return tpr_fpr, window_size

def get_AUC(tpr_fpr):
    ''' calculates the area under the curve (AUC) for a given series of (tpr, fpr) points
    
    Parameters
    ----------
    tpr_fpr : list
        List containing each [tpr, fpr] as elements.

    Returns
    -------
    total_area : float
        Area under the parametric curve (TPR, FPR).
    '''
    tpr, fpr = zip(*tpr_fpr)
    total_area = 0
    for count, p in enumerate(fpr[:-1]):
        x1 = fpr[count]
        x2 = fpr[count+1]
        y1 = tpr[count]
        y2 = tpr[count+1]
        # Area found by using a trapezoid.
        area = (x2 - x1)*y1 + 0.5*(y2-y1)*(x2-x1)
        total_area += area
    return total_area # AUC score

def make_ref_dictionary(traj_length, num_atoms, cluster_time_series, opt_BO_matrix, average=True):
    ''' creates a dictionary containing (reference) transition locations as keys and transition objects as values.
    Transition objects contain the frame number, a list of associated atom pairs, and a boolean to track the
    'found' status.
    
    Parameters
    ----------
    traj_length : int
        Trajectory length.
    num_atoms : int
        Number of atoms in the system.
    cluster_time_series : np.array
        Time series of cluster numbers.  
    opt_BO_matrix : np.array()
        L x n x n matrix, where n is the number of atoms and L is the trajectory length.
    average : bool, default=True
        Determines whether or not cluster BO matrices are averaged.
        
    Returns
    -------
    all_ref_transitions : dict
        Dictionary where the key is a reference reaction event (transition) location and the value
        is a transition class object.
    no_and_frames : list
        c x 2 list where each entry is [cluster number, number of frames belonging to cluster].
    '''
    # create a list containing the average BO matrices for the different clusters
    cluster_BO_matrices, no_and_frames =  find_cluster_BO_matrices(traj_length, num_atoms, opt_BO_matrix, cluster_time_series)
    # create a dictionary where the keys are the cluster pairs and the values are each a list of the top atom pairs 
    # that contribute the most magnitude to the difference matrix
    cluster_diff = cluster_differences(cluster_BO_matrices)
    # create a dictionary containing transition locations as keys and transition objects as values
    all_ref_transitions = {}
    # cluster_diff -> key (cluster_pair) ; value ([list of top atom pairs])
    # iterate over every MD frame
    for i in range(len(cluster_time_series)-1): #skipping last frame of data
        ival = cluster_time_series[i]
        i_plus_1val = cluster_time_series[i+1]
        cluster_pair = []
        # if cluster of frame i is LESS THAN the cluster of frame i+1
        if ival < i_plus_1val:
            # keep the order the same to stay consistant with dictionary keys
            cluster_pair = [ival, i_plus_1val]
        # if cluster of frame i is GREATER THAN the cluster of frame i+1
        elif ival > i_plus_1val:
            # swap the order of i and i+1 to stay consistant with dictionary keys
            cluster_pair = [i_plus_1val, ival]
        # if the two adjacent clusters are different
        if ival != i_plus_1val:
            if average == True:
                # find the atom pairs that contribute most to the BO difference matrix between i and i+1
                for j in cluster_diff[tuple(cluster_pair)]:
                    # if the transition is already in the dictionary
                    if i in all_ref_transitions:
                        # if the transition already has a key, append the next atom pair into the attribute transition.atompairs
                        all_ref_transitions[i].atompairs.append(j)
                    # if the transition DOES NOT already have a key, create one and a transition object as the value
                    else:
                        all_ref_transitions.update({i : transition(i, [j])})
            # SINGLE FRAME
            elif average != True:
                # directly create all_ref_transitions without other methods
                keep_best = 1
                # calculate difference matrix for frame i and i+1
                diff_matrix = np.absolute(opt_BO_matrix[i] - opt_BO_matrix[i+1])
                indices = (-diff_matrix).argpartition(keep_best, axis=None)[:keep_best]
                x, y = np.unravel_index(indices, diff_matrix.shape)
                matrix_max = diff_matrix[x, y]
                # find the indices of all elements that are greater than 50% of the max value
                coords_of_greatest = [index for index, BO in np.ndenumerate(diff_matrix) if BO > (matrix_max * 0.5)]
                temp = []
                for coord in coords_of_greatest:
                    temp.append(diff_matrix[coord[0], coord[1]])
                # increase atom indices by one (so that they start from one)
                adj_coords_of_greatest = [(index[0]+1, index[1]+1) for index in coords_of_greatest]
                all_ref_transitions.update({i : transition(i, adj_coords_of_greatest)})
    return all_ref_transitions, no_and_frames

def truncate_colormap(cmap, minval=0.0, maxval=1.0, resolution=100, n_separation=100):
    '''
    Parameters
    ----------
    cmap : cmap
        Input colormap to be adjusted.
    minval : float, default=0.0
        Minimum cutoff for colormap.
    maxval : float, default=1.0
        Maximum cutoff for colormap.
    resolution : int, default=100
        Resolution.
    n_separation : int, default=100
        Separtation.
        
    Returns
    -------
    new_cmap : cmap
        New, adjusted colormap.
    '''
    # Function to truncate a color map
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, resolution)), N=n_separation)
    return new_cmap



class BOTS_scanner:
    def __init__(self, elements, cluster_time_series, bond_order_list_name, opt_bond_order_list_name, coors_list_name, sampling_rate, average_clusters, sigma_list=[], threshold_list=[]):
        ''' initialize a BOTS_scanner object with a sigma list and a threshold list
        
        Parameters
        ----------
        elements : list
            List of each individual element matching the indices correctly.
        cluster_time_series : np.array
            Time series of cluster numbers.
        bond_order_list_name : string
            Name of bond order list file.
        opt_bond_order_list_name : string
            Name of optimized bond order list file.
        coors_list_name : string
            Name of AIMD trajectory coordinates.
        sampling_rate : int
            AIMD sampling rate, per second.
        average_clusters : bool
            If true, average the BO matrices of frames belonging to the same cluster when creating 
            the bond-wise reference reaction event set.
        sigma_list : np.array
            List of sigma values to scan across.
        threshold_list : np.array
            List of threshold values to scan across.
        
        Returns
        -------
        
        '''
        self.sigma_list = sigma_list
        self.threshold_list = threshold_list
#        self.mult_threshold_list = 
        self.elements = elements
        self.num_atoms = len(elements)
        # length of the trajectory
        self.traj_length = len(cluster_time_series)
        # conversion factor for f to cm^-1
        self.conversion = 33355.0
        self.cluster_time_series = cluster_time_series
        self.num_clusters = max(cluster_time_series)
        # MD steps per second, default used is 10**15
        self.sampling_rate = sampling_rate
        # order of the Butterworth Filter used in the low-pass filter. An order of 6 was found to be optimal.
        self.order = 6
        # create the trajectory of BO matrices
        self.BO_matrix = get_matrix(self.traj_length, self.num_atoms, bond_order_list_name, 'all_BO_matrices.p')
        # create the trajectory of optimized BO matrices
        self.opt_BO_matrix = get_matrix(self.traj_length, self.num_atoms, opt_bond_order_list_name, 'all_opt_BO_matrices.p')
        # create the raw time series
        self.all_raw_time_series, self.raw_rows, self.raw_cols = get_raw_time_series(self.BO_matrix)
        # all rows and columns of time series. Adding one to each element so atom indices begin at one
        self.raw_rows = np.array([x+1 for x in self.raw_rows])
        self.raw_cols = np.array([x+1 for x in self.raw_cols])
        self.binary_cluster_time_series = np.diff(cluster_time_series).astype(bool)
        # number of reference reaction events
        self.num_all_ref = np.count_nonzero(self.binary_cluster_time_series)
        self.parameter_data_cache = {}
        # self.all_ref_transitions -> dictionary containing transition locations as keys and transition objects as values
        # self.no_and_frames -> a list that keeps track of how many frames belong to each cluster. 
        # Each element: [cluster, number of MD frames belonging to that cluster]            
        self.all_ref_transitions, self.no_and_frames = make_ref_dictionary(self.traj_length, self.num_atoms, self.cluster_time_series, self.opt_BO_matrix, average_clusters)
        with open('ref_rxn_lifetimes.txt', 'w') as lifetime_file:
            lifetime_file.write('List of reference reaction event locations and their lifetimes')
            lifetime_file.write('\n\'LEFT CLUSTER\' and \'RIGHT CLUSTER\' refer to the clusters of the reference reaction event transition')
            lifetime_file.write('\n\n\t\t\tLIFETIME\t\t\t\tCLUSTER TRANSITION')
            lifetime_file.write('\n\tRXN LOC\t\t(backward,forward)\tPREV. CLUSTER\t(left->right)\tNEXT CLUSTER')
#            lifetime_file.write('\n\tRXN LOC\t\tLIFETIME\tLEFT CLUSTER\tRIGHT CLUSTER\tNEXT CLUSTER')
            ref_lifetimes = []
            counter = 0
            # find the lifetime of each reference reaction event
            # iterate over each ref rxn event location
            for x, object_x in self.all_ref_transitions.items():
                counter += 1
                # find forward distance
                fd = 0
                fd_step = x+1
                # keep stepping forward while the cluster numbers are different ## or fd_step <= self.traj_length-1:
                while self.cluster_time_series[fd_step] == self.cluster_time_series[x+1] and fd_step <= self.traj_length-2:
                    fd_step += 1                        
                fd = fd_step-(x+1)
                # find forward distance
                rv = 0
                rv_step = x-1
                # keep stepping backward while the cluster numbers are different
                while self.cluster_time_series[rv_step] == self.cluster_time_series[x] and rv_step > 0:
                    rv_step -= 1
                rv = abs(rv_step-x)
    #            if fd <= lifetime or
                # ref rxn event, lifetime, leftmost transition cluster, rightmost transition cluster, next cluster
                print('checking for reaction event %i'%x)
                ref_lifetimes.append([x, min(fd,rv), self.cluster_time_series[x], self.cluster_time_series[x+1], self.cluster_time_series[fd_step]])
#                lifetime_file.write('\n\t%i\t\t%i\t\t%i\t\t%i\t\t%i'%(x, min(fd,rv), self.cluster_time_series[x], self.cluster_time_series[x+1], self.cluster_time_series[fd_step]))
                lifetime_file.write('\n\t%i\t\t%i,%i\t\t\t%i\t\t%i,%i\t\t%i'%(x, rv,fd, self.cluster_time_series[rv_step], self.cluster_time_series[x], self.cluster_time_series[x+1], self.cluster_time_series[fd_step]))

        
        self.x_label = 'AIMD step (fs)'
        
        # write output file
        self.output = []
        self.output.append('BOTS code Output File (Version 5 code)')
        self.output.append('\n\n***** INPUT FILES *****')
        self.output.append('\nRaw Bond Order File: ' + bond_order_list_name)
        self.output.append('\nOptimized Bond Order File: ' + opt_bond_order_list_name)
        self.output.append('\n\n***** PROGRAM SETTINGS *****')
        if average_clusters == True:
            self.output.append('\nCluster bond order matrices averaged')
        elif average_clusters != True:
            self.output.append('\nSingle frame (not averaged) bond order matrices')
        self.output.append('\n\n***** SMOOTHING AND THRESHOLD PARAMETERS *****')
        self.output.append('\nOrder(s) used for low-pass filter:\n')
        self.output.append(str(self.order))
        self.output.append('\nHere is the list of sigma values:\n')
        self.output.append(print_list(sigma_list))
        self.output.append('\nHere is the list of threshold values:\n')
        self.output.append(print_list(threshold_list))
        self.output.append('\n\n***** MD AND MOLECULE INFO *****')
        self.output.append('\nNumber of atoms: ' + str(self.num_atoms))
        num_pairs = self.num_atoms*(self.num_atoms -1)*0.5
        self.output.append('\nNumber of atom pairs: ' + str(num_pairs))
        self.output.append('\nAtom list: ')
        self.output.append(print_list(self.elements))
        self.output.append('\nNumber of frames in MD trajectory: ' + str(self.traj_length))
        self.output.append('\n\n***** PI INVARIANT INFO *****')
        self.output.append('\n(%i Total Clusters)'%len(self.no_and_frames))
        for i in self.no_and_frames:
            self.output.append('\nCluster %i contains %i frames'%(i[0], i[1]))
        self.output.append('\n\nBond-by-bond information')
        self.output.append('\nThere are %i total transitions in the gold standard data'%self.num_all_ref)
        temp_unordered = []
        self.output.append('\n\nTemporal Transition Location\t\tCluster Transition\t\tAtom Pairs Associated with Transition')
        for key, value in self.all_ref_transitions.items():
            new_string = "\n\t"+ str(key) + "\t\t\t\t" + str(self.cluster_time_series[key]) + ", " + str(self.cluster_time_series[key+1]) + "\t\t\t\t" + str(value.atompairs)
            temp_unordered.append([key, new_string])
        # sort reference reaction events (RRE) transitions before printing in output file
        temp_sorted = sorted(temp_unordered, key=operator.itemgetter(0))
        for t in temp_sorted:
            self.output.append(t[1])
        # sort RRE atompairs
        ref_atompairs_in_bots = []
        for key, value in self.all_ref_transitions.items():
            for v in value.atompairs:
                if v not in ref_atompairs_in_bots:
                    ref_atompairs_in_bots.append(v)
        ref_atompairs_in_bots = sorted(ref_atompairs_in_bots, key=operator.itemgetter(0, 1))
        bots_string = '\n\nAtom pairs associated with reference reaction event transitions: ' + print_list(ref_atompairs_in_bots)
        self.output.append(bots_string)

    def generate_output(self):
        ''' writes the output file
        
        Parameters
        ----------

        Returns
        -------
        
        Files Created
        -------------
        BOTS_output.txt
            Output file.
        '''
        # all of the "outfile.append() lines are for creating an output document containing the 
        # relevant information from the run
        out_name = 'BOTS_output.txt'
        with open(out_name, 'w') as out_file:
            for i in self.output:
                out_file.write(i)
    
    def get_scores(self, score_process, matrix_pickle_name='score_matrix.p', return_rxn_events = False):
        ''' take difference between adjacent step cluster numbers and convert frame entries to booleans. 
        Resulting array is zeros for no reaction events and ones for reaction events.
        
        Parameters
        ----------
        score_process : string
            Input 'bondwise' for bond-wise objective function and 'unified' for unified objective function.
        matrix_pickle_name : string
            Name of pickled score matrix file.
        return_rxn_events : bool
            Returns a list of detected reaction events.
        
        Returns
        -------
        
        Files Created
        -------------
        named with matrix_pickle_name, default='score_matrix.p'
            Pickled score matrix, used to create a heat map of sigma + threshold parameters.
        '''
        print("TOTAL NUMBER OF REFERENCE TRANSITIONS ", self.num_all_ref)
        self.score_matrix = np.zeros([len(self.threshold_list), len(self.sigma_list)])
        # create BOTS dictionary data, if it does not already exist
        if os.path.exists('BOTS_dictionary.p'):
            print("\nUSING DICTIONARY\n")
            dictionary = pickle.load(open('BOTS_dictionary.p'))
            dictionary_exists = True# should be set to 'True', this is a work-around for faster testing True
        else:
            print("\nNEED TO MAKE DICTIONARY\n")
            dictionary_exists = False
        add_to_cache = False
        # iterate over each sigma value
        for i, sigma in enumerate(self.sigma_list):
            # generate smoothed and derivative time series
            derivative_time_series, smoothed_time_series = self.derivative(sigma)
            # iterate over each threshold
            for j, j_threshold in enumerate(self.threshold_list):
                # generate objective function score from sigma and threshold
                score = self.single_score(sigma, j_threshold, derivative_time_series, smoothed_time_series, add_to_cache, return_rxn_events, score_process)
                self.score_matrix[j, i] = score
        self.matrix_pickle_name = matrix_pickle_name
        print("MADE IT TO PICKLE PORTION")
        pickle.dump(self.score_matrix, open(self.matrix_pickle_name, 'wb'))
            
    """ NEW, FINAL VERSION of low_pass_smoothing"""
    def low_pass_smoothing_NEWER(self, sigma):
        """ 
        Low-pass smoothing function for bond order or interatomic distance time series.
        
        Parameters
        ----------
        sigma : float
            Filter roll-off frequency expressed in wavenumbers.
            
        Returns
        -------
        filtered : np.ndarray
            Signal after lowpass filter has been applied, in the same shape as all_raw_time_series
        freqx : np.ndarray
            Frequency axis for plotting spectra in units of cm^-1
        ft_original : np.ndarray
            Fourier-transform of doubled signal truncated to original signal length
        ft_filtered : np.ndarray
            Fourier-transform of doubled signal with filter applied, truncated to original signal length
        """
        
        # Calculate frequency cutoff in units of the Nyquist frequency (half of the sampling rate.)
        # First convert to units of the sampling rate (inverse timestep) then multiply by 2:
        # 
        # sigma      1 cm       dt_fs * fs        
        # ----- * ---------- * ------------ * 2 = low_cutoff
        #  cm     33500 * fs        dt            
        #
        # 1) Cutoff frequency in cm^-1
        # 2) Cutoff frequency in fs^-1
        # 3) Cutoff frequency in units of sampling rate dt^-1
        # 4) Cutoff frequency in units of Nyquist frequency
        
        time_series = self.all_raw_time_series
        low_cutoff = float(sigma)/self.conversion * 2.0
        
        # Create Butterworth filter coefficients
        b, a = butter(self.order, low_cutoff, btype='low')
        
        reflect = True
        if reflect:
            # Create the doubled time series
            reflected = np.fliplr(time_series)
            # remove the enpoints of the reflected
            reflected = np.delete(reflected, -1, axis=1)
            reflected = np.delete(reflected, 0, axis=1)
            # attach the signal end-to-end with its reflection. The purpose of this is is get rid of the tailing issue
            doubled_series = np.hstack((time_series, reflected))
            new_len = len(doubled_series[0]) # length of doubled end-to-end time series
            # list of elements that make up the reflected portion (to be used to remove the reflected portion later on)
            removal_list = range(self.traj_length, new_len)
            # create the frequency portion (for plotting)
            w, h = freqz(b, a, worN=new_len, whole=True)
            # fast fourier transform
            ft = np.fft.fft(doubled_series)
            ft_filtered = ft*abs(h)
            filtered = np.fft.ifft(ft_filtered)
            # Delete the data from the reflection addition
            filtered = np.delete(filtered, removal_list, axis=1)
            ft_original = np.delete(ft, removal_list, axis=1)
            ft_filtered = np.delete(ft_filtered, removal_list, axis=1)
            w = np.delete(w, removal_list)
            h = np.delete(abs(h), removal_list)
            freqx = w*self.conversion/(2*np.pi)
        else:
            # create the frequency portion (for plotting)
            w, h = freqz(b, a, worN=self.traj_length, whole=True)
            abs_of_h = abs(h) # butterworth filter
            # fast fourier transform
            ft_original = np.fft.fft(time_series)
            # multiply the FT by the filter to filter out higher frequencies
            ft_filtered = ft_original*abs_of_h
            filtered = np.fft.ifft(ft_filtered)
            freqx = w*self.conversion/(2*np.pi)
            
        return filtered, ft_original, ft_filtered, h, freqx

    def derivative(self, sigma):
        ''' 
        Method that finds the 1st derivative of a smoothed BOTS
                
        Parameters
        ----------
        sigma : float
            Filter roll-off frequency expressed in wavenumbers.
        
        Returns
        -------
        np.asarray(full_list) : 2D np.array
            All the first time derivatives of the filtered time series. 
        traj_list : 2D np.array
            All the low-pass filtered time series.
        '''
        traj_list, self.ft, self.smooth_ft, self.modified_filter, self.freqx = self.low_pass_smoothing_NEWER(sigma)
        # number of time series
        num_series = len(traj_list)
        traj_length = len(traj_list[0])
        full_list = []
        # iterate over the number of time series
        for i in range(num_series):
            # time derivative at point j
            t_prime = 0.0
            # collection of discrete derivative points
            t_prime_list = []
            # time series of interest
            time_series_i = traj_list[i]
            # iterate over the length of the series
            for j in range(traj_length):
                # use forward difference for first point
                if j == 0:
                    t_prime = (time_series_i[j + 1] - time_series_i[j])/2
                # use reverse difference for last point
                elif j == (traj_length - 1):
                    t_prime = (time_series_i[j - 1] - time_series_i[j])/2
                # use central difference derivative for all the middle points
                else: 
                    t_prime = (time_series_i[j + 1] - time_series_i[j - 1])/2
                t_prime_list.append(t_prime)
            # collection of all the derivative time series
            full_list.append(t_prime_list)
        return np.asarray(full_list), traj_list
    
    def single_score(self, sigma, threshold, derivative_time_series, smoothed_time_series, add_to_cache, return_rxn_events=False, score_process='bondwise'):
        ''' calculate and return the objective function score of a single sigma and threshold parameter combination
        
        Parameters
        ----------
        sigma : float
            Filter roll-off frequency expressed in wavenumbers.
        threshold : float
            Value is multiplied by sigma and is a threshold to identify extrema in the first time derivative of the BO time series. 
        derivative_time_series : np.array
            All the first time derivatives of the filtered time series. 
        smoothed_time_series : np.array
            All the low-pass filtered time series.
        add_to_cache : bool
            Determines whether or not to save the parameters for plotting. Parameters are saved as an attribute in a dictionary.
        return_rxn_events : bool, default=False
            Creates a file with the predicted reaction events for the given parameter combination.
        score_process : string, default='bondwise'
            Input 'bondwise' for bond-wise objective function and 'unified' for unified objective function.
            
        Returns
        -------
        score : float
            Value of the objective function (bond-wise or unified) for the given combination of sigma and threshold.

        Files Created
        -------------
        'rxn_event_locations_for_sigma%.1f_thresh%.1f.txt'%(sigma, original_threshold) 
            Temporal locations of predicted reaction events.       
        '''
        original_threshold = threshold
        # DEFAULT (preferred) is 'bondwise'
        # 'bond-wise criterion' objective function
        # convert threshold from a number into a multiple of sigma
        threshold = threshold/self.conversion*sigma
        r = self.raw_rows
        c = self.raw_cols
        relevant_derivative, smooth_rows, smooth_cols, relevant_rows = remaining(derivative_time_series, r, c, threshold)
        if len(relevant_derivative) != 0:
            peaks, peaks_by_row = max_finder(relevant_derivative, threshold)
            tm = []
            for y in peaks_by_row:
                # create transition matrix
                tm.append(make_series(y, self.traj_length))
            transition_matrix = np.array(tm)
            # if the transition matrix is blank, set tpr_vs_fpr, window_size, and score
            if len(transition_matrix) == 0:
                tpr_vs_fpr = np.zeros(self.traj_length)
                window_size = self.traj_length
                score = 0
            # otherwise run bondwise_scoring function
            else:
                if score_process == 'bondwise':
                    tpr_vs_fpr, window_size, found_by = bondwise_scoring(ref_transitions_atom_pairs=self.all_ref_transitions, transition_matrix=transition_matrix, r=smooth_rows, c=smooth_cols, traj_length=self.traj_length)
                    score = get_AUC(tpr_vs_fpr)
                if score_process == 'unified':
                    tpr_vs_fpr, window_size = unified_scoring(transition_matrix=transition_matrix, binary_cluster_time_series=self.binary_cluster_time_series)
                    score = get_AUC(tpr_vs_fpr)

        # if there are no peaks after smoothing, assign a score of zero
        else:
            score = 0
            if add_to_cache == True:
                raise ValueError("parameter combination yields no reaction event predictions")

        print("\nhere is the current score: ", score)
        print("for sigma=", sigma, " thresh=", original_threshold, "\n")#, "\n"
        # if true, then create a cache attribute containing the important information relating to the single score
        if add_to_cache == True:
            sig_thresh_dict = {}
            sig_thresh_dict.update({'tpr_vs_fpr' : tpr_vs_fpr})
            sig_thresh_dict.update({'transition_matrix' : transition_matrix})
            binary_BO_time_series = np.any(transition_matrix, axis=0)
            rxn_event_indices = np.nonzero(binary_BO_time_series)
            sig_thresh_dict.update({'rxn_event_indices' : list(rxn_event_indices[0])})
            print('rxn_event_indices', list(rxn_event_indices[0]))
            sig_thresh_dict.update({'window_size' : window_size})
            sig_thresh_dict.update({'score' : score})
            sig_thresh_dict.update({'r' : smooth_rows})
            sig_thresh_dict.update({'c' : smooth_cols})
            sig_thresh_dict.update({'relevant_rows' : relevant_rows})
            sig_thresh_dict.update({'relevant_smoothed' : smoothed_time_series[relevant_rows]})
            sig_thresh_dict.update({'relevant_derivative' : relevant_derivative})
            sig_thresh_dict.update({'peaks_by_row' : peaks_by_row})
            sig_thresh_dict.update({'score_process' : score_process})
            if score_process == 'bondwise':
                # dictionary where the key is the reference transition, and the value is the bots transition that finds it
                sig_thresh_dict.update({'found_by' : found_by})
            self.parameter_data_cache.update({(sigma, original_threshold) : sig_thresh_dict})
            if return_rxn_events == True:
                with open('rxn_event_locations_for_%s_sigma%.1f_thresh%.1f.txt'%(score_process, sigma, original_threshold), 'w') as rxn_event_file:
                    rxn_event_file.write('Total trajectory length %i'%self.traj_length)
                    rxn_event_file.write('\n\nBOTS predicted reaction event locations using parameter combination sigma=%.1f cm^-1 and threshold=%.1fxsigma'%(sigma, original_threshold))
                    rxn_event_file.write('\n\n%s'%print_list(rxn_event_indices[0]))
        return score
    
    def check_parameter_data_cache(self, sigma, threshold, score_process='bondwise', add_to_cache = True, return_rxn_events=True):
        ''' The purpose of this method is to generate the data for a given combination of sigma and threshold
        and to add the data to a cache dictionary attribute to be used in the plotting methods.
        
        Parameters
        ----------
        sigma : float
            Filter roll-off frequency expressed in wavenumbers.
        threshold : float
            Value is multiplied by sigma and is a threshold to identify extrema in the first time derivative of the BO time series. 
        score_process : string, default='bondwise'
            Input 'bondwise' for bond-wise objective function and 'unified' for unified objective function.
        add_to_cache : bool
            Determines whether or not to save the parameters for plotting. Parameters are saved as an attribute in a dictionary.
        return_rxn_events : bool, default=False
            Creates a file with the predicted reaction events for the given parameter combination.
        
        Returns
        -------
        simply exits the function
        '''
        # if the entry already exists, simply exit method
        if (sigma, threshold) in self.parameter_data_cache and self.parameter_data_cache[(sigma, threshold)]['score_process'] == score_process:
            return
        # if the entry doesn't exist, run single_score() method with 'add_to_cache' set to True
        else:
            derivative_time_series, smoothed_time_series = self.derivative(sigma)
            self.single_score(sigma, threshold, derivative_time_series, smoothed_time_series, add_to_cache, return_rxn_events, score_process)
        return

    ''' not currently in use, not sure if needed
    '''
    def load_scores(self, matrix_pickle_name):
        if os.path.exists(matrix_pickle_name):
            self.score_matrix = pickle.load(open(matrix_pickle_name, 'rb'))
        # if the pickle data for the score-matrix doesn't already exist
        else:
            print("Pickle file %s not found. Need to run BOTS_scanner.get_scores() to generate scores pickle file"%matrix_pickle_name)
        return

    def make_heatmap(self, sigma_list, BO_threshold_list, matrix_pickle_name='score_matrix.p'):
        ''' create a heatmap plot for parameterizing sigma and threshold
        
        Parameters
        ----------
        heat_map_title : string
            Name of heat map file.
        sigma_list : np.array
            List of sigma parameters.
        BO_threshold_list : np.array
            List of threshold parameters.
        matrix_pickle_name : string
            Name of pickled score matrix file.
            
        Returns
        -------
        
        Files Created
        -------------
        heat_map_title
            Heat map of sigma and threshold combinations.        
        '''
        # if the pickle data for the score_matrix already exists, then load it
        if os.path.exists(matrix_pickle_name):
            score_matrix = pickle.load(open(matrix_pickle_name, 'rb'))
        # if the pickle data for the score-matrix doesn't already exist
        else:
            # still need to figure out what to do here
            score_matrix = []
        color_min = 0.5
        color_max = 1.0
#        rot_box_scores = np.rot90(np.array(score_matrix))
        rot_box_scores = np.flipud(np.array(score_matrix)) # flip for purpose of fitting
        num_rows = len(rot_box_scores)
        num_cols = len(rot_box_scores[0])
        '''
        if len(sigma_list) != num_rows:
            num_rows = len(sigma_list)
            rot_box_scores = rot_box_scores[0:len(sigma_list), :]
        if len(BO_threshold_list) != num_cols:
            num_cols = len(BO_threshold_list)
            rot_box_scores = rot_box_scores[:, 0:len(BO_threshold_list)]
        '''
#        print("num_rows ", num_rows)
#        print("num_cols ", num_cols)
        fig, ax = plt.subplots(figsize=[6, 4])
        # Creating the new color map
        jet = plt.get_cmap('jet')
        new_jet = truncate_colormap(jet, 0.1, 0.9)
        '''
        # Using the new color map
        im = ax_2D.imshow(arr, cmap=new_jet)
        '''
        cmap = matplotlib.cm.get_cmap(new_jet)
        # make it so values under 0.5 are white
        cmap.set_under('white')
        im = plt.imshow(rot_box_scores, cmap = cmap, interpolation = 'nearest', aspect='auto')
        norm = matplotlib.colors.Normalize(vmin=0.5, vmax=1.0)
        plt.clim(color_min, color_max)
        font_size = 18
#        plt.rcParams.update({'font.size': 40})
        plt.tick_params(labelsize=font_size)
        tick_limit = 9 # arbitrary, but here to help make sure the axes don't get to crowded
        if num_cols > tick_limit: # reduce the number of row tick marks when there are lots of sigmas
#            col_locs = np.arange(0, num_cols, int(num_cols/4))
#            col_locs = [0, 19, 39, 59]
            col_locs = [0, 34, 69, 104]
#            print("here is the col_locs list", col_locs)
            # add the correct tick labels
#            plt.xticks(col_locs, ['%.0f'%f for f in sigma_list[col_locs]])
#            plt.xticks(col_locs, [10, 30, 50, 70])
#            plt.xticks(col_locs, [15, 35, 55, 75])
            plt.xticks(col_locs, [35, 70, 105, 140])
        elif num_cols <= tick_limit:
            col_locs = np.arange(0, num_cols)
#            print("here is the col_locs list", col_locs)
            plt.xticks(col_locs, ['%.0f'%f for f in sigma_list[col_locs]])
        if num_rows > tick_limit: # reduce the number of col tick marks when there are lots of sigmas
            row_locs = np.arange(0, num_rows, int(num_rows/4))
#            row_locs = [0, 9, 19]
            row_locs = [0, 5, 10]
#            row_locs = [0, 9]#, 19]
#            print("here is the row_locs list", row_locs)
            # add the correct tick labels
#            plt.yticks(row_locs, np.flipud(['%.1f'%f for f in BO_threshold_list[row_locs]]))
#            plt.yticks(row_locs, np.flipud([1, 2, 3]))
#            plt.yticks(row_locs, np.flipud([1, 2]))
            plt.yticks(row_locs, np.flipud([0.5, 1.0, 1.5]))
        elif num_rows <= tick_limit:
            print("num_rows ", num_rows)
            row_locs = np.arange(0, num_rows)
#            print("here is the row_locs list", row_locs)
            print("TEST THIS", np.flipud(row_locs))
            plt.yticks(row_locs, np.flipud(BO_threshold_list[row_locs]))
#        plt.title('abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ') #testing font type
#        plt.rcParams.update({'font.size': font_size})
        plt.xlabel(r'$\sigma$ ' + r'$(cm^{-1})$', fontsize=font_size)#, labelpad = 30)
        plt.ylabel(r'$\mu$ (multiple of $\sigma$)', fontsize=font_size)
        import matplotlib.colors as colors
        # starting x-coordinate, starting y-coordinate, width, height
#        ax_color = fig.add_axes([0.93, 0.238, 0.03, 0.529])
        ax_color = fig.add_axes([0.93, 0.125, 0.03, 0.756])
        matplotlib.colorbar.ColorbarBase(ax_color, cmap=cmap, norm=norm, orientation='vertical', ticks=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        # using 12 instead of 14 because of rescaling the colorbar 
        ax_color.yaxis.set_tick_params(labelsize=font_size)#font_size)
#        cbar = fig.colorbar(im, ticks=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
#        cbar.ax.set_yticklabels([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
#        fig.tight_layout()
        heat_map_title = 'HEAT_MAP.pdf'
        plt.savefig(heat_map_title, bbox_inches='tight')

    def generate_all_plots(self, sigma, threshold, score_process='bondwise'):
        '''
        Parameters
        ----------
        sigma : float
            Filter roll-off frequency expressed in wavenumbers.
        threshold : float
            Value is multiplied by sigma and is a threshold to identify extrema in the first time derivative of the BO time series. 
        score_process : string, default='bondwise'
            Input 'bondwise' for bond-wise objective function and 'unified' for unified objective function.
        
        Returns
        -------
        
        Files Created
        -------------
        *see individual plotting methods
        '''
        print('GENERATING ALL PLOTS')
        # only create these plots if the score_process is 'bondwise'
        if score_process == 'bondwise':
            self.plot_bondwise_comparison(sigma=sigma, threshold=threshold)
            self.plot_cluster_pairs_vs_bots(sigma=sigma, threshold=threshold)
            self.plot_reference_distances(sigma=sigma, threshold=threshold)
        self.plot_regular_vs_FT_vs_deriv(sigma=sigma, threshold=threshold, score_process=score_process)
        self.plot_union_comparison_and_auc(sigma=sigma, threshold=threshold, score_process=score_process)
    
    def plot_regular_vs_FT_vs_deriv(self, sigma, threshold, score_process='bondwise'):
        '''
        Plots a three column plot for a given sigma and threshold where each row is a contributing (non-zero number of 
        predicted reaction events) atom pair. First column is the raw and smoothed BO time series. Second column is the
        raw Fourier Transform (FT), the smooth FT, and the Butterworth filter. Third column is the first time derivative
        of the smooth time series with the +/- threshold as well as the resulting detected reaction reaction events.
        
        Parameters
        ----------
        sigma : float
            Filter roll-off frequency expressed in wavenumbers.
        threshold : float
            Value is multiplied by sigma and is a threshold to identify extrema in the first time derivative of the BO time series. 
        score_process : string, default='bondwise'
            Input 'bondwise' for bond-wise objective function and 'unified' for unified objective function.

        Returns
        -------
        
        Files Created
        -------------
        'BOTS_vs_FT_vs_deriv_all_relevant_using_sigma%.1f_and_BO_thresh%.1f_score%.3f.pdf'%(sigma, threshold, score)
            Heat map file saved with some identifying characteristics in the name.
        '''
        self.check_parameter_data_cache(sigma, threshold, score_process)
        tm = self.parameter_data_cache[(sigma, threshold)]['transition_matrix']
        score = self.parameter_data_cache[(sigma, threshold)]['score']
        r = self.parameter_data_cache[(sigma, threshold)]['r']
        c = self.parameter_data_cache[(sigma, threshold)]['c']
        relevant_rows = self.parameter_data_cache[(sigma, threshold)]['relevant_rows']
        relevant_smoothed = self.parameter_data_cache[(sigma, threshold)]['relevant_smoothed']
        relevant_derivative = self.parameter_data_cache[(sigma, threshold)]['relevant_derivative']
        peaks_by_row = self.parameter_data_cache[(sigma, threshold)]['peaks_by_row']
        relevant_ft = self.ft[relevant_rows]
        relevant_smooth_ft = self.smooth_ft[relevant_rows]
        plot_scale = 0.75

        num_all_plots = len(tm)
        plt.figure()#plt.figure(2)
        fig3 = plt.gcf()
        fig_len = 4 + num_all_plots
        fig3.set_size_inches(12, fig_len)
        plt.rcParams.update({'font.size': 14})
        relevant_raw_TS = self.all_raw_time_series[relevant_rows]
        ymax = max(relevant_raw_TS.flatten())
        deriv_max = np.amax(relevant_derivative)
        converted_thresh = threshold*sigma/self.conversion
        x_ticks = np.linspace(0, self.traj_length, 5)
        new_labels = []
        for n in range(0, 101, 10):
            new_labels.append(self.freqx[n])
        print('new_labels', new_labels)

        for i in range(num_all_plots):
            # FT
            time_series = relevant_raw_TS[i]
            smoothed_time_series = relevant_smoothed[i]
            single_deriv = relevant_derivative[i]
            fti = relevant_ft[i]
            smooth_fti = relevant_smooth_ft[i]
            '''the following code generates a plot that displays the BO trajectory and corresponding FT for
            each relevant trajectory (that actually transitions accross the BO_threshold)'''
            # *********************
            '''BO sime series column'''
            # create an N by 2 plot with BO trajectories on the left (N, 2, odd number)
            ax1 = plt.subplot(num_all_plots,3,3*i+1)
            # adjust the y-axis tick marks for the trajectories
            plt.ylim([0, ymax])
            plt.xlim([0, self.traj_length])
#            plt.xticks(np.linspace(0, self.traj_length, 5))
            plt.xticks(x_ticks)
            plt.yticks([])
            # minus one on each index to be compatible with the elements list
            plt.ylabel('%s%i %s%i             '%(self.elements[r[i]-1], r[i], self.elements[c[i]-1], c[i]), rotation=0)
            plt.xlabel(self.x_label)
            '''FT of trajectory'''
            # create the BO trajectory plot
            # create title over BO trajectory plots on left side on the first iteration
            if i == 0:
                plt.title('Orange = Raw BO Time Series\nPurple = Smoothed (Sigma=%.1f)'%(sigma))
            # scale the modified transition data so that the transitions are the same size of the best BO_threshold
            plt.plot(range(len(time_series)),time_series , 'xkcd:orange', linewidth=0.2)
            plt.plot(range(len(time_series)), smoothed_time_series, 'xkcd:purple')
            # clone axis for right-side labeling
            ax1b = ax1.twinx()
            plt.xlim([0, self.traj_length])
            plt.xticks(x_ticks)
            plt.ylim([0, ymax])
            plt.yticks([0.25])
            '''FT of BO time series column'''
            ax2 = plt.subplot(num_all_plots,3,3*i+2)
            if i == 0:
                plt.title('abs value of FT')
            plt.yticks([])
#            plt.xticks(locs, new_labels, rotation=45)
            plt.xticks(new_labels, rotation=45)

#            plt.xticks(np.linspace(0, self.traj_length, 5))
#            plt.ylim([0, 1.25])
#            plt.plot(range(len(self.modified_filter)), self.modified_filter, 'k')
            plt.plot(self.freqx, self.modified_filter, 'k')
            ax2b = ax2.twinx()
#            plt.ylim([0,1.25])
#            plt.plot(range(len(self.modified_filter)), self.modified_filter, 'k')
#            plt.xticks([])
#            plt.xticks(range(len(self.freqx)), self.freqx)
            plt.xlim([0, 2*sigma])
#            plt.xlim([0, 500])
            plt.xlabel('cm^-1')
            plt.yticks([200])
#            plt.yticks([500])
            plt.ylim([0, 2000])
#            plt.xlim([0, self.traj_length])
#            plt.plot(range(len(fti)), abs(fti), 'xkcd:orange')
#            plt.plot(range(len(smooth_fti)), abs(smooth_fti), 'xkcd:purple')
            plt.plot(self.freqx, abs(fti), 'xkcd:orange')
            plt.plot(self.freqx, abs(smooth_fti), 'xkcd:purple')
#            locs, labels = plt.xticks()
#            print('locs', locs, 'labels', labels)
#            plt.xticks([])
#            new_labels = [self.freqx[int(l)] for l in locs]

#            plt.plot(range(len(self.modified_filter)), self.modified_filter, 'k')
            
            '''derivative of BO time series column'''
            # create an N by 2 plot with BO trajectories on the left (N, 2, even number)
            ax3 = plt.subplot(num_all_plots,3,3*i+3)
            plt.xlim([0, self.traj_length])
#            plt.xticks(np.linspace(0, self.traj_length, 5))
            plt.xticks(x_ticks)
            plt.xlabel(self.x_label)
            plt.yticks([])
            # clone axis for right-side labeling
            ax3b = ax3.twinx()
            plt.xlim([0, self.traj_length])
            # create title over FT plots on right side
            if i == 0:
                plt.title('First Derivative of BOTS\nand Reaction Event Locations')
            plt.ylim(-deriv_max*1.1, deriv_max*1.1)
            if threshold == 1.0:
                thresh_tag = ''
            else:
                thresh_tag = '%.1f'%round(threshold, 1)
            plt.yticks([-converted_thresh, converted_thresh], [r'$-%s\sigma$'%thresh_tag, r'$+%s\sigma$'%thresh_tag])
            for x in peaks_by_row[i]:
                plt.plot([x], [single_deriv[x]], marker='o', markersize=7, color='blue')
            plt.plot(range(len(time_series)), single_deriv, 'g')
            plt.plot(range(len(time_series)), [converted_thresh for i in range(len(time_series))], ':k', linewidth=0.2)
            plt.plot(range(len(time_series)), [-converted_thresh for i in range(len(time_series))], ':k', linewidth=0.2)
        # remove all x-axis labels except for the last three (bottom of each column)
        plt.setp([a.get_xticklabels() for a in fig3.axes[:-6]], visible=False)
        # make all of the subplots close to each other
        plt.subplots_adjust(hspace=0.000)
        bots_vs_deriv_title = 'BOTS_vs_FT_vs_deriv_all_relevant_%s_sigma%.1f_and_BO_thresh%.1f_score%.3f.pdf'%(score_process, sigma, threshold, score)
        plt.savefig(bots_vs_deriv_title)
        fig3.clear()
        plt.gcf().clear()
        plt.clf()
        plt.cla()
        plt.close()
        
    ''' stopped review here
    '''
    
    ''' NEEDS SOME CLEAN-UP STILL
    '''
    def plot_cluster_pairs_vs_bots(self, sigma, threshold):
        ''' create a plot of cluster index vs reference reaction event index
        aka "arrow plot"
        
        Parameters
        ----------
        sigma : float
            Filter roll-off frequency expressed in wavenumbers.
        threshold : float
            Value is multiplied by sigma and is a threshold to identify extrema in the first time derivative of the BO time series. 
        score_process : string, default='bondwise'
            Input 'bondwise' for bond-wise objective function and 'unified' for unified objective function.

        Returns
        -------
        
        Files Created
        -------------
        'bots_vs_cluster_pairs_sig%.1f_thresh%.1f_score%.3f_V3.pdf'%(sigma, threshold, score)
            'arrow plot' with arrows showing reference reaction event cluster transitions and colors showing temporal proximity to 
            predicted reaction events.
        '''
        font_size = 18
        score_process='bondwise'
        plt.rcParams.update({'font.size': font_size})
        self.check_parameter_data_cache(sigma, threshold, score_process)
        score = self.parameter_data_cache[(sigma, threshold)]['score']
        found_by = self.parameter_data_cache[(sigma, threshold)]['found_by']
        bots_ref_cluster_pair = []
        max_separation = 0
        # iterate over key (ref rxn event location) and value
        for key, value in self.all_ref_transitions.items():
            leftmost_cluster = self.cluster_time_series[key]
            rightmost_cluster = self.cluster_time_series[key+1]
            cluster_pair = [leftmost_cluster, rightmost_cluster]
            if key not in found_by:
                raise ValueError("not all reference reactions are found using current parameter combination")
            # create a list of bots-ref distances
            adj_bots = [abs(key-bots) for bots in found_by[key]]
            # cluster pair transition found
            if min(adj_bots) > max_separation:
                max_separation = min(adj_bots)
            # create a list of [bots, key, [cluster pair]], where bots is the closest lying bots to the ref
            bots_ref_cluster_pair.append([found_by[key][np.argmin(max_separation)], key, cluster_pair])
        f, ax = plt.subplots(1, 1, sharey=True, figsize=(11, 6))
        plt.subplots_adjust(left=0.07)
#        arrow_head_length = 5
        arrow_head_length = 2
        # arbitrary scaling value to make the y axis to be more like the x-axis. Otherwise the x-axis is
        # so dense that you can't see any arrowheads
#        scale = 530
        scale = 10
        
        # Creating the new color map
        jet = plt.get_cmap('jet')
        cmap = truncate_colormap(jet, minval=0.1, maxval=0.9, n_separation=max_separation+1)
        # norm is used along with cmap to create the colorbar later
        norm = matplotlib.colors.Normalize(vmin=0, vmax=max_separation)
        x_axis_sum = 0
        bots_previous = 0
        spacer = 0
        base = 10
        repeat_bots = 0
        offset = 1
        bots_dict = {}
        # sort by ref 
        sorted_bots_ref_cluster_pair = sorted(bots_ref_cluster_pair, key=itemgetter(1))
        num_relevant = 0
        
        # iterate through a list containing [bots, ref, [cluster pair]] to plot the cluster transition arrows
        for i in range(len(sorted_bots_ref_cluster_pair)):
            bots = sorted_bots_ref_cluster_pair[i][0]
            ref = sorted_bots_ref_cluster_pair[i][1]
            pair = sorted_bots_ref_cluster_pair[i][2]
            separation = abs(bots-ref)
            print('separation', separation)
            # for a cluster transition
            # make sure that cluster ordering is consistent
            if pair[0] > pair[1]:
                y = pair[0]
                y_component = -abs(pair[0]-pair[1])
            elif pair[0] < pair[1]:
                y = pair[0]
                y_component = abs(pair[0]-pair[1])
            # assign arrow color using distance of bots event from corresponding ref event
            rgba = cmap(separation)
            print('rgba', rgba)
#            arrow_width = 14
            arrow_width = 0.5
#            arrowhead_width = arrow_width*3
            arrowhead_width = arrow_width*2
            # no spacer for first bots event, or ref events found by the same bots event
            if i == 0 or bots_previous == bots:
                spacer = 0
                if i == 0:
                    num_relevant += 1
            else:
                num_relevant += 1
                spacer = int(math.log(abs(bots-bots_previous), base)) #+1
            x_axis_sum += spacer
#            x = num_relevant_bots+offset+x_axis_sum
            # use this line if arrows ARE overlaid
#            x = num_relevant+x_axis_sum
            # use this line if arrows ARE NOT overlaid
            x = i+x_axis_sum+offset
#            print('i', i, 'x', x, 'bots_previous', bots_previous, 'bots', bots)
            ax.arrow(x, y*scale, 0, y_component*scale, width=arrow_width, head_width=arrowhead_width, head_length=arrowhead_width*5 ,length_includes_head=True, color=rgba)
            dot_color_style = 'k:'
            line_width = 2
            if i == 0:
                plt.plot([0, x], [y*scale, y*scale], dot_color_style, linewidth=line_width, dash_capstyle='round')
            if i != 0:
                plt.plot([x_previous, x], [y*scale, y*scale], dot_color_style, linewidth=line_width, dash_capstyle='round')
            if i == len(sorted_bots_ref_cluster_pair)-1:
                plt.plot([x, self.num_all_ref+x_axis_sum+2], [(y+y_component)*scale, (y+y_component)*scale], dot_color_style, linewidth=line_width, dash_capstyle='round')
            bots_dict[ref] = x
            x_previous = x
            bots_previous = bots
        # adds some space after the last bots rxn event
        adjust = 2
        print('repeat_bots', repeat_bots)
        print('x_axis_sum', x_axis_sum)
        # the spacing in between bots rxn events are irregular so a sum is used to scale the x-axis correctly
        print('x-axis length', self.num_all_ref+x_axis_sum+adjust)
        ax.set_xlim(0, self.num_all_ref+x_axis_sum+adjust)
        ax.xaxis.set_tick_params(rotation=35)
        # scale y axis
        ax.set_ylim(0, (self.num_clusters+1)*scale)
        # manually set the cluster number ticks because of the scaling used
        y_ticks = [x for x in range(1, self.num_clusters +1)]
        y_locs = [x*scale for x in y_ticks]
        ax.set_yticks(y_locs)
        ax.set_yticklabels(y_ticks)
        ax.set_ylabel('cluster index')
        x_ticks = []
        x_locs = []
        for bots, x in bots_dict.items():
            x_ticks.append(bots)
            x_locs.append(x)
        if len(x_locs) > 20:
            iteration = 5
        else:
            iteration = 2
        # display x ticks spaced by iteration
        selected_locs = x_locs[0::iteration]
        selected_ticks = x_ticks[0::iteration]
        print('x_locs', selected_locs)
        print('x_ticks', selected_ticks)
        ax.set_xticks(selected_locs)
        ax.set_xticklabels(selected_ticks)
        ax.set_xlabel('Time step of reference reaction event')
        ax.yaxis.grid(linewidth=0.5)
        plt_title = 'bots_vs_cluster_pairs_%s_sig%.1f_thresh%.1f_score%.3f_V3.pdf'%(score_process, sigma, threshold, score)
        # starting x-coordinate, starting y-coordinate, width, height
        axb = f.add_axes([0.9, 0.125, 0.02, 0.755])
        matplotlib.colorbar.ColorbarBase(axb, cmap=cmap, norm=norm, orientation='vertical')
        axb.set_ylabel('Time offset from nearest detected reaction event')
        plt.savefig(plt_title, bbox_inches='tight')

    def plot_reference_distances(self, sigma, threshold):
        ''' create a plot of cluster index vs reference reaction event index
        aka "arrow plot"
        
        Parameters
        ----------
        sigma : float
            Filter roll-off frequency expressed in wavenumbers.
        threshold : float
            Value is multiplied by sigma and is a threshold to identify extrema in the first time derivative of the BO time series. 
        score_process : string, default='bondwise'
            Input 'bondwise' for bond-wise objective function and 'unified' for unified objective function.

        Returns
        -------
        
        Files Created
        -------------
        'reference_rxn_events_distance_%s.pdf'%score_process
        '''
        font_size = 18
        score_process='bondwise'
        plt.rcParams.update({'font.size': font_size})
        self.check_parameter_data_cache(sigma, threshold, score_process)
        score = self.parameter_data_cache[(sigma, threshold)]['score']
        found_by = self.parameter_data_cache[(sigma, threshold)]['found_by']
        bots_ref_cluster_pair = []
        max_separation = 0
        
        # iterate over key (ref rxn event location) and value
        for key, value in self.all_ref_transitions.items():
            leftmost_cluster = self.cluster_time_series[key]
            rightmost_cluster = self.cluster_time_series[key+1]
            cluster_pair = [leftmost_cluster, rightmost_cluster]
            if key not in found_by:
                raise ValueError("not all reference reactions are found using current parameter combination")
            # create a list of bots-ref distances
            adj_bots = [abs(key-bots) for bots in found_by[key]]
            # cluster pair transition found
            if min(adj_bots) > max_separation:
                max_separation = min(adj_bots)
            # create a list of [bots, key, [cluster pair]], where bots is the closest lying bots to the ref
            bots_ref_cluster_pair.append([found_by[key][np.argmin(max_separation)], key, cluster_pair])
        f, ax = plt.subplots(1, 1, sharey=True, figsize=(11, 6))
        plt.subplots_adjust(left=0.07)
        # sort by ref 
        sorted_bots_ref_cluster_pair = sorted(bots_ref_cluster_pair, key=itemgetter(1))
        all_separation = []
        all_data = []
        # iterate through a list containing [bots, ref, [cluster pair]] to plot the cluster transition arrows
        for i in range(len(sorted_bots_ref_cluster_pair)):
            bots = sorted_bots_ref_cluster_pair[i][0]
            ref = sorted_bots_ref_cluster_pair[i][1]
            pair = sorted_bots_ref_cluster_pair[i][2]
            separation = abs(bots-ref)
            all_separation.append(separation)
            all_data.append([separation, ref, pair])
        all_sorted = np.sort(all_separation)
        plt.plot(all_sorted)
        plt.xlim(0, len(all_sorted))
        plt.ylim(0, max(all_sorted))
        plt.ylabel('Time offset from nearest predicted rxn event')
        plt.xlabel('reference rxn events ranked by distance')
        plt.tight_layout()
        fig_title = 'reference_rxn_events_distance_%s.pdf'%score_process
        plt.savefig(fig_title)
        all_data_sorted = sorted(all_data, key=itemgetter(0))
                
        with open('cluster_transitions_%s.txt'%score_process, 'w') as out_file:
            out_file.write('sigma = %.2f '%sigma + ' threshold = %.2f'%threshold)
            out_file.write('\ndistances are from reference reaction event to nearest BOTS reaction event')
            out_file.write('\n\nCLUSTER PAIR\tAVERAGE\tMEDIAN\tMAX\tNUM OCCURANCES')

            cluster_pair_dist_matrix = np.zeros((self.num_clusters, self.num_clusters))
            for row in range(1, self.num_clusters+1):
                for col in range(row+1, self.num_clusters+1):
                    pair_total = []
                    ave_dist = 0
                    median_dist = 0
                    max_dist = 0
                    for i in range(len(all_data)):
                        if [row, col] == all_data[i][2]:
                            pair_total.append(all_data[i][0])
                        elif [col, row] == all_data[i][2]:
                            pair_total.append(all_data[i][0])
                    # if the pair_total list is non-empty
                    if len(pair_total) != 0:
                        # row and col indices start from 1, so that is subtracted so the entries are in
                        # the correct locations
                        # average
                        ave_dist = sum(pair_total)/len(pair_total)
                        max_dist = max(pair_total)
                        median_dist = statistics.median(pair_total)
                        # if statement to align the table
                        if row >= 10 and col >= 10:
                            out_file.write('\n(%i, %i)\t%.1f\t%i\t%i\t%i'%(row, col, ave_dist, median_dist, max_dist, len(pair_total)))
                        else:
                            out_file.write('\n(%i, %i)\t\t%.1f\t%i\t%i\t%i'%(row, col, ave_dist, median_dist, max_dist, len(pair_total)))
                    # else if the pair_total list is empty
                    elif len(pair_total) == 0:
                        cluster_pair_dist_matrix[row-1, col-1] = 0

        cluster_data = []
        for c in range(1, self.num_clusters+1):
            c_total_sep = []
            for row in range(len(all_data_sorted)):
                if c in all_data_sorted[row][2]:
                    c_total_sep.append(all_data_sorted[row][0])
            average = sum(c_total_sep)/len(c_total_sep)
            # cluster, average, median, max, standard deviation, number of events
            cluster_data.append([c, average, statistics.median(c_total_sep), max(c_total_sep), np.std(c_total_sep), len(c_total_sep)])
        # output file that examines how difficult each cluster is to find (using all reaction events that cluster is involved in)
        sorted_cluster_data = sorted(cluster_data, key=itemgetter(1))
        with open('cluster_difficulty_%s.txt'%score_process, 'w') as out_file:
            out_file.write('sigma = %.2f '%sigma + ' threshold = %.2f'%threshold)
            out_file.write('\n%i total reference reaction events'%self.num_all_ref)
            out_file.write('\ndistances are from reference reaction event to nearest BOTS reaction event')
            out_file.write('\n\n\tcluster number\tAVERAGE dist\tMEDIAN dist\tMAX dist\tSTANDARD DEVIATION\tNUMBER OF EVENTS\n')
            for l in range(self.num_clusters):
                new_string = '\n\t%i\t\t%.2f\t\t%.2f\t\t%.2f\t\t%.2f\t\t\t%i'%(sorted_cluster_data[l][0], sorted_cluster_data[l][1], sorted_cluster_data[l][2], sorted_cluster_data[l][3], sorted_cluster_data[l][4], sorted_cluster_data[l][5])
                out_file.write(new_string)

    def plot_union_comparison_and_auc(self, sigma, threshold, score_process='bondwise'):
        ''' create a two-panel plot with the left plot showing reaction event lines and
        descending windows and the right plot showing as many TPR/FPR plots generated for
        the objective function.
        
        Parameters
        ----------
        sigma : float
            Filter roll-off frequency expressed in wavenumbers.
        threshold : float
            Value is multiplied by sigma and is a threshold to identify extrema in the first time derivative of the BO time series. 
        score_process : string, default='bondwise'
            Input 'bondwise' for bond-wise objective function and 'unified' for unified objective function.

        Returns
        -------
        
        Files Created
        -------------
        'union_comparison_and_auc_plot_%s_sig%.1f_thresh%.1f_score%.3f.png'%(score_process, sigma, threshold, score)
        '''
        self.check_parameter_data_cache(sigma, threshold, score_process)
        tpr_vs_fpr = self.parameter_data_cache[(sigma, threshold)]['tpr_vs_fpr']
        score = self.parameter_data_cache[(sigma, threshold)]['score']
        BO_transition_matrix = self.parameter_data_cache[(sigma, threshold)]['transition_matrix']
        window_size_100_percent = self.parameter_data_cache[(sigma, threshold)]['window_size']
        font_size = 14

        bots_scale = 0.85
        window_scale = 0.7
        fig, (ax3, ax4) = plt.subplots(1, 2, figsize=[9,4], gridspec_kw={'width_ratios':[2,1]})

        ax3.set_ylim(0,1.025)
        ax3.set_yticks([])

        upscale_for_plotting = 5
        ref_color = 'r'
        bots_color = '#1E90FF'
        win_color = '#FFA500'
        area_color = '#FFD27F'
        grey_win_color = '#C1CDCD'
        grey_area_color = '#E6EBEB'
        # scaled transition matrix so predicted reaction events are shorter than reference reaction events
        tm = np.any(BO_transition_matrix, axis=0).astype(int)*bots_scale
        BO_transition_list = np.nonzero(np.any(BO_transition_matrix, axis=0))[0]
        upscale_for_plotting = 1
        
        for window_size in range(window_size_100_percent):
            box_data_best = np.zeros(self.traj_length)
            height = window_scale*(window_size_100_percent-window_size*upscale_for_plotting)/window_size_100_percent
            for i in BO_transition_list:
                # create windows with height inversely proportinal to their size
                box_data_best[ max(i-window_size, 0) : min(i+window_size, len(box_data_best)-1) ] = True
            ax3.fill_between(range(len(box_data_best)), 0, box_data_best*height, facecolor=win_color, alpha = 1)
        ax3.plot(self.binary_cluster_time_series, ref_color)
        ax3.plot(tm, bots_color)        
        ax3.set_xlabel('AIMD step', fontsize=font_size)
        ax3.set_xlim(0, self.traj_length)
        ax3.set_xticks(np.linspace(0, self.traj_length, 5))
        ax3.tick_params(axis='x', labelsize=font_size)
        ax3.tick_params(axis='y', labelsize=font_size)
        ax3.legend(['Reference', 'BOTS', 'Window = +/- %i'%window_size], prop={'size':font_size})
        
        tpr, fpr = zip(*tpr_vs_fpr)
        # break the tpr/fpr sequence into two portions
        shorter_tpr = tpr[0:window_size_100_percent]
        shorter_fpr = fpr[0:window_size_100_percent]
        latter_tpr = tpr[window_size_100_percent-1:]
        latter_fpr = fpr[window_size_100_percent-1:]        
                        
        ax4b = ax4.twinx()
        # greyed out portion
        ax4.plot(latter_fpr, latter_tpr, color = grey_win_color, marker = 'o')
        ax4.fill_between(latter_fpr, latter_tpr, facecolor=grey_area_color, interpolate=True)
        # shortened portion
        ax4.plot(shorter_fpr, shorter_tpr, color = win_color, marker = 'o')
        ax4.fill_between(shorter_fpr, shorter_tpr, facecolor=area_color, interpolate=True)

        ax4.set_xlabel('False Positive Rate', size = font_size)
        ax4b.set_ylabel('True Positive Rate', size = font_size)
        ax4.set_ylim(0, 1)
        ax4.set_xlim(0, 1)
        ax4.set_xticks([0.0, 0.5, 1.0])
        ax4.set_yticks([])
        ax4b.set_yticks([0.5, 1.0])
        ax4.tick_params(axis='x', labelsize=font_size)
        ax4b.tick_params(axis='y', labelsize=font_size)
        plot_title = 'union_comparison_and_auc_plot_%s_sig%.1f_thresh%.1f_score%.3f.png'%(score_process, sigma, threshold, score)
        fig.tight_layout()
        plt.savefig(plot_title)

    def plot_bondwise_comparison(self, sigma, threshold):
        ''' create a plot for the 'bondwise' scoring method, but still using derivatives
        
        Parameters
        ----------
        sigma : float
            Filter roll-off frequency expressed in wavenumbers.
        threshold : float
            Value is multiplied by sigma and is a threshold to identify extrema in the first time derivative of the BO time series. 
        score_process : string, default='bondwise'
            Input 'bondwise' for bond-wise objective function and 'unified' for unified objective function.

        Returns
        -------
        
        '''
        score_process='bondwise'
        self.check_parameter_data_cache(sigma, threshold, score_process=score_process)
        BO_transition_matrix = self.parameter_data_cache[(sigma, threshold)]['transition_matrix']
        window_size = self.parameter_data_cache[(sigma, threshold)]['window_size']
        score = self.parameter_data_cache[(sigma, threshold)]['score']
        r = self.parameter_data_cache[(sigma, threshold)]['r']
        c = self.parameter_data_cache[(sigma, threshold)]['c']
        found_by = self.parameter_data_cache[(sigma, threshold)]['found_by']

        bots_scale = 0.85
        window_scale = 0.7

        upscale_for_plotting = 5
        win_color = '#FFA500'

        num_all_plots = len(BO_transition_matrix)
        plt.figure()
        fig4 = plt.gcf()
        fig_len = 4 + num_all_plots
        fig4.set_size_inches(8, fig_len)
        
        bots_ref_cluster_pair = []
        max_separation = 0
        # iterate over key (ref rxn event location) and value
        for key, value in self.all_ref_transitions.items():
            leftmost_cluster = self.cluster_time_series[key]
            rightmost_cluster = self.cluster_time_series[key+1]
            cluster_pair = [leftmost_cluster, rightmost_cluster]
            if key not in found_by:
                raise ValueError("not all reference reactions are found using current parameter combination")
            # create a list of bots-ref distances
            adj_bots = [abs(key-bots) for bots in found_by[key]]
            # cluster pair transition found
            if min(adj_bots) > max_separation:
                max_separation = min(adj_bots)
            # create a list of [bots, key, [cluster pair]], where bots is the closest lying bots to the ref
            bots_ref_cluster_pair.append([found_by[key][np.argmin(max_separation)], key, cluster_pair])
        sorted_bots_ref_cluster_pair = sorted(bots_ref_cluster_pair, key=itemgetter(1))
        
        # truncate jet colormap
        jet = plt.get_cmap('jet')
        new_jet = truncate_colormap(jet, 0.1, 0.9)

        colors = new_jet(np.linspace(0,1,max_separation+1))

        for i in range(num_all_plots):
            '''BO trajectory'''
            # create an N by 2 plot with BO trajectories on the left (N, 2, odd number)
            ax1 = plt.subplot(num_all_plots, 1, i+1)
            # create title over FT plots on right side
            if i == 0:
                plt.title('Bond-wise BOTS-predicted vs Reference\nReaction Event Locations')
            ''' trying to make a plot that shows each bond pair with its individualized comparison plot
            '''
            BO_transition_list = np.nonzero(BO_transition_matrix[i])[0]
            upscale_for_plotting = 1
            
            plt.ylim(0,1.2)
            tm_bots = BO_transition_matrix[i]*bots_scale#plot_scale

            for w in range(window_size):
                box_data_best = np.zeros(self.traj_length)
                height = window_scale*(window_size-w*upscale_for_plotting)/window_size
                # create windows with height that is inversely proportional to their size
                for b in BO_transition_list:
                    box_data_best[ max(b-w, 0) : min(b+w, len(box_data_best)-1) ] = True
                ax1.fill_between(range(len(box_data_best)), 0, box_data_best*height, facecolor=win_color, alpha = 1)
            
            # create the reference transition time series for the current atom pair
            ref_transitions = []
            # iterate over each transition object (reference reaction event)
            g = 0
            for t_loc, t_object in self.all_ref_transitions.items():
                # if a atom pair containing BOTS predicitons matches those pairs corresponding to the reference reaction event, 
                # add it to ref_loc
                if (r[i], c[i]) in t_object.atompairs:
                    ref_loc = np.zeros(self.traj_length)
                    ref_loc[t_loc] = 1
                    for row in range(len(sorted_bots_ref_cluster_pair)):
                        if t_loc == sorted_bots_ref_cluster_pair[row][1]:
                            separation = abs(sorted_bots_ref_cluster_pair[row][0] - sorted_bots_ref_cluster_pair[row][1])
                            plt.plot(ref_loc, c=colors[separation])
                g += 1
            # from the reference transitions, make the reference transition series
            ref_series = np.zeros(self.traj_length)
            ref_series[ref_transitions] = 1
            plt.plot(tm_bots, 'k')

            plt.yticks([])
            plt.ylabel('%s%i %s%i             '%(self.elements[r[i]-1], r[i], self.elements[c[i]-1], c[i]), rotation=0)
            plt.xlabel(self.x_label)
            plt.xlim(0, self.traj_length)
            x_ticks = np.linspace(0, self.traj_length, 5)
            plt.xticks(x_ticks)
        # remove all x-axis labels except for the last two (bottom of each column)
        plt.setp([a.get_xticklabels() for a in fig4.axes[:-1]], visible=False)
        # make all of the subplots close to each other
        plt.subplots_adjust(hspace=0.000)
        plot_title = 'bondwise_comparison_%s_sig%.1f_thresh%.1f_score%.3f.png'%(score_process, sigma, threshold, score)
        plt.savefig(plot_title)
        fig4.clear()
        plt.gcf().clear()
        plt.clf()
        plt.cla()
        plt.close()
                
