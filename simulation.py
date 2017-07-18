from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import regionprops, label
from copy import deepcopy

# simulate with randomize weights and connections
def simulate(grid_cells, weighted_connections):
    # determine weighted synaptic input to each place cell
    place_cells = np.dot(weighted_connections.T, grid_cells) # faster now!

    # determine which place cells fire by E%-winner-take-all
    place_cell_winners = np.zeros((2000, 10000))
    for i in range(10000):
        maximum = np.max(place_cells[:,i])
        idx = place_cells[:,i] > maximum*0.95
        place_cell_winners[idx,i] = place_cells[idx,i]
    
    # get a list of active place cells
    active = list()
    for i in range(2000):
        activity = np.sum(place_cell_winners[i,:] > 0)
        if activity > 0:
            active.append(i)

    # get a list of place cells with only one place field
    valid_place_cells = []
    for cell in active:
        activity = place_cell_winners[cell,:]
        im_matrix = np.zeros((100, 100))
        i_rate = 0
        for i in range(100):
            for j in range(100):
                im_matrix[i, j] = activity[i_rate]
                i_rate += 1
        thresholded = im_matrix >= np.max(im_matrix)*0.8
        labelled = label(thresholded, neighbors=8)
        props = regionprops(labelled, intensity_image=im_matrix)
        areas = np.array([region.filled_area for region in props]) > 200
        if np.sum(areas) == 1:
            valid_place_cells.append(cell)
    
    # combine and return simulation results
    result = {"place_cell_ids" : np.array(valid_place_cells),
              "place_cells_rates" : place_cell_winners}
    return result

def connections_from_indecies(connection_indecies):
    x = np.zeros((2000, 10000))
    x = x.ravel()
    for i in range(2000):
        idx = connection_indecies[:, i] + i*10000
        for idx_i in idx:
            x[idx_i] += 1.0
    connections = x.reshape((2000, 10000))
    return connections

def scramble_connections(connection_indecies, n_scrambled):
    altered_connections = deepcopy(connection_indecies)
    for i in range(2000):
        to_change = np.random.choice(np.arange(1200), replace=False, size=n_scrambled)
        new_connections = np.random.randint(0, 10000, size=n_scrambled)
        altered_connections[to_change, i] = new_connections
    return altered_connections

def simulate_with_n_scrambled(
    original_connections, n_scrambled, weight_grid, grid_cells):
    '''
    original_connections =
    n_scrambled =
    weight_grid =
    '''
    results = []
    for n in n_scrambled:
        conn_indecies = scramble_connections(original_connections, n)
        connections = connections_from_indecies(conn_indecies)
        result = simulate(grid_cells, connections.T*weight_grid)
        results.append(result)
    return results

def place_field_correlation(result_original_synapses, result_modified_synapses):
    ids_1 = result_original_synapses["place_cell_ids"]
    ids_2 = result_modified_synapses["place_cell_ids"]
    place_cells_in_both = list(set(ids_1) & set(ids_2))
    correlations = np.zeros(len(place_cells_in_both))
    for i, ID in enumerate(place_cells_in_both):
        a = result_original_synapses["place_cells_rates"][ID, :]
        b = result_modified_synapses["place_cells_rates"][ID, :]
        a[a>0] = 1.0
        b[b>0] = 1.0
        correlations[i] = np.corrcoef(a, b)[1,0]
    return correlations

def place_cell_preservation(result_original_synapses, result_modified_synapses):   
    ids_1 = result_original_synapses["place_cell_ids"]
    ids_2 = result_modified_synapses["place_cell_ids"]
    place_cells_in_both = list(set(ids_1) & set(ids_2))
    preserved_cells = len(place_cells_in_both) / len(ids_1)
    return preserved_cells

def position_sorted_indecies(results, sort_by=0):
    linear = results[sort_by]['place_cells_rates'][:, :100]
    place_cell_indecies = []
    centroids = []
    for i in range(linear.shape[0]):
        im = np.array([linear[i,:], np.zeros(100)])
        if np.sum(im) > 0:
            thresholded = im >= np.max(im)*0.8
            labelled = label(thresholded)
            props = regionprops(labelled, intensity_image=im)
            areas = np.array([region.filled_area for region in props]) > 5
            centroid = np.array([region.centroid for region in props])[areas]
            if len(centroid) == 1:
                centroids.append(centroid[0][1])
                place_cell_indecies.append(i)
    centroids = np.array(centroids)
    place_cell_indecies = np.array(place_cell_indecies)
    idx = place_cell_indecies[np.argsort(centroids)]
    return idx

def new_weight_grid():
    def synaptic_size_distribution(s):
        A = 100.7
        B = 0.02
        sigma_1 = 0.022
        sigma_2 = 0.018
        sigma_3 = 0.15
        distribution = A*(1-np.e**(-s/sigma_1))* \
            (np.e**(-s/sigma_2)+B*np.e**(-s/sigma_3))
        return distribution

    synaptic_weight = lambda s: s/0.2 * s/(s + 0.0314)

    random_coordinates = np.array([
        np.random.uniform(0, 0.2, size=10000000),
        np.random.uniform(0, 27, size=10000000)])
    distribution_height = synaptic_size_distribution(random_coordinates[0,:])
    in_dist_index = distribution_height > random_coordinates[1,:]
    weight_population = random_coordinates[0, in_dist_index]

    # set up weight matrix for multiplication with grid cell array
    weight_idx = np.random.randint(0, len(weight_population), size=10000*2000)
    weights = weight_population[weight_idx]
    weight_grid = weights.reshape((10000, 2000))
    
    return weight_grid