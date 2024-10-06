import mne
import numpy as np
from scipy.signal import hilbert
from scipy.stats import pearsonr
import networkx as nx
import matplotlib.pyplot as plt


# Define the function to compute envelope correlation
def compute_envelope_correlation(data, sfreq, fmin, fmax):
    """
    Compute envelope correlation for EEG data.
    
    Parameters:
    - data: EEG data, shape (n_channels, n_times)
    - sfreq: Sampling frequency
    - fmin: Lower bound of the frequency range for band-pass filtering
    - fmax: Upper bound of the frequency range for band-pass filtering
    
    Returns:
    - corr_matrix: Envelope correlation matrix (n_channels x n_channels)
    """
    # Step 1: Band-pass filter the data
    filtered_data = mne.filter.filter_data(data, sfreq, fmin, fmax, verbose=False)
    
    # Step 2: Compute the analytic signal using the Hilbert transform
    analytic_signal = hilbert(filtered_data, axis=1)
    
    # Step 3: Compute the envelope (magnitude of the analytic signal)
    envelopes = np.abs(analytic_signal)
    
    # Step 4: Compute the correlation matrix between the envelopes
    n_channels = envelopes.shape[0]
    corr_matrix = np.zeros((n_channels, n_channels))
    
    for i in range(n_channels):
        for j in range(i, n_channels):
            if i == j:
                corr_matrix[i, j] = 1.0  # Correlation with itself
            else:
                # Pearson correlation between the envelopes of channels i and j
                corr, _ = pearsonr(envelopes[i], envelopes[j])
                corr_matrix[i, j] = corr
                corr_matrix[j, i] = corr  # Symmetric matrix

    
    return corr_matrix

# Define the function to create a graph from the correlation matrix
def create_connectome_graph(corr_matrix, ch_names, threshold=0.5):
    """
    Create a graph based on the correlation matrix.
    
    Parameters:
    - corr_matrix: Correlation matrix (n_channels x n_channels)
    - ch_names: List of channel names
    - threshold: Correlation threshold to include edges in the graph
    
    Returns:
    - G: NetworkX graph representing the connectome
    """
    G = nx.Graph()
    
    # Add nodes (channels)
    G.add_nodes_from(ch_names)
    
    # Add edges based on the correlation matrix
    n_channels = corr_matrix.shape[0]
    for i in range(n_channels):
        for j in range(i+1, n_channels):  # No need to check diagonal or double-check edges
            if corr_matrix[i, j] > threshold:  # Only include edges above the threshold
                G.add_edge(ch_names[i], ch_names[j], weight=corr_matrix[i, j])
    
    return G


def read_adj_matrix_envelope(subject, runs):
    # Step 1: Load the EEGBCI Motor Imagery dataset
    raw_fnames = mne.datasets.eegbci.load_data(subject, runs)
    raw = mne.io.concatenate_raws([mne.io.read_raw_edf(f, preload=True) for f in raw_fnames])


    # Step 2: Filter the data to the frequency band of interest (e.g., alpha band 8-12 Hz)
    # raw.filter(12., 30., fir_design='firwin')

    # Pick EEG channels only
    picks = mne.pick_types(raw.info, eeg=True, exclude='bads')

    # Extract the EEG data (shape: n_channels x n_times)
    data = raw.get_data(picks=picks)
    sfreq = raw.info['sfreq']  # Sampling frequency

    # Step 3: Compute envelope correlation
    fmin, fmax = 8, 12  # Alpha band (8-12 Hz)
    envelope_corr_matrix = compute_envelope_correlation(data, sfreq, fmin, fmax)

    # Step 4: Create a connectome graph based on envelope correlation
    ch_names = raw.ch_names  # EEG channel names
    threshold = 0.2  # Set a threshold to include edges in the graph
    G = create_connectome_graph(envelope_corr_matrix, ch_names, threshold)

    return([nx.to_numpy_array(G), G])