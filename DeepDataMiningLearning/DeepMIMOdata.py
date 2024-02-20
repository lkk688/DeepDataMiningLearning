#https://www.deepmimo.net/versions/v2-python/
#pip install DeepMIMO

import DeepMIMO
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from matplotlib import colors

def ebnodb2no(ebno_db, num_bits_per_symbol, coderate):
    r"""Compute the noise variance `No` for a given `Eb/No` in dB.
    Input
    -----
    ebno_db : float
        The `Eb/No` value in dB.

    num_bits_per_symbol : int
        The number of bits per symbol.

    coderate : float
        The coderate used.

    resource_grid : ResourceGrid
        An (optional) instance for OFDM transmissions.

    Output
    ------
    : float
        The value of :math:`N_o` in linear scale.
    """
    #ebno = tf.math.pow(tf.cast(10., dtype), ebno_db/10.)
    ebno = np.power(10, ebno_db/10.0)
    energy_per_symbol = 1
    tmp= (ebno * coderate * float(num_bits_per_symbol)) / float(energy_per_symbol)
    n0 = 1/tmp
    return n0

#https://numpy.org/doc/stable/reference/random/generated/numpy.random.randint.html#numpy.random.randint
def complex_normal(shape, var=1.0):
    r"""Generates a tensor of complex normal random variables.

    Input
    -----
    shape : tf.shape, or list
        The desired shape.

    var : float
        The total variance., i.e., each complex dimension has
        variance ``var/2``.

    dtype: tf.complex
        The desired dtype. Defaults to `tf.complex64`.

    Output
    ------
    : ``shape``, ``dtype``
        Tensor of complex normal random variables.
    """
    # Half the variance for each dimension
    #var_dim = np.complex64(var/2)
    #var_dim = tf.cast(var, dtype.real_dtype)/tf.cast(2, dtype.real_dtype)
    #stddev = np.sqrt(var_dim)

    # Generate complex Gaussian noise with the right variance
    stddev = np.sqrt(var/2)
    xr = np.random.normal(loc=0.0, scale=stddev, size=shape)
    xi = np.random.normal(loc=0.0, scale=stddev, size=shape)
    x = xr + 1j*xi
    # xr = tf.random.normal(shape, stddev=stddev, dtype=dtype.real_dtype)
    # xi = tf.random.normal(shape, stddev=stddev, dtype=dtype.real_dtype)
    # x = tf.complex(xr, xi)

    return x

def pam_gray(b):
    # pylint: disable=line-too-long
    r"""Maps a vector of bits to a PAM constellation points with Gray labeling.

    This recursive function maps a binary vector to Gray-labelled PAM
    constellation points. It can be used to generated QAM constellations.
    The constellation is not normalized.

    Input
    -----
    b : [n], NumPy array
        Tensor with with binary entries.

    Output
    ------
    : signed int
        The PAM constellation point taking values in
        :math:`\{\pm 1,\pm 3,\dots,\pm (2^n-1)\}`.

    Note
    ----
    This algorithm is a recursive implementation of the expressions found in
    Section 5.1 of [3GPPTS38211]_. It is used in the 5G standard.
    """ # pylint: disable=C0301

    if len(b)>1:
        return (1-2*b[0])*(2**len(b[1:]) - pam_gray(b[1:]))
    return 1-2*b[0]

def qam(num_bits_per_symbol, normalize=True):
    r"""Generates a QAM constellation.

    This function generates a complex-valued vector, where each element is
    a constellation point of an M-ary QAM constellation. The bit
    label of the ``n`` th point is given by the length-``num_bits_per_symbol``
    binary represenation of ``n``.

    Input
    -----
    num_bits_per_symbol : int
        The number of bits per constellation point.
        Must be a multiple of two, e.g., 2, 4, 6, 8, etc.

    normalize: bool
        If `True`, the constellation is normalized to have unit power.
        Defaults to `True`.

    Output
    ------
    : :math:`[2^{\text{num_bits_per_symbol}}]`, np.complex64
        The QAM constellation.

    Note
    ----
    The bit label of the nth constellation point is given by the binary
    representation of its position within the array and can be obtained
    through ``np.binary_repr(n, num_bits_per_symbol)``.


    The normalization factor of a QAM constellation is given in
    closed-form as:

    .. math::
        \sqrt{\frac{1}{2^{n-2}}\sum_{i=1}^{2^{n-1}}(2i-1)^2}

    where :math:`n= \text{num_bits_per_symbol}/2` is the number of bits
    per dimension.

    This algorithm is a recursive implementation of the expressions found in
    Section 5.1 of [3GPPTS38211]_. It is used in the 5G standard.
    """ # pylint: disable=C0301

    try:
        assert num_bits_per_symbol % 2 == 0 # is even
        assert num_bits_per_symbol >0 # is larger than zero
    except AssertionError as error:
        raise ValueError("num_bits_per_symbol must be a multiple of 2") \
        from error
    assert isinstance(normalize, bool), "normalize must be boolean"

    # Build constellation by iterating through all points
    c = np.zeros([2**num_bits_per_symbol], dtype=np.complex64)
    for i in range(0, 2**num_bits_per_symbol):
        b = np.array(list(np.binary_repr(i,num_bits_per_symbol)),
                     dtype=np.int16)
        c[i] = pam_gray(b[0::2]) + 1j*pam_gray(b[1::2]) # PAM in each dimension

    if normalize: # Normalize to unit energy
        n = int(num_bits_per_symbol/2)
        qam_var = 1/(2**(n-2))*np.sum(np.linspace(1,2**n-1, 2**(n-1))**2)
        c /= np.sqrt(qam_var)
    return c

def pam(num_bits_per_symbol, normalize=True):
    r"""Generates a PAM constellation.

    This function generates a real-valued vector, where each element is
    a constellation point of an M-ary PAM constellation. The bit
    label of the ``n`` th point is given by the length-``num_bits_per_symbol``
    binary represenation of ``n``.

    Input
    -----
    num_bits_per_symbol : int
        The number of bits per constellation point.
        Must be positive.

    normalize: bool
        If `True`, the constellation is normalized to have unit power.
        Defaults to `True`.

    Output
    ------
    : :math:`[2^{\text{num_bits_per_symbol}}]`, np.float32
        The PAM constellation.

    Note
    ----
    The bit label of the nth constellation point is given by the binary
    representation of its position within the array and can be obtained
    through ``np.binary_repr(n, num_bits_per_symbol)``.


    The normalization factor of a PAM constellation is given in
    closed-form as:

    .. math::
        \sqrt{\frac{1}{2^{n-1}}\sum_{i=1}^{2^{n-1}}(2i-1)^2}

    where :math:`n= \text{num_bits_per_symbol}` is the number of bits
    per symbol.

    This algorithm is a recursive implementation of the expressions found in
    Section 5.1 of [3GPPTS38211]_. It is used in the 5G standard.
    """ # pylint: disable=C0301

    try:
        assert num_bits_per_symbol >0 # is larger than zero
    except AssertionError as error:
        raise ValueError("num_bits_per_symbol must be positive") \
        from error
    assert isinstance(normalize, bool), "normalize must be boolean"

    # Build constellation by iterating through all points
    c = np.zeros([2**num_bits_per_symbol], dtype=np.float32)
    for i in range(0, 2**num_bits_per_symbol):
        b = np.array(list(np.binary_repr(i,num_bits_per_symbol)),
                     dtype=np.int16)
        c[i] = pam_gray(b)

    if normalize: # Normalize to unit energy
        n = int(num_bits_per_symbol)
        pam_var = 1/(2**(n-1))*np.sum(np.linspace(1,2**n-1, 2**(n-1))**2)
        c /= np.sqrt(pam_var)
    return c

#ref from class Constellation
def CreateConstellation(constellation_type, num_bits_per_symbol,normalize=True):
    r"""
    Constellation(constellation_type, num_bits_per_symbol, initial_value=None, normalize=True, center=False, trainable=False, dtype=tf.complex64, **kwargs)

    Constellation that can be used by a (de)mapper.

    This class defines a constellation, i.e., a complex-valued vector of
    constellation points. A constellation can be trainable. The binary
    representation of the index of an element of this vector corresponds
    to the bit label of the constellation point. This implicit bit
    labeling is used by the ``Mapper`` and ``Demapper`` classes.

    Parameters
    ----------
    constellation_type : One of ["qam", "pam", "custom"], str
        For "custom", the constellation points are randomly initialized
        if no ``initial_value`` is provided.

    num_bits_per_symbol : int
        The number of bits per constellation symbol, e.g., 4 for QAM16.

    initial_value : :math:`[2^\text{num_bits_per_symbol}]`, NumPy array or Tensor
        Initial values of the constellation points. If ``normalize`` or
        ``center`` are `True`, the initial constellation might be changed.

    normalize : bool
        If `True`, the constellation is normalized to have unit power.
        Defaults to `True`.

    center : bool
        If `True`, the constellation is ensured to have zero mean.
        Defaults to `False`.

    trainable : bool
        If `True`, the constellation points are trainable variables.
        Defaults to `False`.

    dtype : [tf.complex64, tf.complex128], tf.DType
        The dtype of the constellation.

    Output
    ------
    : :math:`[2^\text{num_bits_per_symbol}]`, ``dtype``
        The constellation.

    Note
    ----
    One can create a trainable PAM/QAM constellation. This is
    equivalent to creating a custom trainable constellation which is
    initialized with PAM/QAM constellation points.
    """
    num_bits_per_symbol = int(num_bits_per_symbol)
    if constellation_type=="qam":
        assert num_bits_per_symbol%2 == 0 and num_bits_per_symbol>0,\
            "num_bits_per_symbol must be a multiple of 2"
        num_bits_per_symbol = int(num_bits_per_symbol)

        points = qam(num_bits_per_symbol, normalize=normalize)
    if constellation_type=="pam":
        assert num_bits_per_symbol>0,\
            "num_bits_per_symbol must be integer"
        num_bits_per_symbol = int(num_bits_per_symbol)

        points = pam(num_bits_per_symbol, normalize=normalize)
    return points

def show(points, num_bits_per_symbol, labels=True, figsize=(7,7)):
    """Generate a scatter-plot of the constellation.

    Input
    -----
    labels : bool
        If `True`, the bit labels will be drawn next to each constellation
        point. Defaults to `True`.

    figsize : Two-element Tuple, float
        Width and height in inches. Defaults to `(7,7)`.

    Output
    ------
    : matplotlib.figure.Figure
        A handle to a matplot figure object.
    """
    maxval = np.max(np.abs(points))*1.05
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    plt.xlim(-maxval, maxval)
    plt.ylim(-maxval, maxval)
    plt.scatter(np.real(points), np.imag(points))
    ax.set_aspect("equal", adjustable="box")
    plt.xlabel("Real Part")
    plt.ylabel("Imaginary Part")
    plt.grid(True, which="both", axis="both")
    plt.title("Constellation Plot")
    if labels is True:
        for j, p in enumerate(points):
            plt.annotate(
                np.binary_repr(j, num_bits_per_symbol),
                (np.real(p), np.imag(p))
            )
    return fig

def plotcomplex(y):
    plt.figure(figsize=(8,8))
    plt.axes().set_aspect(1)
    plt.grid(True)
    plt.title('Channel output')
    plt.xlabel('Real Part')
    plt.ylabel('Imaginary Part')
    #plt.scatter(tf.math.real(y), tf.math.imag(y))
    plt.scatter(np.real(y), np.imag(y))
    plt.tight_layout()

# Input: DeepMIMO dataset, UE and BS indices to be included
#
# For a given 1D vector of BS or UE indices, the generated dataset will be stacked as different samples
#
# By default, the adapter will only select the first BS of the DeepMIMO dataset and all UEs
# The adapter assumes BSs are transmitters and users are receivers. 
# Uplink channels could be generated using (transpose) reciprocity.
#
# For multi-user channels, provide a 2D numpy matrix of size (num_samples x num_rx)
#
# Examples:
# ue_idx = np.array([[0, 1 ,2], [1, 2, 3]]) generates (num_bs x 3 UEs) channels
# with 2 data samples from the BSs to the UEs [0, 1, 2] and [1, 2, 3], respectively.
#
# For single-basestation channels with the data from different basestations stacked,
# provide a 1D array of basestation indices
#
# For multi-BS channels, provide a 2D numpy matrix of (num_samples x num_tx)
#
# Examples:
# bs_idx = np.array([[0, 1], [2, 3], [4, 5]]) generates (2 BSs x num_rx) channels
# by stacking the data of channels from the basestations (0 and 1), (2 and 3), 
# and (4 and 5) to the UEs.
#
class DeepMIMODataset(Dataset):
    def __init__(self, DeepMIMO_dataset, bs_idx = None, ue_idx = None):
        self.dataset = DeepMIMO_dataset  
        # Set bs_idx based on given parameters
        # If no input is given, choose the first basestation
        if bs_idx is None:
            bs_idx = np.array([[0]])
        self.bs_idx = self._verify_idx(bs_idx)
        
        # Set ue_idx based on given parameters
        # If no input is given, set all user indices
        if ue_idx is None:
            ue_idx = np.arange(DeepMIMO_dataset[0]['user']['channel'].shape[0])
        self.ue_idx = self._verify_idx(ue_idx) #(9231, 1)
        
        # Extract number of antennas from the DeepMIMO dataset
        self.num_rx_ant = DeepMIMO_dataset[0]['user']['channel'].shape[1] #1 
        self.num_tx_ant = DeepMIMO_dataset[0]['user']['channel'].shape[2] #16
        
        # Determine the number of samples based on the given indices
        self.num_samples_bs = self.bs_idx.shape[0] #1
        self.num_samples_ue = self.ue_idx.shape[0] #9231
        self.num_samples = self.num_samples_bs * self.num_samples_ue #9231
        
        # Determine the number of tx and rx elements in each channel sample based on the given indices
        self.num_rx = self.ue_idx.shape[1] #1
        self.num_tx = self.bs_idx.shape[1] #1
        
        # Determine the number of available paths in the DeepMIMO dataset
        self.num_paths = DeepMIMO_dataset[0]['user']['channel'].shape[-1] #10
        self.num_time_steps = 1 # Time step = 1 for static scenarios
        
        # The required path power shape
        self.ch_shape = (self.num_rx, 
                         self.num_rx_ant, 
                         self.num_tx, 
                         self.num_tx_ant, 
                         self.num_paths, 
                         self.num_time_steps) #(rx=1, rx_ant=1, tx=1, tx_ant=16, paths=10, timestesp=1)
        
        # The required path delay shape 
        self.t_shape = (self.num_rx, self.num_tx, self.num_paths) #(rx=1,tx=1,paths=10)
    
    def __getitem__(self, index):
        ue_idx = index // self.num_samples_bs
        bs_idx = index % self.num_samples_bs
        # Generate zero vectors
        a = np.zeros(self.ch_shape, dtype=np.csingle)
        tau = np.zeros(self.t_shape, dtype=np.single)
        # Place the DeepMIMO dataset power and delays into the channel sample
        for i_ch in range(self.num_rx): # for each receiver in the sample
            for j_ch in range(self.num_tx): # for each transmitter in the sample
                i_ue = self.ue_idx[ue_idx][i_ch] # UE channel sample i - channel RX i_ch
                i_bs = self.bs_idx[bs_idx][j_ch] # BS channel sample i - channel TX j_ch
                a[i_ch, :, j_ch, :, :, 0] = self.dataset[i_bs]['user']['channel'][i_ue]
                tau[i_ch, j_ch, :self.dataset[i_bs]['user']['paths'][i_ue]['num_paths']] = self.dataset[i_bs]['user']['paths'][i_ue]['ToA'] 
        return a, tau ## yield this sample h=(num_rx=1, 1, num_tx=1, 16, 10, 1), tau=(num_rx=1,num_tx=1,ToA=10)
    
    def __len__(self):
        return self.num_samples
    
    # Verify the index values given as input
    def _verify_idx(self, idx):
        idx = self._idx_to_numpy(idx)
        idx = self._numpy_size_check(idx)
        return idx
    
    # Convert the possible input types to numpy (integer - range - list)
    def _idx_to_numpy(self, idx):
        if isinstance(idx, int): # If the input is an integer - a single ID - convert it to 2D numpy array
            idx = np.array([[idx]])
        elif isinstance(idx, list) or isinstance(idx, range): # If the input is a list or range - convert it to a numpy array
            idx = np.array(idx)
        elif isinstance(idx, np.ndarray):
            pass
        else:
            raise TypeError('The index input type must be an integer, list, or numpy array!') 
        return idx
    
    # Check the size of the given input and convert it to a 2D matrix of proper shape (num_tx x num_samples) or (num_rx x num_samples)
    def _numpy_size_check(self, idx):
        if len(idx.shape) == 1:
            idx = idx.reshape((-1, 1))
        elif len(idx.shape) == 2:
            pass
        else:
            raise ValueError('The index input must be integer, vector or 2D matrix!')
        return idx
    
    # Override length of the generator to provide the available number of samples
    def __len__(self):
        return self.num_samples
                
def get_deepMIMOdata(scenario='O1_60', dataset_folder=r'D:\Dataset\CommunicationDataset\O1_60'):
    # Load the default parameters
    parameters = DeepMIMO.default_params()

    # Set scenario name
    parameters['scenario'] = scenario #https://deepmimo.net/scenarios/o1-scenario/

    # Set the main folder containing extracted scenarios
    parameters['dataset_folder'] = dataset_folder #r'D:\Dataset\CommunicationDataset\O1_60'

    # To only include 10 strongest paths in the channel computation, set
    parameters['num_paths'] = 10

    # To activate only the first basestation, set
    parameters['active_BS'] = np.array([1])
    # To activate the basestations 6, set
    #parameters['active_BS'] = np.array([6])

    parameters['OFDM']['bandwidth'] = 0.05 # 50 MHz
    print(parameters['OFDM']['subcarriers']) #512
    #parameters['OFDM']['subcarriers'] = 512 # OFDM with 512 subcarriers
    #parameters['OFDM']['subcarriers_limit'] = 64 # Keep only first 64 subcarriers

    # To activate the user rows 1-5, set
    parameters['user_row_first'] = 1 #400 # First user row to be included in the dataset
    parameters['user_row_last'] = 100 #450 # Last user row to be included in the dataset

    # Consider 3 active basestations
    #parameters['active_BS'] = np.array([1, 5, 8])
    # Configuration of the antenna arrays
    parameters['bs_antenna']['shape'] = np.array([16, 1, 1]) # BS antenna shape through [x, y, z] axes
    parameters['ue_antenna']['shape'] = np.array([1, 1, 1]) # UE antenna shape through [x, y, z] axes, single antenna

    # The OFDM_channels parameter allows choosing between the generation of channel impulse
    # responses (if set to 0) or frequency domain channels (if set to 1).
    # It is set to 0 for this simulation, as the channel responses in frequency domain
    parameters['OFDM_channels'] = 0

    # Generate data
    DeepMIMO_dataset = DeepMIMO.generate_data(parameters)

    ## User locations
    active_bs_idx = 0 # Select the first active basestation in the dataset
    print(DeepMIMO_dataset[active_bs_idx]['user'].keys()) #['paths', 'LoS', 'location', 'distance', 'pathloss', 'channel']
    print(DeepMIMO_dataset[active_bs_idx]['user']['location'].shape) #(9231, 3)  num_ue_locations: 9231
    j=0 #user j
    print(DeepMIMO_dataset[active_bs_idx]['user']['location'][j]) #The Euclidian location of the user in the form of [x, y, z].

    # Number of basestations
    print(len(DeepMIMO_dataset)) #1
    # Keys of a basestation dictionary
    print(DeepMIMO_dataset[0].keys()) #['user', 'basestation', 'location']
    # Keys of a channel
    print(DeepMIMO_dataset[0]['user'].keys()) #['paths', 'LoS', 'location', 'distance', 'pathloss', 'channel']
    # Number of UEs
    print(len(DeepMIMO_dataset[0]['user']['channel'])) #9231
    print(DeepMIMO_dataset[active_bs_idx]['user']['channel'].shape) #(num_ue_locations=9231, 1, bs_antenna=16, strongest_path=10) 
    # Shape of the channel matrix
    print(DeepMIMO_dataset[0]['user']['channel'].shape) #(9231, 1, 16, 10)

    i=0
    j=0
    #The channel matrix between basestation i and user j
    DeepMIMO_dataset[i]['user']['channel'][j]
    #Float matrix of size (number of RX antennas) x (number of TX antennas) x (number of OFDM subcarriers)

    # Shape of BS 0 - UE 0 channel
    print(DeepMIMO_dataset[i]['user']['channel'][0].shape) #(1, 16, 10)
    
    # Path properties of BS 0 - UE 0
    print(DeepMIMO_dataset[i]['user']['paths'][j]) #Ray-tracing Path Parameters in dictionary
    #Azimuth and zenith angle-of-arrivals – degrees (DoA_phi, DoA_theta)
    # Azimuth and zenith angle-of-departure – degrees (DoD_phi, DoD_theta)
    # Time of arrival – seconds (ToA)
    # Phase – degrees (phase)
    # Power – watts (power)
    # Number of paths (num_paths)

    print(DeepMIMO_dataset[i]['user']['LoS'][j]) #Integer of values {-1, 0, 1} indicates the existence of the LOS path in the channel.
    # (1): The LoS path exists.
    # (0): Only NLoS paths exist. The LoS path is blocked (LoS blockage).
    # (-1): No paths exist between the transmitter and the receiver (Full blockage).

    print(DeepMIMO_dataset[i]['user']['distance'][j])
    #The Euclidian distance between the RX and TX locations in meters.

    print(DeepMIMO_dataset[i]['user']['pathloss'][j])
    #The combined path-loss of the channel between the RX and TX in dB.


    print(DeepMIMO_dataset[i]['location'])
    #Basestation Location [x, y, z].
    print(DeepMIMO_dataset[i]['user']['location'][j])
    #The Euclidian location of the user in the form of [x, y, z].

    plt.figure(figsize=(12,8))
    plt.scatter(DeepMIMO_dataset[active_bs_idx]['user']['location'][:, 1], # y-axis location of the users
            DeepMIMO_dataset[active_bs_idx]['user']['location'][:, 0], # x-axis location of the users
            s=1, marker='x', c='C0', label='The users located on the rows %i to %i (R%i to R%i)'%
            (parameters['user_row_first'], parameters['user_row_last'],
            parameters['user_row_first'], parameters['user_row_last']))
    # First 181 users correspond to the first row
    plt.scatter(DeepMIMO_dataset[active_bs_idx]['user']['location'][0:181, 1],
            DeepMIMO_dataset[active_bs_idx]['user']['location'][0:181, 0],
            s=1, marker='x', c='C1', label='First row of users (R%i)'% (parameters['user_row_first']))

    ## Basestation location
    plt.scatter(DeepMIMO_dataset[active_bs_idx]['location'][1],
            DeepMIMO_dataset[active_bs_idx]['location'][0],
            s=50.0, marker='o', c='C2', label='Basestation')

    plt.gca().invert_xaxis() # Invert the x-axis to align the figure with the figure above
    plt.ylabel('x-axis')
    plt.xlabel('y-axis')
    plt.grid()
    plt.legend();

    dataset = DeepMIMO_dataset
    ## Visualization of a channel matrix
    plt.figure()
    # Visualize channel magnitude response
    # First, select indices of a user and bs
    ue_idx = 0
    bs_idx = 0
    # Import channel
    channel = dataset[bs_idx]['user']['channel'][ue_idx]
    # Take only the first antenna pair
    plt.imshow(np.abs(np.squeeze(channel).T))
    plt.title('Channel Magnitude Response')
    plt.xlabel('TX Antennas')
    plt.ylabel('Subcarriers')

    ## Visualization of the UE positions and path-losses
    loc_x = dataset[bs_idx]['user']['location'][:, 0] #(9231,)
    loc_y = dataset[bs_idx]['user']['location'][:, 1]
    loc_z = dataset[bs_idx]['user']['location'][:, 2]
    pathloss = dataset[bs_idx]['user']['pathloss'] #(9231,
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    im = ax.scatter(loc_x, loc_y, loc_z, c=pathloss)
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_zlabel('z (m)')

    bs_loc_x = dataset[bs_idx]['basestation']['location'][:, 0]
    bs_loc_y = dataset[bs_idx]['basestation']['location'][:, 1]
    bs_loc_z = dataset[bs_idx]['basestation']['location'][:, 2]
    ax.scatter(bs_loc_x, bs_loc_y, bs_loc_z, c='r')
    ttl = plt.title('UE and BS Positions')

    fig = plt.figure()
    ax = fig.add_subplot()
    im = ax.scatter(loc_x, loc_y, c=pathloss)
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    fig.colorbar(im, ax=ax)
    ttl = plt.title('UE Grid Path-loss (dB)')

    return DeepMIMO_dataset

class PilotPattern():
    r"""Class defining a pilot pattern for an OFDM ResourceGrid.

    Parameters
    ----------
    mask : [num_tx, num_streams_per_tx, num_ofdm_symbols, num_effective_subcarriers], bool
        Tensor indicating resource elements that are reserved for pilot transmissions.

    pilots : [num_tx, num_streams_per_tx, num_pilots], complex
        The pilot symbols to be mapped onto the ``mask``.

    normalize : bool
        Indicates if the ``pilots`` should be normalized to an average
        energy of one across the last dimension. This can be useful to
        ensure that trainable ``pilots`` have a finite energy.
        Defaults to `False`.

    dtype : Dtype
        Defines the datatype for internal calculations and the output
        dtype. Defaults to `complex64`.
    """
    def __init__(self, mask, pilots, trainable=False, normalize=False,
                 dtype=np.complex64):
        super().__init__()
        self._dtype = dtype
        #self._mask = tf.cast(mask, tf.int32)
        self._mask = mask.astype(np.int32) #(1, 1, 14, 76)
        #self._pilots = tf.Variable(tf.cast(pilots, self._dtype), trainable)
        self._pilots = pilots.astype(self._dtype) #(1, 1, 0) complex
        self.normalize = normalize
        #self._check_settings()

    @property
    def num_tx(self):
        """Number of transmitters"""
        return self._mask.shape[0]

    @property
    def num_streams_per_tx(self):
        """Number of streams per transmitter"""
        return self._mask.shape[1]

    @ property
    def num_ofdm_symbols(self):
        """Number of OFDM symbols"""
        return self._mask.shape[2]

    @ property
    def num_effective_subcarriers(self):
        """Number of effectvie subcarriers"""
        return self._mask.shape[3]

    @property
    def num_pilot_symbols(self):
        """Number of pilot symbols per transmit stream."""
        #return tf.shape(self._pilots)[-1]
        return np.shape(self._pilots)[-1]


    @property
    def num_data_symbols(self):
        """ Number of data symbols per transmit stream."""
        # return tf.shape(self._mask)[-1]*tf.shape(self._mask)[-2] - \
        #        self.num_pilot_symbols
        return np.shape(self._mask)[-1]*np.shape(self._mask)[-2] - \
               self.num_pilot_symbols

    @property
    def normalize(self):
        """Returns or sets the flag indicating if the pilots
           are normalized or not
        """
        return self._normalize

    @normalize.setter
    def normalize(self, value):
        #self._normalize = tf.cast(value, tf.bool)
        self._normalize =value


    @property
    def mask(self):
        """Mask of the pilot pattern"""
        return self._mask

    @property
    def pilots(self):
        """Returns or sets the possibly normalized tensor of pilot symbols.
           If pilots are normalized, the normalization will be applied
           after new values for pilots have been set. If this is
           not the desired behavior, turn normalization off.
        """
        def norm_pilots():
            #scale = tf.abs(self._pilots)**2
            scale = np.abs(self._pilots)**2
            #scale = 1/tf.sqrt(tf.reduce_mean(scale, axis=-1, keepdims=True))
            scale = 1/np.sqrt(np.mean(scale, axis=-1, keepdims=True))
            #scale = tf.cast(scale, self._dtype)
            scale = scale.astype(self._dtype)

            return scale*self._pilots

        #conditionally execute different operations based on the value of a boolean tensor
        #return tf.cond(self.normalize, norm_pilots, lambda: self._pilots)
        if self.normalize:
            return norm_pilots()
        else:
            return self._pilots

    @pilots.setter
    def pilots(self, value):
        self._pilots.assign(value)


    def show(self, tx_ind=None, stream_ind=None, show_pilot_ind=False):
        """Visualizes the pilot patterns for some transmitters and streams.

        Input
        -----
        tx_ind : list, int
            Indicates the indices of transmitters to be included.
            Defaults to `None`, i.e., all transmitters included.

        stream_ind : list, int
            Indicates the indices of streams to be included.
            Defaults to `None`, i.e., all streams included.

        show_pilot_ind : bool
            Indicates if the indices of the pilot symbols should be shown.

        Output
        ------
        list : matplotlib.figure.Figure
            List of matplot figure objects showing each the pilot pattern
            from a specific transmitter and stream.
        """
        # mask = self.mask.numpy() #(1, 1, 14, 76)
        # pilots = self.pilots.numpy() #(1, 1, 152)
        mask = self.mask
        pilots = self.pilots

        if tx_ind is None:
            tx_ind = range(0, self.num_tx) #range(0,1)
        elif not isinstance(tx_ind, list):
            tx_ind = [tx_ind]

        if stream_ind is None:
            stream_ind = range(0, self.num_streams_per_tx) #range(0,1)
        elif not isinstance(stream_ind, list):
            stream_ind = [stream_ind]

        figs = []
        for i in tx_ind: #range(0,1)
            for j in stream_ind: #range(0,1)
                q = np.zeros_like(mask[0,0]) #(14, 76)
                q[np.where(mask[i,j])] = (np.abs(pilots[i,j])==0) + 1
                legend = ["Data", "Pilots", "Masked"]
                fig = plt.figure()
                plt.title(f"TX {i} - Stream {j}")
                plt.xlabel("OFDM Symbol")
                plt.ylabel("Subcarrier Index")
                plt.xticks(range(0, q.shape[1]))
                cmap = plt.cm.tab20c
                b = np.arange(0, 4)
                norm = colors.BoundaryNorm(b, cmap.N)
                im = plt.imshow(np.transpose(q), origin="lower", aspect="auto", norm=norm, cmap=cmap)
                cbar = plt.colorbar(im)
                cbar.set_ticks(b[:-1]+0.5)
                cbar.set_ticklabels(legend)

                if show_pilot_ind:
                    c = 0
                    for t in range(self.num_ofdm_symbols):
                        for k in range(self.num_effective_subcarriers):
                            if mask[i,j][t,k]:
                                if np.abs(pilots[i,j,c])>0:
                                    plt.annotate(c, [t, k])
                                c+=1
                figs.append(fig)

        return figs

class EmptyPilotPattern(PilotPattern):
    """Creates an empty pilot pattern.

    Generates a instance of :class:`PilotPattern` with
    an empty ``mask`` and ``pilots``.

    Parameters
    ----------
    num_tx : int
        Number of transmitters.

    num_streams_per_tx : int
        Number of streams per transmitter.

    num_ofdm_symbols : int
        Number of OFDM symbols.

    num_effective_subcarriers : int
        Number of effective subcarriers
        that are available for the transmission of data and pilots.
        Note that this number is generally smaller than the ``fft_size``
        due to nulled subcarriers.

    dtype : Dtype
        Defines the datatype for internal calculations and the output
        dtype. Defaults to `complex64`.
    """
    def __init__(self,
                 num_tx,
                 num_streams_per_tx,
                 num_ofdm_symbols,
                 num_effective_subcarriers,
                 dtype=np.complex64):

        assert num_tx > 0, \
            "`num_tx` must be positive`."
        assert num_streams_per_tx > 0, \
            "`num_streams_per_tx` must be positive`."
        assert num_ofdm_symbols > 0, \
            "`num_ofdm_symbols` must be positive`."
        assert num_effective_subcarriers > 0, \
            "`num_effective_subcarriers` must be positive`."

        shape = [num_tx, num_streams_per_tx, num_ofdm_symbols,
                      num_effective_subcarriers] #[1, 1, 14, 76]
        #mask = tf.zeros(shape, tf.bool)
        mask = np.zeros(shape, np.bool_)
        #pilots = tf.zeros(shape[:2]+[0], dtype)
        pilots = np.zeros(shape[:2]+[0], np.bool_) #(1, 1, 0)
        super().__init__(mask, pilots, trainable=False, normalize=False,
                         dtype=dtype)

class MyResourceGrid():
    r"""Defines a `ResourceGrid` spanning multiple OFDM symbols and subcarriers.

    Parameters
    ----------
        num_ofdm_symbols : int
            Number of OFDM symbols.

        fft_size : int
            FFT size (, i.e., the number of subcarriers).

        subcarrier_spacing : float
            The subcarrier spacing in Hz.

        num_tx : int
            Number of transmitters.

        num_streams_per_tx : int
            Number of streams per transmitter.

        cyclic_prefix_length : int
            Length of the cyclic prefix.

        num_guard_carriers : int
            List of two integers defining the number of guardcarriers at the
            left and right side of the resource grid.

        dc_null : bool
            Indicates if the DC carrier is nulled or not.

        pilot_pattern : One of [None, "kronecker", "empty", PilotPattern]
            Defaults to `None` which is equivalent to `"empty"`.

        pilot_ofdm_symbol_indices : List, int
            List of indices of OFDM symbols reserved for pilot transmissions.
            Only needed if ``pilot_pattern="kronecker"``. Defaults to `None`.

        dtype : 
            Defines the datatype for internal calculations and the output
    """
    def __init__(self,
                 num_ofdm_symbols,
                 fft_size,
                 subcarrier_spacing,
                 num_tx=1,
                 num_streams_per_tx=1,
                 cyclic_prefix_length=0,
                 num_guard_carriers=(0,0),
                 dc_null=False,
                 pilot_pattern=None,
                 pilot_ofdm_symbol_indices=None,
                 dtype=np.complex64):
        super().__init__()
        self._dtype = dtype
        self._num_ofdm_symbols = num_ofdm_symbols #14
        self._fft_size = fft_size #76
        self._subcarrier_spacing = subcarrier_spacing #30000
        self._cyclic_prefix_length = int(cyclic_prefix_length) #6
        self._num_tx = num_tx #1
        self._num_streams_per_tx = num_streams_per_tx #1
        self._num_guard_carriers = np.array(num_guard_carriers) #(0,0)
        self._dc_null = dc_null #False
        self._pilot_ofdm_symbol_indices = pilot_ofdm_symbol_indices #[2,11]
        self.pilot_pattern = pilot_pattern #'kronecker'
        self._check_settings()

    @property
    def cyclic_prefix_length(self):
        """Length of the cyclic prefix."""
        return self._cyclic_prefix_length

    @property
    def num_tx(self):
        """Number of transmitters."""
        return self._num_tx

    @property
    def num_streams_per_tx(self):
        """Number of streams  per transmitter."""
        return self._num_streams_per_tx

    @property
    def num_ofdm_symbols(self):
        """The number of OFDM symbols of the resource grid."""
        return self._num_ofdm_symbols

    @property
    def num_resource_elements(self):
        """Number of resource elements."""
        return self._fft_size*self._num_ofdm_symbols

    @property
    def num_effective_subcarriers(self):
        """Number of subcarriers used for data and pilot transmissions."""
        n = self._fft_size - self._dc_null - np.sum(self._num_guard_carriers) #no change 76
        return n

    @property
    def effective_subcarrier_ind(self):
        """Returns the indices of the effective subcarriers."""
        num_gc = self._num_guard_carriers
        sc_ind = range(num_gc[0], self.fft_size-num_gc[1])
        if self.dc_null:
            sc_ind = np.delete(sc_ind, self.dc_ind-num_gc[0])
        return sc_ind

    @property
    def num_data_symbols(self):
        """Number of resource elements used for data transmissions."""
        n = self.num_effective_subcarriers * self._num_ofdm_symbols - \
               self.num_pilot_symbols
        #return tf.cast(n, tf.int32)
        return n.astype(np.int32)


    @property
    def num_pilot_symbols(self):
        """Number of resource elements used for pilot symbols."""
        return self.pilot_pattern.num_pilot_symbols

    @property
    def num_zero_symbols(self):
        """Number of empty resource elements."""
        n = (self._fft_size-self.num_effective_subcarriers) * \
               self._num_ofdm_symbols
        #return tf.cast(n, tf.int32)
        return n.astype(np.int32)

    @property
    def num_guard_carriers(self):
        """Number of left and right guard carriers."""
        return self._num_guard_carriers

    @property
    def dc_ind(self):
        """Index of the DC subcarrier.

        If ``fft_size`` is odd, the index is (``fft_size``-1)/2.
        If ``fft_size`` is even, the index is ``fft_size``/2.
        """
        return int(self._fft_size/2 - (self._fft_size%2==1)/2)

    @property
    def fft_size(self):
        """The FFT size."""
        return self._fft_size

    @property
    def subcarrier_spacing(self):
        """The subcarrier spacing [Hz]."""
        return self._subcarrier_spacing

    @property
    def ofdm_symbol_duration(self):
        """Duration of an OFDM symbol with cyclic prefix [s]."""
        return (1. + self.cyclic_prefix_length/self.fft_size) \
                / self.subcarrier_spacing

    @property
    def bandwidth(self):
        """The occupied bandwidth [Hz]: ``fft_size*subcarrier_spacing``."""
        return self.fft_size*self.subcarrier_spacing

    @property
    def num_time_samples(self):
        """The number of time-domain samples occupied by the resource grid."""
        return (self.fft_size + self.cyclic_prefix_length) \
                * self._num_ofdm_symbols

    @property
    def dc_null(self):
        """Indicates if the DC carriers is nulled or not."""
        return self._dc_null

    @property
    def pilot_pattern(self):
        """The used PilotPattern."""
        return self._pilot_pattern

    @pilot_pattern.setter
    def pilot_pattern(self, value):
        if value is None:
            value = EmptyPilotPattern(self._num_tx,
                                      self._num_streams_per_tx,
                                      self._num_ofdm_symbols,
                                      self.num_effective_subcarriers,
                                      dtype=self._dtype)
        elif isinstance(value, PilotPattern):
            pass
        elif isinstance(value, str):
            assert value in ["kronecker", "empty"],\
                "Unknown pilot pattern"
            if value=="empty":
                value = EmptyPilotPattern(self._num_tx,
                                      self._num_streams_per_tx,
                                      self._num_ofdm_symbols,
                                      self.num_effective_subcarriers,
                                      dtype=self._dtype)
            elif value=="kronecker":
                assert self._pilot_ofdm_symbol_indices is not None,\
                    "You must provide pilot_ofdm_symbol_indices."
                #Kronecker not implemented
                # value = KroneckerPilotPattern(self,
                #         self._pilot_ofdm_symbol_indices, dtype=self._dtype)
                value = EmptyPilotPattern(self._num_tx,
                                      self._num_streams_per_tx,
                                      self._num_ofdm_symbols,
                                      self.num_effective_subcarriers,
                                      dtype=self._dtype)
        else:
            raise ValueError("Unsupported pilot_pattern")
        self._pilot_pattern = value

    def _check_settings(self):
        """Validate that all properties define a valid resource grid"""
        assert self._num_ofdm_symbols > 0, \
            "`num_ofdm_symbols` must be positive`."
        assert self._fft_size > 0, \
            "`fft_size` must be positive`."
        assert self._cyclic_prefix_length>=0, \
            "`cyclic_prefix_length must be nonnegative."
        assert self._cyclic_prefix_length<=self._fft_size, \
            "`cyclic_prefix_length cannot be longer than `fft_size`."
        assert self._num_tx > 0, \
            "`num_tx` must be positive`."
        assert self._num_streams_per_tx > 0, \
            "`num_streams_per_tx` must be positive`."
        assert len(self._num_guard_carriers)==2, \
            "`num_guard_carriers` must have two elements."
        assert np.all(np.greater_equal(self._num_guard_carriers, 0)), \
            "`num_guard_carriers` must have nonnegative entries."
        assert np.sum(self._num_guard_carriers)<=self._fft_size-self._dc_null,\
            "Total number of guardcarriers cannot be larger than `fft_size`."
        assert self._dtype in [np.complex64, np.complex128], \
            "dtype must be complex64 or complex128"
        return True

    def build_type_grid(self):
        """Returns a tensor indicating the type of each resource element.

        Resource elements can be one of

        - 0 : Data symbol
        - 1 : Pilot symbol
        - 2 : Guard carrier symbol
        - 3 : DC carrier symbol

        Output
        ------
        : [num_tx, num_streams_per_tx, num_ofdm_symbols, fft_size], tf.int32
            Tensor indicating for each transmitter and stream the type of
            the resource elements of the corresponding resource grid.
            The type can be one of [0,1,2,3] as explained above.
        """
        shape = [self._num_tx, self._num_streams_per_tx, self._num_ofdm_symbols] #[1,1,14]
        #gc_l = 2*tf.ones(shape+[self._num_guard_carriers[0]], tf.int32) #(1, 1, 14, 0)
        gc_l = 2*np.ones(shape+[self._num_guard_carriers[0]], np.int32)
        #gc_r = 2*tf.ones(shape+[self._num_guard_carriers[1]], tf.int32) #(1, 1, 14, 0)
        gc_r = 2*np.ones(shape+[self._num_guard_carriers[1]], np.int32) #(1, 1, 14, 0)
        #dc   = 3*tf.ones(shape + [tf.cast(self._dc_null, tf.int32)], tf.int32) #(1, 1, 14, 0)
        dc   = 3*np.ones(shape + [int(self._dc_null)], np.int32) #(1, 1, 14, 0)
        mask = self.pilot_pattern.mask #(1, 1, 14, 76)
        split_ind = self.dc_ind-self._num_guard_carriers[0] #38-0=38
        # rg_type = tf.concat([gc_l,                 # Left Guards
        #                      mask[...,:split_ind], # Data & pilots
        #                      dc,                   # DC
        #                      mask[...,split_ind:], # Data & pilots
        #                      gc_r], -1)            # Right guards
        rg_type = np.concatenate([gc_l,                 # Left Guards (1, 1, 14, 0)
                             mask[...,:split_ind], # Data & pilots
                             dc,                   # DC (1, 1, 14, 0)
                             mask[...,split_ind:], # Data & pilots
                             gc_r], -1)            # Right guards (1, 1, 14, 0)
        return rg_type #(1, 1, 14, 76) #38+38

    def show(self, tx_ind=0, tx_stream_ind=0):
        """Visualizes the resource grid for a specific transmitter and stream.

        Input
        -----
        tx_ind : int
            Indicates the transmitter index.

        tx_stream_ind : int
            Indicates the index of the stream.

        Output
        ------
        : `matplotlib.figure`
            A handle to a matplot figure object.
        """
        fig = plt.figure()
        data = self.build_type_grid()[tx_ind, tx_stream_ind] #0,0 =>[14,76]
        cmap = colors.ListedColormap([[60/256,8/256,72/256],
                              [45/256,91/256,128/256],
                              [45/256,172/256,111/256],
                              [250/256,228/256,62/256]])
        bounds=[0,1,2,3,4]
        norm = colors.BoundaryNorm(bounds, cmap.N)
        img = plt.imshow(np.transpose(data), interpolation="nearest",
                         origin="lower", cmap=cmap, norm=norm,
                         aspect="auto")
        cbar = plt.colorbar(img, ticks=[0.5, 1.5, 2.5,3.5],
                            orientation="vertical", shrink=0.8)
        cbar.set_ticklabels(["Data", "Pilot", "Guard carrier", "DC carrier"])
        plt.title("OFDM Resource Grid")
        plt.ylabel("Subcarrier Index")
        plt.xlabel("OFDM Symbol")
        plt.xticks(range(0, data.shape[0]))

        return fig
    
#ref from class Constellation
def CreateConstellation(constellation_type, num_bits_per_symbol,normalize=True):
    r"""
    Constellation(constellation_type, num_bits_per_symbol, initial_value=None, normalize=True, center=False, trainable=False, dtype=tf.complex64, **kwargs)

    Constellation that can be used by a (de)mapper.

    This class defines a constellation, i.e., a complex-valued vector of
    constellation points. A constellation can be trainable. The binary
    representation of the index of an element of this vector corresponds
    to the bit label of the constellation point. This implicit bit
    labeling is used by the ``Mapper`` and ``Demapper`` classes.

    Parameters
    ----------
    constellation_type : One of ["qam", "pam", "custom"], str
        For "custom", the constellation points are randomly initialized
        if no ``initial_value`` is provided.

    num_bits_per_symbol : int
        The number of bits per constellation symbol, e.g., 4 for QAM16.

    initial_value : :math:`[2^\text{num_bits_per_symbol}]`, NumPy array or Tensor
        Initial values of the constellation points. If ``normalize`` or
        ``center`` are `True`, the initial constellation might be changed.

    normalize : bool
        If `True`, the constellation is normalized to have unit power.
        Defaults to `True`.

    center : bool
        If `True`, the constellation is ensured to have zero mean.
        Defaults to `False`.

    trainable : bool
        If `True`, the constellation points are trainable variables.
        Defaults to `False`.

    dtype : [complex64, complex128], DType
        The dtype of the constellation.

    Output
    ------
    : :math:`[2^\text{num_bits_per_symbol}]`, ``dtype``
        The constellation.

    Note
    ----
    One can create a trainable PAM/QAM constellation. This is
    equivalent to creating a custom trainable constellation which is
    initialized with PAM/QAM constellation points.
    """
    num_bits_per_symbol = int(num_bits_per_symbol)
    if constellation_type=="qam":
        assert num_bits_per_symbol%2 == 0 and num_bits_per_symbol>0,\
            "num_bits_per_symbol must be a multiple of 2"
        num_bits_per_symbol = int(num_bits_per_symbol)

        points = qam(num_bits_per_symbol, normalize=normalize)
    if constellation_type=="pam":
        assert num_bits_per_symbol>0,\
            "num_bits_per_symbol must be integer"
        num_bits_per_symbol = int(num_bits_per_symbol)

        points = pam(num_bits_per_symbol, normalize=normalize)
    return points


class Mapper:
    def __init__(self,
                 constellation_type=None,
                 num_bits_per_symbol=None,
                 constellation=None,
                 return_indices=False,
                 #dtype=tf.complex64,
                 #**kwargs
                ):
          self.num_bits_per_symbol = num_bits_per_symbol
          self.binary_base = 2**np.arange(num_bits_per_symbol-1, -1, -1, dtype=int) #array([2, 1], dtype=int32) [8, 4, 2, 1]
          self.points = CreateConstellation(constellation_type, num_bits_per_symbol) #(4,)
          self._return_indices = return_indices
    
    def create_symbol(self, inputs):
        #inputs: (64, 1024) #batch_size, bits len
        new_shape = [-1] + [int(inputs.shape[-1] / self.num_bits_per_symbol), self.num_bits_per_symbol] #[-1, 512, 2]
        reinputs_reshaped = np.reshape(inputs, new_shape) #(64, 512, 2)
        # Convert the last dimension to an integer
        int_rep = reinputs_reshaped * self.binary_base #(64, 512, 2)
        int_rep = np.sum(int_rep, axis=-1) #(64, 512)
        int_rep = int_rep.astype(np.int32)
        print(int_rep.shape)
        # Map integers to constellation symbols
        #x = tf.gather(self.points, int_rep, axis=0)
        symbs_list = [self.points[val_int] for val_int in int_rep]
        symbols=np.array(symbs_list) #(64, 512) complex64
        print(symbols.dtype)
        return symbols
    
    def __call__(self, inputs): #(64, 1, 1, 2128)
        #convert inputs.shape to a python list
        input_shape = list(inputs.shape) #[64, 1, 1, 2128]
        # Reshape inputs to the desired format
        new_shape = [-1] + input_shape[1:-1] + \
           [int(input_shape[-1] / self.num_bits_per_symbol),
            self.num_bits_per_symbol] #[-1, 1, 1, 532, 4]
        #inputs_reshaped = tf.cast(tf.reshape(inputs, new_shape), tf.int32)
        inputs_reshaped = np.reshape(inputs, new_shape).astype(np.int32) #(64, 1, 1, 532, 4)

        # Convert the last dimension to an integer
        #int_rep = tf.reduce_sum(inputs_reshaped * self._binary_base, axis=-1)
        int_rep = inputs_reshaped * self.binary_base #(64, 1, 1, 532, 4)
        int_rep = np.sum(int_rep, axis=-1) #(64, 1, 1, 532)
        int_rep = int_rep.astype(np.int32) #(64, 1, 1, 532)

        # Map integers to constellation symbols
        #x = tf.gather(self.constellation.points, int_rep, axis=0)
        symbs_list = [self.points[val_int] for val_int in int_rep]
        x=np.array(symbs_list) #(64, 1, 1, 532)

        if self._return_indices:
            return x, int_rep
        else:
            return x

class BinarySource:
    """BinarySource(dtype=float32, seed=None, **kwargs)

    Layer generating random binary tensors.

    Parameters
    ----------
    dtype : DType
        Defines the output datatype of the layer.
        Defaults to `float32`.

    seed : int or None
        Set the seed for the random generator used to generate the bits.
        Set to `None` for random initialization of the RNG.

    Input
    -----
    shape : 1D tensor/array/list, int
        The desired shape of the output tensor.

    Output
    ------
    : ``shape``, ``dtype``
        Tensor filled with random binary values.
    """
    def __init__(self, dtype=np.float32, seed=None, **kwargs):
        #super().__init__(dtype=dtype, **kwargs)
        self._seed = seed
        if self._seed is not None:
            self._rng = np.random.RandomState(self._seed)

    def __call__(self, inputs): #inputs is shape
        if self._seed is not None:
            return self._rng.randint(low=0, high=2, size=inputs).astype(np.float32)
            # return tf.cast(self._rng.uniform(inputs, 0, 2, tf.int32),
            #                dtype=super().dtype)
        else:
            return np.random.randint(low=0, high=2, size=inputs).astype(np.float32)
            # return tf.cast(tf.random.uniform(inputs, 0, 2, tf.int32),
            #                dtype=super().dtype)
    

class SymbolSource():
    r"""SymbolSource(constellation_type=None, num_bits_per_symbol=None, constellation=None, return_indices=False, return_bits=False, seed=None, dtype=tf.complex64, **kwargs)

    Layer generating a tensor of arbitrary shape filled with random constellation symbols.
    Optionally, the symbol indices and/or binary representations of the
    constellation symbols can be returned.

    Parameters
    ----------
    constellation_type : One of ["qam", "pam", "custom"], str
        For "custom", an instance of :class:`Constellation`
        must be provided.

    num_bits_per_symbol : int
        The number of bits per constellation symbol.
        Only required for ``constellation_type`` in ["qam", "pam"].

    constellation :  Constellation

    return_indices : bool
        If enabled, the function also returns the symbol indices.
        Defaults to `False`.

    return_bits : bool
        If enabled, the function also returns the binary symbol
        representations (i.e., bit labels).
        Defaults to `False`.

    seed : int or None
        The seed for the random generator.
        `None` leads to a random initialization of the RNG.
        Defaults to `None`.

    dtype : One of [complex64, complex128], DType
        The output dtype. Defaults to complex64.

    Input
    -----
    shape : 1D tensor/array/list, int
        The desired shape of the output tensor.

    Output
    ------
    symbols : ``shape``, ``dtype``
        Tensor filled with random symbols of the chosen ``constellation_type``.

    symbol_indices : ``shape``, int32
        Tensor filled with the symbol indices.
        Only returned if ``return_indices`` is `True`.

    bits : [``shape``, ``num_bits_per_symbol``], int32
        Tensor filled with the binary symbol representations (i.e., bit labels).
        Only returned if ``return_bits`` is `True`.
    """
    def __init__(self,
                 constellation_type=None,
                 num_bits_per_symbol=None,
                 constellation=None,
                 return_indices=False,
                 return_bits=False,
                 seed=None,
                 dtype=np.complex64,
                 **kwargs
                ):
        super().__init__(dtype=dtype, **kwargs)
        self.points = CreateConstellation(constellation_type, num_bits_per_symbol) #(4,)
        self._num_bits_per_symbol = num_bits_per_symbol
        self._return_indices = return_indices
        self._return_bits = return_bits
        #self._binary_source = BinarySource(seed=seed, dtype=dtype.real_dtype)
        self._binary_source = BinarySource()
        self._mapper = Mapper(constellation=constellation,
                              return_indices=return_indices,
                              dtype=dtype)

    def call(self, inputs):
        #shape = tf.concat([inputs, [self._num_bits_per_symbol]], axis=-1)
        shape = np.concatenate([inputs, [self._num_bits_per_symbol]], axis=-1)
        shape = shape.astype(np.int32)
        #b = self._binary_source(tf.cast(shape, tf.int32))
        b = self._binary_source(shape)
        if self._return_indices:
            x, ind = self._mapper(b)
        else:
            x = self._mapper(b)

        #result = tf.squeeze(x, -1)
        result = np.squeeze(x, -1)
        if self._return_indices or self._return_bits:
            result = [result]
        if self._return_indices:
            #result.append(tf.squeeze(ind, -1))
            result.append(np.squeeze(ind, -1))
        if self._return_bits:
            result.append(b)

        return result
    
def subcarrier_frequencies(num_subcarriers, subcarrier_spacing,
                           dtype=np.complex64):
    r"""
    Compute the baseband frequencies of ``num_subcarrier`` subcarriers spaced by
    ``subcarrier_spacing``, i.e.,

    >>> # If num_subcarrier is even:
    >>> frequencies = [-num_subcarrier/2, ..., 0, ..., num_subcarrier/2-1] * subcarrier_spacing
    >>>
    >>> # If num_subcarrier is odd:
    >>> frequencies = [-(num_subcarrier-1)/2, ..., 0, ..., (num_subcarrier-1)/2] * subcarrier_spacing


    Input
    ------
    num_subcarriers : int
        Number of subcarriers

    subcarrier_spacing : float
        Subcarrier spacing [Hz]

    dtype

    Output
    ------
        frequencies : [``num_subcarrier``], float
            Baseband frequencies of subcarriers
    """
    real_dtype = np.float32

    #if tf.equal(tf.math.floormod(num_subcarriers, 2), 0):
    #num_subcarrier is even
    #use numpy to check num_subcarriers is an even number or not
    #if np.equal(np.floor(num_subcarriers/2), 0):
    if num_subcarriers%2 == 0:
        start=-num_subcarriers/2
        limit=num_subcarriers/2
    else:
        start=-(num_subcarriers-1)/2
        limit=(num_subcarriers-1)/2+1

    # frequencies = tf.range( start=start,
    #                         limit=limit,
    #                         dtype=real_dtype)
    frequencies = np.arange(start=start, stop=limit, dtype=real_dtype) #step=1
    frequencies = frequencies*subcarrier_spacing
    return frequencies

def myexpand_to_rank(tensor, target_rank, axis=-1):
    """Inserts as many axes to a tensor as needed to achieve a desired rank.

    This operation inserts additional dimensions to a ``tensor`` starting at
    ``axis``, so that so that the rank of the resulting tensor has rank
    ``target_rank``. The dimension index follows Python indexing rules, i.e.,
    zero-based, where a negative index is counted backward from the end.

    Args:
        tensor : A tensor.
        target_rank (int) : The rank of the output tensor.
            If ``target_rank`` is smaller than the rank of ``tensor``,
            the function does nothing.
        axis (int) : The dimension index at which to expand the
               shape of ``tensor``. Given a ``tensor`` of `D` dimensions,
               ``axis`` must be within the range `[-(D+1), D]` (inclusive).

    Returns:
        A tensor with the same data as ``tensor``, with
        ``target_rank``- rank(``tensor``) additional dimensions inserted at the
        index specified by ``axis``.
        If ``target_rank`` <= rank(``tensor``), ``tensor`` is returned.
    """
    #num_dims = tf.maximum(target_rank - tf.rank(tensor), 0)
    num_dims = np.maximum(target_rank - tensor.ndim, 0) #difference in rank, >0 7
    #Adds multiple length-one dimensions to a tensor.
    #It inserts ``num_dims`` dimensions of length one starting from the dimension ``axis``
    #output = insert_dims(tensor, num_dims, axis)
    rank = tensor.ndim #1
    axis = axis if axis>=0 else rank+axis+1 #0
    #shape = tf.shape(tensor)
    shape = np.shape(tensor) #(76,)
    new_shape = np.concatenate([shape[:axis],
                           np.ones([num_dims], np.int32),
                           shape[axis:]], 0) #(8,) array([ 1.,  1.,  1.,  1.,  1.,  1.,  1., 76.])
    # new_shape = tf.concat([shape[:axis],
    #                        tf.ones([num_dims], tf.int32),
    #                        shape[axis:]], 0)
    #output = tf.reshape(tensor, new_shape)
    new_shape = new_shape.astype(np.int32)
    output = np.reshape(tensor, new_shape) #(76,)

    return output #(1, 1, 1, 1, 1, 1, 1, 76)

def cir_to_ofdm_channel(frequencies, a, tau, normalize=False):
    r"""
    Compute the frequency response of the channel at ``frequencies``.

    Given a channel impulse response
    :math:`(a_{m}, \tau_{m}), 0 \leq m \leq M-1` (inputs ``a`` and ``tau``),
    the channel frequency response for the frequency :math:`f`
    is computed as follows:

    .. math::
        \widehat{h}(f) = \sum_{m=0}^{M-1} a_{m} e^{-j2\pi f \tau_{m}}

    Input
    ------
    frequencies : [fft_size], tf.float
        Frequencies at which to compute the channel response

    a : [batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, num_time_steps], complex
        Path coefficients

    tau : [batch size, num_rx, num_tx, num_paths] or [batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths], float
        Path delays

    normalize : bool
        to ensure unit average energy per resource element. Defaults to `False`.

    Output
    -------
    h_f : [batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_time_steps, fft_size], complex
        Channel frequency responses at ``frequencies``
    """

    real_dtype = tau.dtype #torch.float32

    if len(tau.shape) == 4:
        # Expand dims to broadcast with h. Add the following dimensions:
        #  - number of rx antennas (2)
        #  - number of tx antennas (4)
        tau_tmp=np.expand_dims(tau, axis=2) #[64, 1, 1, 10][batch size, num_rx, num_tx, num_paths] => [batch size, num_rx, 1, num_tx, num_paths]
        tau = np.expand_dims(tau_tmp, axis=4) #[batch size, num_rx, 1, num_tx, 1, num_paths] (64, 1, 1, 1, 1, 10)
        #tau = tf.expand_dims(tf.expand_dims(tau, axis=2), axis=4)
        # Broadcast is not supported yet by TF for such high rank tensors.
        # We therefore do part of it manually
        #tau = tf.tile(tau, [1, 1, 1, 1, a.shape[4], 1])
        tau = np.tile(tau, [1, 1, 1,1, a.shape[4], 1]) #(64, 1, 1, 1, 16, 10)

    # Add a time samples dimension for broadcasting
    tau = np.expand_dims(tau, axis=6) #[batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, fft_size] (64, 1, 1, 1, 16, 10, 1)
    #tau = tf.expand_dims(tau, axis=6)

    # Bring all tensors to broadcastable shapes
    tau = np.expand_dims(tau, axis=-1) ##[batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, fft_size , 1] (64, 1, 1, 1, 16, 10, 1, 1)
    #tau = tf.expand_dims(tau, axis=-1)
    h = np.expand_dims(a, axis=-1) #[batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, num_time_steps, 1] (64, 1, 1, 1, 16, 10, 1, 1)
    #h = tf.expand_dims(a, axis=-1)
    frequencies = myexpand_to_rank(frequencies, tau.ndim, axis=0) #(1, 1, 1, 1, 1, 1, 1, 76)

    ## Compute the Fourier transforms of all cluster taps
    # Exponential component
    # e = tf.exp(tf.complex(tf.constant(0, real_dtype),
    #     -2*PI*frequencies*tau))
    tmp_complex = 0 - 1j*2*np.pi*frequencies*tau #(64, 1, 1, 1, 16, 10, 1, 76)
    e = np.exp(tmp_complex)

    h_f = h*e #(64, 1, 1, 1, 16, 10, 1, 76)
    # Sum over all clusters to get the channel frequency responses
    #h_f = tf.reduce_sum(h_f, axis=-3)
    h_f = np.sum(h_f, axis=-3) #(64, 1, 1, 1, 16, 1, 76) #combine 10 paths
    #h_f : [batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_time_steps, fft_size]

    if normalize:
        # Normalization is performed such that for each batch example and
        # link the energy per resource grid is one.
        # Average over TX antennas, RX antennas, OFDM symbols and
        # subcarriers.
        # c = tf.reduce_mean( tf.square(tf.abs(h_f)), axis=(2,4,5,6),
        #                     keepdims=True)
        c = np.mean(np.square( np.abs(h_f)), axis=(2,4,5,6),
                            keepdims=True)
        #c = tf.complex(tf.sqrt(c), tf.constant(0., real_dtype))
        c = np.sqrt(c) + 1j * 0.0 #(64, 1, 1, 1, 1, 1, 1)
        #h_f = tf.math.divide_no_nan(h_f, c)
        h_f = np.divide(h_f, c, out=h_f, where=~np.isnan(c))

    return h_f #(64, 1, 1, 1, 16, 1, 76)

def mygenerate_OFDMchannel(h, tau, fft_size, subcarrier_spacing=60000.0, dtype=np.complex64, normalize_channel=True):
    #h: [64, 1, 1, 1, 16, 10, 1], tau: [64, 1, 1, 10]
    #Generate OFDM channel
    # Frequencies of the subcarriers
    num_subcarriers = fft_size #resource_grid.fft_size #76
    subcarrier_spacing = subcarrier_spacing #resource_grid.subcarrier_spacing #60000
    frequencies = subcarrier_frequencies(num_subcarriers,
                                        subcarrier_spacing,
                                        dtype) #[76]
    h_freq = cir_to_ofdm_channel(frequencies, h, tau, normalize_channel)
    #Channel frequency responses at ``frequencies`` 
    return h_freq #[64, 1, 1, 1, 16, 1, 76] [batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_time_steps, fft_size]

class StreamManagement():
    """Class for management of streams in multi-cell MIMO networks.

    Parameters
    ----------
    rx_tx_association : [num_rx, num_tx], np.int
        A binary NumPy array where ``rx_tx_association[i,j]=1`` means
        that receiver `i` gets one or multiple streams from
        transmitter `j`.

    num_streams_per_tx : int
        Indicates the number of streams that are transmitted by each
        transmitter.

    Note
    ----
    Several symmetry constraints on ``rx_tx_association`` are imposed
    to ensure efficient processing. All row sums and all column sums
    must be equal, i.e., all receivers have the same number of associated
    transmitters and all transmitters have the same number of associated
    receivers. It is also assumed that all transmitters send the same
    number of streams ``num_streams_per_tx``.
    """
    def __init__(self,
                 rx_tx_association,
                 num_streams_per_tx):

        super().__init__()
        self._num_streams_per_tx = int(num_streams_per_tx) #1
        self.rx_tx_association = rx_tx_association #(1,1)

    @property
    def rx_tx_association(self):
        """Association between receivers and transmitters.

        A binary NumPy array of shape `[num_rx, num_tx]`,
        where ``rx_tx_association[i,j]=1`` means that receiver `i`
        gets one ore multiple streams from transmitter `j`.
        """
        return self._rx_tx_association

    @property
    def num_rx(self):
        "Number of receivers."
        return self._num_rx

    @property
    def num_tx(self):
        "Number of transmitters."
        return self._num_tx

    @property
    def num_streams_per_tx(self):
        "Number of streams per transmitter."
        return self._num_streams_per_tx

    @property
    def num_streams_per_rx(self):
        "Number of streams transmitted to each receiver."
        return int(self.num_tx*self.num_streams_per_tx/self.num_rx)

    @property
    def num_interfering_streams_per_rx(self):
        "Number of interfering streams received at each eceiver."
        return int(self.num_tx*self.num_streams_per_tx
                   - self.num_streams_per_rx)

    @property
    def num_tx_per_rx(self):
        "Number of transmitters communicating with a receiver."
        return self._num_tx_per_rx

    @property
    def num_rx_per_tx(self):
        "Number of receivers communicating with a transmitter."
        return self._num_rx_per_tx

    @property
    def precoding_ind(self):
        """Indices needed to gather channels for precoding.

        A NumPy array of shape `[num_tx, num_rx_per_tx]`,
        where ``precoding_ind[i,:]`` contains the indices of the
        receivers to which transmitter `i` is sending streams.
        """
        return self._precoding_ind

    @property
    def stream_association(self):
        """Association between receivers, transmitters, and streams.

        A binary NumPy array of shape
        `[num_rx, num_tx, num_streams_per_tx]`, where
        ``stream_association[i,j,k]=1`` means that receiver `i` gets
        the `k` th stream from transmitter `j`.
        """
        return self._stream_association

    @property
    def detection_desired_ind(self):
        """Indices needed to gather desired channels for receive processing.

        A NumPy array of shape `[num_rx*num_streams_per_rx]` that
        can be used to gather desired channels from the flattened
        channel tensor of shape
        `[...,num_rx, num_tx, num_streams_per_tx,...]`.
        The result of the gather operation can be reshaped to
        `[...,num_rx, num_streams_per_rx,...]`.
        """
        return self._detection_desired_ind

    @property
    def detection_undesired_ind(self):
        """Indices needed to gather undesired channels for receive processing.

        A NumPy array of shape `[num_rx*num_streams_per_rx]` that
        can be used to gather undesired channels from the flattened
        channel tensor of shape `[...,num_rx, num_tx, num_streams_per_tx,...]`.
        The result of the gather operation can be reshaped to
        `[...,num_rx, num_interfering_streams_per_rx,...]`.
        """
        return self._detection_undesired_ind

    @property
    def tx_stream_ids(self):
        """Mapping of streams to transmitters.

        A NumPy array of shape `[num_tx, num_streams_per_tx]`.
        Streams are numbered from 0,1,... and assiged to transmitters in
        increasing order, i.e., transmitter 0 gets the first
        `num_streams_per_tx` and so on.
        """
        return self._tx_stream_ids

    @property
    def rx_stream_ids(self):
        """Mapping of streams to receivers.

        A Numpy array of shape `[num_rx, num_streams_per_rx]`.
        This array is obtained from ``tx_stream_ids`` together with
        the ``rx_tx_association``. ``rx_stream_ids[i,:]`` contains
        the indices of streams that are supposed to be decoded by receiver `i`.
        """
        return self._rx_stream_ids

    @property
    def stream_ind(self):
        """Indices needed to gather received streams in the correct order.

        A NumPy array of shape `[num_rx*num_streams_per_rx]` that can be
        used to gather streams from the flattened tensor of received streams
        of shape `[...,num_rx, num_streams_per_rx,...]`. The result of the
        gather operation is then reshaped to
        `[...,num_tx, num_streams_per_tx,...]`.
        """
        return self._stream_ind

    @rx_tx_association.setter
    def rx_tx_association(self, rx_tx_association):
        """Sets the rx_tx_association and derives related properties. """

        # Make sure that rx_tx_association is a binary NumPy array
        rx_tx_association = np.array(rx_tx_association, np.int32)
        assert all(x in [0,1] for x in np.nditer(rx_tx_association)), \
            "All elements of `stream_association` must be 0 or 1."

        # Obtain num_rx, num_tx from stream_association shape
        self._num_rx, self._num_tx = np.shape(rx_tx_association)

        # Each receiver must be associated with the same number of transmitters
        num_tx_per_rx = np.sum(rx_tx_association, 1)
        assert np.min(num_tx_per_rx) == np.max(num_tx_per_rx), \
            """Each receiver needs to be associated with the same number
               of transmitters."""
        self._num_tx_per_rx = num_tx_per_rx[0]

        # Each transmitter must be associated with the same number of receivers
        num_rx_per_tx = np.sum(rx_tx_association, 0)
        assert np.min(num_rx_per_tx) == np.max(num_rx_per_tx), \
            """Each transmitter needs to be associated with the same number
               of receivers."""
        self._num_rx_per_tx = num_rx_per_tx[0]

        self._rx_tx_association = rx_tx_association

        # Compute indices for precoding
        self._precoding_ind = np.zeros([self.num_tx, self.num_rx_per_tx],
                                        np.int32)
        for i in range(self.num_tx):
            self._precoding_ind[i,:] = np.where(self.rx_tx_association[:,i])[0]

        # Construct the stream association matrix
        # The element [i,j,k]=1 indicates that receiver i, get the kth stream
        # from transmitter j.
        stream_association = np.zeros(
            [self.num_rx, self.num_tx, self.num_streams_per_tx], np.int32)
        n_streams = np.min([self.num_streams_per_rx, self.num_streams_per_tx])
        tmp = np.ones([n_streams])
        for j in range(self.num_tx):
            c = 0
            for i in range(self.num_rx):
                # If receiver i gets anything from transmitter j
                if rx_tx_association[i,j]:
                    stream_association[i,j,c:c+self.num_streams_per_rx] = tmp
                    c += self.num_streams_per_rx
        self._stream_association = stream_association

        # Get indices of desired and undesired channel coefficients from
        # the flattened stream_association. These indices can be used by
        # a receiver to gather channels of desired and undesired streams.
        self._detection_desired_ind = \
                 np.where(np.reshape(stream_association, [-1])==1)[0]

        self._detection_undesired_ind = \
                 np.where(np.reshape(stream_association, [-1])==0)[0]

        # We number streams from 0,1,... and assign them to the TX
        # TX 0 gets the first num_streams_per_tx and so on:
        self._tx_stream_ids = np.reshape(
                    np.arange(0, self.num_tx*self.num_streams_per_tx),
                    [self.num_tx, self.num_streams_per_tx])

        # We now compute the stream_ids for each receiver
        self._rx_stream_ids = np.zeros([self.num_rx, self.num_streams_per_rx],
                                        np.int32)
        for i in range(self.num_rx):
            c = []
            for j in range(self.num_tx):
                # If receiver i gets anything from transmitter j
                if rx_tx_association[i,j]:
                    tmp = np.where(stream_association[i,j])[0]
                    tmp += j*self.num_streams_per_tx
                    c += list(tmp)
            self._rx_stream_ids[i,:] = c

        # Get indices to bring received streams back to the right order in
        # which they were transmitted.
        self._stream_ind = np.argsort(np.reshape(self._rx_stream_ids, [-1]))

def flatten_last_dims(tensor, num_dims=2):
    """
    Flattens the last `n` dimensions of a tensor.

    This operation flattens the last ``num_dims`` dimensions of a ``tensor``.
    It is a simplified version of the function ``flatten_dims``.

    Args:
        tensor : A tensor.
        num_dims (int): The number of dimensions
            to combine. Must be greater than or equal to two and less or equal
            than the rank of ``tensor``.

    Returns:
        A tensor of the same type as ``tensor`` with ``num_dims``-1 lesser
        dimensions, but the same number of elements.
    """

    if num_dims==len(np.shape(tensor)): #len(tensor.shape):
        new_shape = [-1]
    else:
        shape = np.shape(tensor) #tf.shape(tensor)
        flipshape = shape[-num_dims:]
        last_dim = np.prod(flipshape)
        #last_dim = tf.reduce_prod(tensor.shape[-num_dims:])
        #new_shape = tf.concat([shape[:-num_dims], [last_dim]], 0)
        new_shape = np.concatenate([shape[:-num_dims], [last_dim]], 0)

    return np.reshape(tensor, new_shape) #tf.reshape(tensor, new_shape)

##Scatters updates into a tensor of shape shape according to indices
def scatter_nd_numpy(indices, updates, shape):
    target = np.zeros(shape, dtype=updates.dtype)
    indices = tuple(indices.reshape(-1, indices.shape[-1]).T)
    updates = updates.ravel()
    np.add.at(target, indices, updates)
    return target

def tensor_scatter_nd_update(tensor, indices, updates):
    """
    Updates the `tensor` by scattering `updates` into it at the specified `indices`.

    :param tensor: Existing tensor to update
    :param indices: Array of indices where updates should be placed
    :param updates: Array of values to scatter
    :return: Updated tensor
    """
    # Create a tuple of indices for advanced indexing
    index_tuple = tuple(indices.T) #(1064, 4) = > 4 tuple array (1064,) each, same to np.where output, means all places

    # Scatter values from updates into tensor
    tensornew = tensor.copy() #(1, 1, 14, 76, 64)
    tensornew[index_tuple] = updates #updates(1064,64) to all 1064 locations
    #(1, 1, 14, 76, 64) updates(1064, 64)

    #print the last dimension data of tensornew
    #print(tensornew[0,0,0,0,:]) #(64,)
    #print(updates[0,:]) #(64,)
    return tensornew #(1, 1, 14, 76, 64)

def scatter_numpy(tensor, indices, values):
    """
    Scatters values into a tensor at specified indices.
    
    :param tensor: The target tensor to scatter values into.
    :param indices: Indices where values should be placed.
    :param values: Values to scatter.
    :return: Updated tensor after scattering.
    """
    # Scatter values
    tensor[tuple(indices.T)] = values

    return tensor

class MyResourceGridMapper:
    r"""ResourceGridMapper(resource_grid, dtype=complex64, **kwargs)

    Maps a tensor of modulated data symbols to a ResourceGrid.

    Takes as input a tensor of modulated data symbols
    and maps them together with pilot symbols onto an
    OFDM `ResourceGrid`. 

    Parameters
    ----------
    resource_grid : ResourceGrid
    dtype

    x = mapper(b) #[64,1,1,912] 912 symbols
    x_rg = rg_mapper(x) ##[64,1,1,14,76] 14*76=1064

    Input
    -----
    : [batch_size, num_tx, num_streams_per_tx, num_data_symbols], complex
        The modulated data symbols to be mapped onto the resource grid.

    Output
    ------
    : [batch_size, num_tx, num_streams_per_tx, num_ofdm_symbols=14, fft_size=76], complex
        The full OFDM resource grid in the frequency domain.
    """
    def __init__(self, resource_grid, dtype=np.complex64, **kwargs):
        self._resource_grid = resource_grid
        """Precompute a tensor of shape
        [num_tx, num_streams_per_tx, num_ofdm_symbols, fft_size]
        which is prefilled with pilots and stores indices
        to scatter data symbols.
        """
        self._rg_type = self._resource_grid.build_type_grid() #(1, 1, 14, 76)
        #[num_tx, num_streams_per_tx, num_ofdm_symbols, fft_size]
        #        - 0 : Data symbol
        # - 1 : Pilot symbol
        # - 2 : Guard carrier symbol
        # - 3 : DC carrier symbol

        #Return the indices of non-zero elements in _rg_type via pytorch
        tupleindex = np.where(self._rg_type==1)#result is a tuple with first all the row indices, then all the column indices.
        self._pilot_ind=np.stack(tupleindex, axis=1) #shape=(0,4)
        #_rg_type array(1, 1, 14, 76)
        datatupleindex = np.where(self._rg_type==0) 
        #0 (all 0),1(all 0),2 (0-13),3 (0-75) tuple, (1064,) each, index for each dimension of _rg_type(1, 1, 14, 76)
        self._data_ind=np.stack(datatupleindex, axis=1) #(1064, 4)
        #self._pilot_ind = tf.where(self._rg_type==1) #shape=(0, 4)
        #self._data_ind = tf.where(self._rg_type==0) #[1064, 4]

        #test
        # test=self._rg_type.copy() #(1, 1, 14, 76)
        # data_test=test[datatupleindex]  #(1064,) 1064=14*76
        # print(data_test.shape)
        # pilot_test=test[tupleindex] #empty
        # print(pilot_test.shape)

    def __call__(self, inputs):#inputs: (64, 1, 1, 1064)
        #inputs: [batch_size, num_tx, num_streams_per_tx, num_data_symbols]
        # Map pilots on empty resource grid
        pilots = flatten_last_dims(self._resource_grid.pilot_pattern.pilots, 3) #empty

        #the indices tensor is the _pilot_ind tensor, which is a2D tensor that contains the indices of the pilot symbols in the resource grid. 
        #The values tensor is the pilots tensor, which is a1D tensor that contains the pilot symbols. 
        #The shape tensor is the _rg_type.shape tensor, which is a1D tensor that specifies the shape of the resource grid.
        ##Scatters pilots into a tensor of shape _rg_type.shape according to _pilot_ind
        # template = tf.scatter_nd(self._pilot_ind,
        #                          pilots,
        #                          self._rg_type.shape)
        template = scatter_nd_numpy(self._pilot_ind, pilots, self._rg_type.shape) #(1, 1, 14, 76) all 0 complex?

        # Expand the template to batch_size)
        # expand the last dimension for template via numpy
        template = np.expand_dims(template, axis=-1) ##[1, 1, 14, 76, 1]
        #template = tf.expand_dims(template, -1) #[1, 1, 14, 76, 1]

        # Broadcast the resource grid template to batch_size
        #batch_size = tf.shape(inputs)[0]
        batch_size = np.shape(inputs)[0]
        shapelist=list(np.shape(template)) ##[1, 1, 14, 76, 1]
        new_shape = np.concatenate([shapelist[:-1], [batch_size]], 0) #shape 5: array([ 1,  1, 14, 76, 64]
        #new_shape = tf.concat([tf.shape(template)[:-1], [batch_size]], 0) #shape 5: array([ 1,  1, 14, 76, 64]
        #template = tf.broadcast_to(template, new_shape)
        template = np.broadcast_to(template, new_shape) #(1, 1, 14, 76, 64)

        # Flatten the inputs and put batch_dim last for scatter update
        newflatten=flatten_last_dims(inputs, 3) #inputs:(64, 1, 1, 1064) =>(64, 1064)
        inputs = np.transpose(newflatten) #(1064, 64)
        #inputs = tf.transpose(newflatten)
        #The tf.tensor_scatter_nd_update function is a more efficient version of the scatter_nd function. 
        #update the resource grid with the data symbols. The input tensor is the resource grid, the values tensor is the data symbols, 
        #and the shape tensor is the _rg_type.shape tensor. The output tensor is the resource grid with the data symbols scattered in.
        #Scatter inputs into an existing template tensor according to _data_ind indices 
        #rg = tf.tensor_scatter_nd_update(template, self._data_ind, inputs)

        #Scatter inputs(1064, 64) into an existing template tensor(1, 1, 14, 76, 64) according to _data_ind indices (tuple from rg_type(1, 1, 14, 76)) 
        rg = tensor_scatter_nd_update(template, self._data_ind, inputs)
        #rg = scatter_nd_numpy(template, self._data_ind, inputs) #(1, 1, 14, 76, 64), (1064, 4), (1064, 64)
        #rg = tf.transpose(rg, [4, 0, 1, 2, 3])
        rg = np.transpose(rg, [4, 0, 1, 2, 3]) #(64, 1, 1, 14, 76)

        return rg
    
def hard_decisions(llr, datatype=np.int32):
    """Transforms LLRs into hard decisions.

    Positive values are mapped to :math:`1`.
    Nonpositive values are mapped to :math:`0`.

    Input
    -----
    llr : any non-complex of LLRs.

    Output
    ------
    : Same shape and dtype as ``llr``
        The hard decisions.
    """
    #zero = tf.constant(0, dtype=llr.dtype)

    #return tf.cast(tf.math.greater(llr, zero), dtype=llr.dtype)
    return np.greater(llr, 0.0).astype(datatype)
    #return tf.cast(tf.math.greater(llr, zero), dtype=datatype)

#Apply single-tap channel frequency responses to channel inputs.
#channel_freq = ApplyOFDMChannel(add_awgn=True)

def ApplyOFDMChannel(symbol_resourcegrid, channel_frequency, noiselevel=0, add_awgn=True):
    r"""
    For each OFDM symbol :math:`s` and subcarrier :math:`n`, the single-tap channel
        is applied as follows:

        .. math::
            y_{s,n} = \widehat{h}_{s, n} x_{s,n} + w_{s,n}

        where :math:`y_{s,n}` is the channel output computed by this layer,
        :math:`\widehat{h}_{s, n}` the frequency channel response (``h_freq``),
        :math:`x_{s,n}` the channel input ``x``, and :math:`w_{s,n}` the additive noise.

        For multiple-input multiple-output (MIMO) links, the channel output is computed for each antenna of each receiver and 
        by summing over all the antennas of all transmitters.
    """
    #input: (x, h_freq, no) or (x, h_freq):
    #inputs x :  [batch size, num_tx, num_tx_ant, num_ofdm_symbols, fft_size], complex
    #h_freq : [batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_ofdm_symbols, num_subcarriers], complex Channel frequency responses
    #h_freq: (64, 1, 1, 1, 16, 1, 76)
    x_rg = symbol_resourcegrid
    h_freq = channel_frequency
    no = noiselevel

    #output:
    #y = channel_freq([x_rg, h_freq, no]) #h_freq is array
    #Channel outputs y : [batch size, num_rx, num_rx_ant, num_ofdm_symbols, fft_size], complex    
    #print(y.shape) #[64, 1, 1, 14, 76] dim (3,4 removed)

    # Apply the channel response .ndim
    x = myexpand_to_rank(x_rg, h_freq.ndim, axis=1) #array[64, 1(added), 1(added), 1, 1, 14, 76]
    #y = tf.reduce_sum(tf.reduce_sum(h_freq*x, axis=4), axis=3) #[64, 1, 1, 14, 76] (16 removed)
    hx = np.sum(h_freq*x, axis=4) #array(64, 1, 1, 1, 14, 76)
    y = np.sum(hx, axis=3) #array(64, 1, 1, 14, 76) dim (3,4 removed)
    #[batch_size, num_rx, num_rx_ant, num_ofdm_symbols,fft_size]

    if add_awgn:
        noise=complex_normal(y.shape, var=1.0)
        print(noise.dtype)
        noise = noise.astype(y.dtype)
        noise *= np.sqrt(no)
        y=y+noise
    
    return y ##[batch_size, num_rx, num_rx_ant, num_ofdm_symbols,fft_size] (64, 1, 1, 14, 76)
        
def calculate_BER(bits, bits_est):
    errors = (bits != bits_est).sum()
    N = len(bits.flatten())
    BER = 1.0 * errors / N
    # error_count = torch.sum(bits_est != bits.flatten()).float()  # Count of unequal bits
    # error_rate = error_count / bits_est.numel()  # Error rate calculation
    # BER = torch.round(error_rate * 1000) / 1000  # Round to 3 decimal places
    return BER

if __name__ == '__main__':

    DeepMIMO_dataset = get_deepMIMOdata()
    scenario='O1_60'
    dataset_folder=r'D:\Dataset\CommunicationDataset\O1_60'
    #DeepMIMO provides multiple [scenarios](https://deepmimo.net/scenarios/) that one can select from. 
    #In this example, we use the O1 scenario with the carrier frequency set to 60 GHz (O1_60). 
    #Please download the "O1_60" data files [from this page](https://deepmimo.net/scenarios/o1-scenario/).
    #The downloaded zip file should be extracted into a folder, and the parameter `'dataset_folder` should be set to point to this folder
    DeepMIMO_dataset = get_deepMIMOdata(scenario=scenario, dataset_folder=dataset_folder)

    # Number of receivers for the model.
    # MISO is considered here.
    num_rx = 1
    num_tx = 1 #new add
    batch_size =64
    fft_size = 76

    # The number of UE locations in the generated DeepMIMO dataset
    num_ue_locations = len(DeepMIMO_dataset[0]['user']['channel']) # 9231
    # Pick the largest possible number of user locations that is a multiple of ``num_rx``
    ue_idx = np.arange(num_rx*(num_ue_locations//num_rx)) #(9231,) 0~9230
    # Optionally shuffle the dataset to not select only users that are near each others
    np.random.shuffle(ue_idx)
    # Reshape to fit the requested number of users
    ue_idx = np.reshape(ue_idx, [-1, num_rx]) # In the shape of (floor(9231/num_rx) x num_rx) (9231,1)

    testdataset = DeepMIMODataset(DeepMIMO_dataset=DeepMIMO_dataset, ue_idx=ue_idx)
    h, tau = next(iter(testdataset)) #h: (1, 1, 1, 16, 10, 1), tau:(1, 1, 10)
    #complex gains `h` and delays `tau` for each path
    #print(h.shape) #[num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, num_time_steps]
    #print(tau.shape) #[num_rx, num_tx, num_paths]

    # torch dataloaders
    data_loader = DataLoader(dataset=testdataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    h_b, tau_b = next(iter(data_loader)) #h_b: [64, 1, 1, 1, 16, 10, 1], tau_b=[64, 1, 1, 10]
    #print(h_b.shape) #[batch, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, num_time_steps]
    #print(tau_b.shape) #[batch, num_rx, num_tx, num_paths]

    tau_b=tau_b.numpy()#torch tensor to numpy
    h_b=h_b.numpy()
    plt.figure()
    plt.title("Channel impulse response realization")
    plt.stem(tau_b[0,0,0,:]/1e-9, np.abs(h_b)[0,0,0,0,0,:,0])#10 different pathes
    plt.xlabel(r"$\tau$ [ns]")
    plt.ylabel(r"$|a|$")

    # Generate the OFDM channel response
    ##h: [64, 1, 1, 1, 16, 10, 1], tau: [64, 1, 1, 10] => (64, 1, 1, 1, 16, 1, 76) 
    h_freq = mygenerate_OFDMchannel(h_b, tau_b, fft_size, subcarrier_spacing=60000.0, dtype=np.complex64, normalize_channel=True)
    #h_freq : [batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_time_steps, fft_size]
    #(64, 1, 1, 1, 16, 1, 76)

    h_freq_plt = h_freq[0,0,0,0,0,0] #get the last dimension: fft_size [76]
    plt.figure()
    plt.plot(np.real(h_freq_plt))
    plt.plot(np.imag(h_freq_plt))
    plt.xlabel("Subcarrier index")
    plt.ylabel("Channel frequency response")
    plt.legend(["Ideal (real part)", "Ideal (imaginary part)"]);
    plt.title("Comparison of channel frequency responses");
    
    num_streams_per_tx = num_rx ##1
    STREAM_MANAGEMENT = StreamManagement(np.ones([num_rx, 1], int), num_streams_per_tx) #RX_TX_ASSOCIATION, NUM_STREAMS_PER_TX

    cyclic_prefix_length = 0 #6
    num_guard_carriers = [0, 0]
    dc_null=False
    pilot_ofdm_symbol_indices=[2,11]
    pilot_pattern = "empty" #"kronecker" #"kronecker", "empty"
    #fft_size = 76
    num_ofdm_symbols=14
    RESOURCE_GRID = MyResourceGrid( num_ofdm_symbols=num_ofdm_symbols,
                                        fft_size=fft_size,
                                        subcarrier_spacing=60e3, #30e3,
                                        num_tx=num_tx, #1
                                        num_streams_per_tx=num_streams_per_tx, #1
                                        cyclic_prefix_length=cyclic_prefix_length,
                                        num_guard_carriers=num_guard_carriers,
                                        dc_null=dc_null,
                                        pilot_pattern=pilot_pattern,
                                        pilot_ofdm_symbol_indices=pilot_ofdm_symbol_indices)
    RESOURCE_GRID.show() #14(OFDM symbol)*76(subcarrier) array=1064

    num_bits_per_symbol = 4
    coderate = 1 #0.5
    # Codeword length
    n = int(RESOURCE_GRID.num_data_symbols * num_bits_per_symbol) #912*4=3648, if empty 1064*4=4256
    # Number of information bits per codeword
    k = int(n * coderate)        

    # Transmitter
    binary_source = BinarySource()

    mapper = Mapper("qam", num_bits_per_symbol)
    rg_mapper = MyResourceGridMapper(RESOURCE_GRID) #ResourceGridMapper(RESOURCE_GRID)

    # Start Transmitter 
    b = binary_source([batch_size, 1, num_streams_per_tx, k]) #[64,1,1,2128] [batch_size, num_tx, num_streams_per_tx, num_databits]

    x = mapper(b) #array[64,1,1,1064] 1064*4=4256 [batch_size, num_tx, num_streams_per_tx, num_data_symbols]
    x_rg = rg_mapper(x) ##array[64,1,1,14,76] 14*76=1064 no pilot
    #output: [batch_size, num_tx, num_streams_per_tx, num_ofdm_symbols, fft_size]

    #set noise level
    ebnorange=np.linspace(-7, -5.25, 10)
    ebno_db = 15.0
    #no = ebnodb2no(ebno_db, num_bits_per_symbol, coderate, RESOURCE_GRID)
    no = ebnodb2no(ebno_db, num_bits_per_symbol, coderate)


    # Generate the OFDM channel
    #h_freq : [batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_time_steps, fft_size]
    #(64, 1, 1, 1, 16, 1, 76)
    y = ApplyOFDMChannel(symbol_resourcegrid=x_rg, channel_frequency=h_freq, noiselevel=no, add_awgn=True)
    # y is the symbol received after the channel and noise
    #Channel outputs y : [batch size, num_rx, num_rx_ant, num_ofdm_symbols, fft_size], complex    
    print(y.shape) #[64, 1, 1, 14, 76] dim (3,4 removed)

    