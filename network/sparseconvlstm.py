import torch.nn as nn
import torch
import spconv


class SparseConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, spatial_shape, kernel_size, bias):
        """
        Initialize SparseConvLSTM cell.

        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.

        hidden_dim: int
            Number of channels of hidden state.

        spatial_shape: (int, int, int, int, int)
            Spatial shape of the sparse data

        kernel_size: (int, int)
            Size of the convolutional kernel.

        bias: bool
            Whether or not to add the bias.
        """

        super(SparseConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.spatial_shape = spatial_shape
        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2, kernel_size[2] // 2
        self.bias = bias

        self.sparse_conv = spconv.SubMConv3d(in_channels=self.input_dim + self.hidden_dim,
                                             out_channels=4 * self.hidden_dim,
                                             kernel_size=self.kernel_size,
                                             padding=self.padding,
                                             bias=self.bias,
                                             use_hash=False)
        print(self.sparse_conv.__dict__)
        '''
        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)
        '''

    def forward(self, input_tensor_sparse, cur_state, coors, batch_size):

        # Make sure the coordinates are integer
        coors = coors.int()
        print(coors.shape)

        print(input_tensor_sparse.features.shape)
        print(input_tensor_sparse.indices.shape)

        h_cur, c_cur = cur_state
        print(h_cur.features.shape)
        print(c_cur.features.shape)

        # Transform the input arguments into an individual Sparse Tensor
        # input_tensor_sparse = spconv.SparseConvTensor(input_tensor, coors, self.spatial_shape,
        #                                              batch_size)
        combined_sparse = input_tensor_sparse

        # concatenate along channel axis
        combined_sparse.features = torch.cat(
            [input_tensor_sparse.features, h_cur.features], dim=1)

        print(combined_sparse.features.shape)
        print(combined_sparse.indices.shape)
        # input_tensor_sparse.indices = torch.cat(
        #    [input_tensor_sparse.indices, h_cur.indices], dim=1)

        combined_sparse_conv = self.sparse_conv(combined_sparse)

        cc_i, cc_f, cc_o, cc_g = torch.split(
            combined_sparse_conv.features, self.hidden_dim, dim=1)

        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur.features + i * g
        h_next = o * torch.tanh(c_next)

        return spconv.SparseConvTensor(h_next, coors, self.spatial_shape, batch_size), spconv.SparseConvTensor(c_next, coors, self.spatial_shape, batch_size)

    def init_hidden(self, num_points, coords, spatial_shape, batch_size):
        coords = coords.int()
        return (spconv.SparseConvTensor(torch.zeros(num_points, self.hidden_dim, device=self.sparse_conv.weight.device), coords, spatial_shape, batch_size), )*2


class SparseConvLSTM(nn.Module):

    """

    Parameters:
        input_dim: Number of channels in input
        hidden_dim: Number of hidden channels
        spatial_Shape: Spatial shape of the sparse data
        kernel_size: Size of kernel in convolutions
        num_layers: Number of LSTM layers stacked on each other
        batch_first: Whether or not dimension 0 is the batch or not
        bias: Bias or no bias in Convolution
        return_all_layers: Return the list of computations for all layers
        Note: Will do same padding.

    Input:
        A tensor of size B, T, C, H, W or T, B, C, H, W
    Output:
        A tuple of two lists of length num_layers (or length 1 if return_all_layers is False).
            0 - layer_output_list is the list of lists of length T of each output
            1 - last_state_list is the list of last states
                    each element of the list is a tuple (h, c) for hidden state and memory
    Example:
        >> x = torch.rand((32, 10, 64, 128, 128))
        >> SparseConvLSTM = SparseConvLSTM(64, 16, 3, 1, True, True, False)
        >> _, last_states = SparseConvLSTM(x)
        >> h = last_states[0][0]  # 0 for layer index, 0 for h index
    """

    def __init__(self, input_dim, hidden_dim, spatial_shape, kernel_size, num_layers,
                 batch_first=False, bias=True, return_all_layers=False):
        super(SparseConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.spatial_shape = spatial_shape
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []

        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(SparseConvLSTMCell(input_dim=cur_input_dim,
                                                hidden_dim=self.hidden_dim[i],
                                                spatial_shape=self.spatial_shape,
                                                kernel_size=self.kernel_size[i],
                                                bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor,  coords, batch_size, hidden_state=None):
        """

        Parameters
        ----------
        input_tensor: todo
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful

        Returns
        -------
        last_state_list, layer_output
        """

        '''
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)
        '''

        #b, _, num_points, num_features = input_tensor.size()
        coords = coords.int()
        num_points, num_features = input_tensor[0].features.size()

        # Implement stateful SparseConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            # Since the init is done in forward. Can send image size here
            hidden_state = self._init_hidden(
                num_points, input_tensor[0].indices, self.spatial_shape, 1)

        layer_output_list = []
        last_state_list = []

        #seq_len = input_tensor.size(batch_size)
        seq_len = len(input_tensor)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):

            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):

                if t != seq_len-1:

                    i = t
                    k = t+1

                else:

                    i = t
                    k = int((t+1)/2)

                coords[k] = coords[k].int()
                h, c = self.cell_list[layer_idx](
                    cur_layer_input[i], (h, c), coords[k], 1)
                output_inner.append(h)

            #layer_output = torch.stack(output_inner, dim=1)
            layer_output = output_inner
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, num_points, coords, spatial_shape, batch_size):
        init_states = []
        coords = coords.int()

        for i in range(self.num_layers):
            init_states.append(
                self.cell_list[i].init_hidden(num_points, coords, spatial_shape, batch_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param
