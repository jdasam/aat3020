import torch
import torch.nn as nn


class MyLSTM(nn.Module):
  def __init__(self, input_size: int, hidden_size: int):
    super().__init__()
    '''
    TODO: Define the weights and biases for a single-layer, uni-directional LSTM.

    Arguments:
      input_size  (int): Dimensionality of input x. Denoted as `d` in the equations.
      hidden_size (int): Dimensionality of hidden state h. Denoted as `h` in the equations.

    Module definitions:
      self.weight_ih (nn.Linear): Combines [W_i | W_f | W_c | W_o] with bias.
                                   in_features=input_size, out_features=4*hidden_size
      self.weight_hh (nn.Linear): Combines [U_i | U_f | U_c | U_o] with bias.
                                   in_features=hidden_size, out_features=4*hidden_size

    Note: Weight order must follow PyTorch's official LSTM convention: [i | f | g | o].
    (LSTMLanguageModel in Problem 2 and gate visualization assume this order.)
    '''

    self.weight_ih = nn.Linear(in_features=, out_features=, bias=True) # TODO: complete this layer by selecting proper in_features and out_features
    self.weight_hh = nn.Linear(in_features=, out_features=, bias=True) # TODO: complete this layer by selecting proper in_features and out_features
    self.hidden_size = hidden_size

  def _cal_single_step(self, x_t:torch.Tensor, last_hidden:torch.Tensor, last_cell:torch.Tensor):
    '''
    Args:
      x_t         : input at timestep t. Shape: [Num_Batch, Num_input_dim]
      last_hidden : hidden state at timestep (t-1). Shape: [Num_Batch, Num_hidden_dim]
      last_cell   : cell state at timestep (t-1). Shape: [Num_Batch, Num_hidden_dim]

    Returns:
      updated_hidden (torch.Tensor): hidden state at timestep t. Shape: [Num_Batch, Num_hidden_dim]
      updated_cell   (torch.Tensor): cell state at timestep t. Shape: [Num_Batch, Num_hidden_dim]

    TODO: Implement the LSTM equations using self.weight_ih and self.weight_hh.
    '''
    assert x_t.ndim == last_hidden.ndim == last_cell.ndim == 2, "_cal_single_step only takes 2-dim Tensors"
    assert x_t.shape[0] == last_hidden.shape[0] == last_cell.shape[0], "batch size must be the same"

    # Write your code from here

    return updated_hidden, updated_cell

  def forward(self, x:torch.Tensor, hidden_and_cell_state:tuple=None):
    '''
    Args:
      x (torch.Tensor): Input sequence. Shape: [Num_Batch, Num_Timestep, Num_input_dim]
      hidden_and_cell_state (optional, tuple): Hidden and cell state from the previous timestep.

    Returns:
      output, (last_hidden, last_cell)
      Note: the second return value must be a tuple of two tensors.

      output (torch.Tensor): Concatenated hidden states for all timesteps.
                             Shape: [Batch_Size, Num_Time_Steps, Hidden_State_Size]
      last_hidden (torch.Tensor): The hidden state after processing all timesteps.
      last_cell   (torch.Tensor): The cell state after processing all timesteps.

    TODO: Implement using a for loop and `self._cal_single_step`.
    '''

    # Leave the code below as it is
    if hidden_and_cell_state is not None and isinstance(hidden_and_cell_state, tuple):
      last_hidden = hidden_and_cell_state[0]
      last_cell = hidden_and_cell_state[1]
    else:
      last_hidden = torch.zeros([x.shape[0], self.hidden_size], device=x.device)
      last_cell = torch.zeros([x.shape[0], self.hidden_size], device=x.device)

    '''
    Write your code from here
    '''

    return output, (last_hidden, last_cell)



def main():
  input_size = 16
  hidden_size = 32

  model = MyLSTM(input_size, hidden_size)

  dummy_batch_size = 8
  dummy_time_steps = 20
  dummy_input = torch.randn([dummy_batch_size, dummy_time_steps, input_size])

  output, (last_hidden_state, last_cell_state) = model(dummy_input)

  assert output.shape[0] == dummy_batch_size, "0th dimension of output must be the batch size"
  assert output.shape[1] == dummy_time_steps, "1st dimension of output must be the number of time steps"
  assert output.shape[2] == hidden_size, "2nd dimension of output must be the hidden_size"

  total_output, (last_hidden_state, last_cell_state) = model(dummy_input)

  hidden_and_cell_state = (torch.zeros([dummy_batch_size, hidden_size]), torch.zeros([dummy_batch_size, hidden_size]))
  for i in range(dummy_time_steps):
    time_step_output, hidden_and_cell_state = model(dummy_input[:,i:i+1], hidden_and_cell_state)

  assert (total_output[:,-1:] == time_step_output).all(), 'LSTM output must match for sliced input using for-loop'

  lstm_pre_impl = nn.LSTM(input_size, hidden_size, batch_first=True)

  lstm_pre_impl.weight_hh_l0.data = model.weight_hh.weight.data
  lstm_pre_impl.bias_hh_l0.data = model.weight_hh.bias.data

  lstm_pre_impl.weight_ih_l0.data = model.weight_ih.weight.data
  lstm_pre_impl.bias_ih_l0.data = model.weight_ih.bias.data

  output, (last_hidden_state, last_cell_state) = model(dummy_input)
  output_compare, (last_hidden_state_compare, last_cell_state_compare) = lstm_pre_impl(dummy_input)

  assert torch.allclose(output, output_compare, atol=1e-6), "LSTM output does not match PyTorch's official implementation"
  assert torch.allclose(last_hidden_state, last_hidden_state_compare, atol=1e-6), "Last hidden state does not match"
  assert torch.allclose(last_cell_state, last_cell_state_compare, atol=1e-6), "Last cell state does not match"

  print("Test passed! Your LSTM implementation returns the exactly same result as PyTorch's official single-layer uni-directional LSTM.")


if __name__ == '__main__':
  main()
