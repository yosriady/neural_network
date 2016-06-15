defmodule NeuralNetwork.Network do
  @moduledoc """
    This module implements a feed-forward artifical neural network with backpropagation.
  """

  alias NeuralNetwork.Num
  alias NeuralNetwork.Neuron

  defstruct [:sizes, :biases, :weights]

  @type sizes :: list(pos_integer)
  @type input :: float
  @type inputs :: list(input)
  @type output :: float
  @type outputs :: list(float)
  @type activations :: list(outputs)
  @type training_data :: list(tuple)
  @type test_data :: list(tuple)
  @type mini_batch :: list(tuple)
  @type epochs :: pos_integer
  @type mini_batch_size :: pos_integer
  @type eta :: pos_integer
  @type bias :: float
  @type weight :: float
  @type delta_nabla_biases :: list(list(bias))
  @type delta_nabla_weights :: list(list(weight))
  @type t :: %NeuralNetwork.Network{
    sizes: list(pos_integer), # size of layers i.e. [2,3,1]
    biases: list(list(bias)),
    weights: list(list(weight))
  }

  @doc """
    Creates a new neural network.
  """
  @spec new(sizes) :: t
  def new(sizes) do
    weighted_layers = Enum.drop(sizes, 1) # layers except the input layer
    layers_except_output = Enum.drop(sizes, -1)
    %NeuralNetwork.Network{
      sizes: sizes,
      # biases is a list of lists, one list per layer, where each list
      # stores the biases of nodes in that layer
      biases: (for n <- weighted_layers, do: Num.gauss(n)),
      # weights is a list of matrices, one matrix per layer
      # Given a network of layer sizes [2,3,1] and
      # neurons `A`, `B` in the first layer & neurons  `1`, `2`, `3` in the second
      # we have:
      #
      # Enum.at(weights, 0)
      # > [
      #   [A1, B1],
      #   [A2, B2],
      #   [A3, B3]
      # ]
      #
      # A1 is the incoming edge from the first layer neuron `A` to
      # the second layer neuron `1`
      weights: (for tuple <- Enum.zip(layers_except_output, weighted_layers), do: Num.randn(elem(tuple, 1), elem(tuple, 0)))
    }
  end

  @doc """
    Calculates the outputs of each neuron layer by layer,
    where inputs is a vector of size equal to the size of the (first) input layer

    Returns a list of vectors, from input to activation to output - one for each layer.
  """
  @spec feedforward(t, inputs) :: activations
  def feedforward(%NeuralNetwork.Network{sizes: sizes, biases: biases, weights: weights}, inputs)
  when length(inputs) == hd(sizes) do
    weighted_layers = Enum.drop(sizes, 1)
    activations = _feedforward(weights, biases, weighted_layers, [inputs], inputs)
    activations
  end

  # Feedforward algorithm Base case
  defp _feedforward([], [], [], activations, _outputs) do
     activations
  end

  # Feedforward algorithm Recursive case
  defp _feedforward([layer_weights | remaining_weights],
                    [layer_biases | remaining_biases],
                    [layer_size | remaining_layers],
                    activations,
                    inputs) do
    # Activate the current layer of neurons, returning a vector of outputs
    layer_outputs = 0..(layer_size - 1)
    |> Enum.map(fn(i) -> Neuron.new(Enum.at(layer_weights, i), Enum.at(layer_biases, i)) end)
    |> Enum.map(fn(n) -> Neuron.activate(n, inputs) end)

    # We recursively feedforward to the next layers until the final output layer
    _feedforward(remaining_weights,
                 remaining_biases,
                 remaining_layers,
                 activations ++ [layer_outputs], # keep past layer activations, from input to output (for backpropagation)
                 layer_outputs)
  end

  @doc """
    Trains the neural network using mini-batch stochastic gradient descent.

    training_data is a list of tuples {x, y} containing two lists:
    the training inputs x and
    the desired outputs y.

    If test_data is supplied, the network will be evaluated against the test data
    after each eopch, and partial progress printed out.

    Returns a new, trained network.
  """
  def sgd(network, _training_data, 0, _mini_batch_size, _eta, _test_data) do # Base case
    network
  end

  @spec sgd(t, training_data, epochs, mini_batch_size, eta, test_data) :: t
  def sgd(network, training_data, epochs, mini_batch_size, eta, test_data) do # Recursive case
    training_data = training_data
                    |> Enum.shuffle
    mini_batches = training_data
                    |> Enum.chunk(mini_batch_size)
    updated_network = Enum.reduce(mini_batches, network, fn(mini_batch, network) ->
        update_mini_batch(network, mini_batch, eta)
    end)

    sgd(updated_network, training_data, epochs - 1, mini_batch_size, eta, test_data)
  end

  @doc """
    Updates the network weights and biases by applying gradient descent using
    backpropagation to a single mini batch.

    mini_batch is a list of tuples {x, y} containing two lists:
    the training inputs x and
    the desired outputs y.

    eta is the learning rate

    Returns a new network with updated weights and biases.
  """
  @spec update_mini_batch(t, mini_batch, eta) :: t
  def update_mini_batch(%NeuralNetwork.Network{sizes: sizes, biases: biases, weights: weights} = network, mini_batch, eta) do
    weighted_layers = Enum.drop(sizes, 1) # sizes of layers except the input layer
    layers_except_output = Enum.drop(sizes, -1) # sizes of layers except the output layer
    nabla_biases = (for n <- weighted_layers, do: Num.zeros(n)) # Set all biases to 0
    nabla_weights = (for tuple <- Enum.zip(layers_except_output, weighted_layers), do: Num.zeros(elem(tuple, 1), elem(tuple, 0))) # Set all weights to 0
    Enum.reduce(mini_batch, {nabla_biases, nabla_weights}, fn({x, y}, {nb, nw}) ->
      {delta_nabla_biases, delta_nabla_weights} = backpropagate(network, x, y)

      sum_pair = (fn {h, t} -> h + t end)
      nabla_biases = Num.merge_matrices(nb, delta_nabla_biases, sum_pair)
      nabla_weights = Num.merge_lists_of_matrices(nw, delta_nabla_weights, sum_pair)
      {nabla_biases, nabla_weights}
    end)

    gradient_sum_pair = (fn {h, t} -> h - (eta / length(mini_batch)) * t end)
    new_biases = Num.merge_matrices(biases, nabla_biases, gradient_sum_pair)
    new_weights = Num.merge_lists_of_matrices(weights, nabla_weights, gradient_sum_pair)
    %NeuralNetwork.Network{network | weights: new_weights, biases: new_biases}
  end


  @doc """
    Returns a tuple of `delta_nabla_biases`, and `delta_nabla_weights` in the same shape
    as biases and weights. Deltas represent the gradient for the cost function,
    the network changes in reaction to observed distance from the desired target
    output.

    Input is a vector list of training inputs,
    Output is a vector list of desired outputs
  """
  @spec backpropagate(t, inputs, outputs) :: {delta_nabla_biases, delta_nabla_weights}
  def backpropagate(%NeuralNetwork.Network{sizes: sizes}, _input, _output) do
    weighted_layers = Enum.drop(sizes, 1) # sizes of layers except the input layer
    layers_except_output = Enum.drop(sizes, -1) # sizes of layers except the output layer
    nabla_biases = (for n <- weighted_layers, do: Num.zeros(n)) # Set all biases to 0
    nabla_weights = (for tuple <- Enum.zip(layers_except_output, weighted_layers), do: Num.zeros(elem(tuple, 1), elem(tuple, 0))) # Set all weights to 0

        #   TODO: chapter 2

    {nabla_biases, nabla_weights}
  end

  @doc """
    Trains a neural network.

    The network will train until the training error has gone below the threshold or
    the max number of iterations (default 20000) has been reached, whichever comes first.
  """
  def train() do
    # TODO
  end


  @doc """
    Return the number of test inputs for which the neural network outputs the
    correct result.
  """
  @spec evaluate(t, test_data) :: pos_integer
  def evaluate(network, test_data) do
    actual_outputs = test_data
        |> Enum.map(fn {test_inputs, _} -> feedforward(network, test_inputs) end)
        |> List.last # Get only the final output activation
    desired_outputs = test_data
        |> Enum.map(fn {_, test_outputs} -> test_outputs end) # Get test outputs
    matches = Enum.zip(actual_outputs, desired_outputs) # Compare actual vs desired outputs
      |> Enum.reduce(0, fn({actual, desired}, count) ->
          if actual == desired, do: count + 1, else: count
      end)
    matches
  end

  @doc """
    Traverse the network for a given input and return a result.
  """
  def run() do
    # TODO
  end
end
