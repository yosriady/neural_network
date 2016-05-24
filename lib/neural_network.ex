defmodule NeuralNetwork do
  @moduledoc """
    This module implements a feed-forward artifical neural network with backpropagation.
  """

  defstruct [:layers, :iterations, :error_treshold, :learning_rate]

  @type t :: %NeuralNetwork{
    layers: list(),
    iterations: pos_integer, # maximum training iterations
    error_treshold: float, # error threshold to reach
    learning_rate: float # learning rate
  }

  @doc """
    Creates a new neural network.
    Accepts arguments such as error treshold, iterations, and learning rate.
  """
  def new(layers) do
    # TODO
    %NeuralNetwork{

    }
  end

  @doc """
    Trains a neural network.

    The network will train until the training error has gone below the threshold or
    the max number of iterations (default 20000) has been reached, whichever comes first.
  """
  def train(data, [iterations: iterations,
                   error_treshold: error_treshold,
                   learning_rate: learning_rate]) do
    # TODO
  end

  @doc """
    Traverse the network for a given input and return a result.
  """
  def run([]) do
    # TODO
  end

end
