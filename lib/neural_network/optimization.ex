defmodule NeuralNetwork.Optimization do
  @moduledoc """
    This module implements cost functions used in network cost minimization.
  """

  @type expected :: list(float)
  @type actual :: list(float)

  @doc """
    Cost function (mean squared error) used to optimize the weights and biases of a network.

    Expected and actual are a list of output vectors.
    Actual vectors are calculated using the weights, biases, and activation function.
  """
  @spec cost(expected, actual) :: float
  def cost(expected, actual)
  when length(expected) == length(actual) and length(expected) > 0 do
    n = length(expected)
    quadratic_sum = 0..(n-1)
      |> Enum.map(fn i -> Enum.at(expected, i) - Enum.at(actual, i) end)
      |> Enum.map(fn i -> :math.pow(i, 2) end)
      |> Enum.sum
    (1/(2*n)) * quadratic_sum
  end
end
