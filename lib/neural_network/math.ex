defmodule NeuralNetwork.Math do
  @moduledoc """
    This module implements some math-related utility functions.
  """

  @doc """
    Sigmoid function.
  """
  @spec sigmoid(float) :: float
  def sigmoid(z) do
      1 / (1 + :math.pow(e, -z))
  end

  @doc """
    Derivative of the Sigmoid function.
  """
  @spec sigmoid_prime(float) :: float
  def sigmoid_prime(z) do
      sigmoid(z) * (1 - sigmoid(z))
  end

  @doc """
    The mathematical constant e.
  """
  @spec e :: float
  def e, do: 2.718281828459045

end
