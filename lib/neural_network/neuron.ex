defmodule NeuralNetwork.Neuron do
  @moduledoc """
    This module defines an artifical neuron: a perceptron with a sigmoid activation function.
    A way you can think about the neuron is that it's a device that makes decisions by weighing up evidence.
  """

  alias NeuralNetwork.Math

  defstruct [:size, :weights, :bias]

  @type size :: pos_integer
  @type inputs :: list(float)
  @type weights :: list(float)
  @type bias :: float
  @type output :: float
  @type t :: %NeuralNetwork.Neuron{
    size: size,
    weights: list(float),
    bias: bias
  }

  @spec new(size, weights, bias) :: t
  def new(size, weights, bias)
  when length(weights) == size do
    %NeuralNetwork.Neuron{
      size: size,
      weights: weights,
      bias: bias
    }
  end

  @spec output(t, inputs) :: output
  def output(%NeuralNetwork.Neuron{size: size, weights: weights, bias: bias}, inputs)
  when length(inputs) == size do
    sum = 0..(size-1)
      |> Enum.map(fn i -> Enum.at(weights, i) * Enum.at(inputs, i) end)
      |> Enum.sum
    Math.sigmoid(sum + bias)
  end
end
