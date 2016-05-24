defmodule NeuralNetwork.Perceptron do
  @moduledoc """
    This module defines an artifical neuron: a perceptron.
    A way you can think about the perceptron is that it's a device that makes decisions by weighing up evidence.
  """

  defstruct [:size, :weights, :bias]

  @type size :: pos_integer
  @type inputs :: list(float)
  @type weights :: list(float)
  @type bias :: float
  @type output :: float
  @type t :: %NeuralNetwork.Perceptron{
    size: size,
    weights: list(float),
    bias: bias
  }

  @spec new(size, weights, bias) :: t
  def new(size, weights, bias)
  when length(weights) == size do
    %NeuralNetwork.Perceptron{
      size: size,
      weights: weights,
      bias: bias
    }
  end

  @spec output(t, inputs) :: output
  def output(%NeuralNetwork.Perceptron{size: size, weights: weights, bias: bias}, inputs)
  when length(inputs) == size do
    sum = 0..(size-1)
      |> Enum.map(fn i -> Enum.at(weights, i) * Enum.at(inputs, i) end)
      |> Enum.sum
    sum + bias
  end
end
