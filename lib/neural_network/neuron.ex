defmodule NeuralNetwork.Neuron do
  @moduledoc """
    This module defines an artifical neuron: a perceptron with a sigmoid activation function.
    A way you can think about the neuron is that it's a device that makes decisions by weighing up evidence.
  """

  alias NeuralNetwork.Math

  defstruct [:weights, :bias]

  @type inputs :: list(float)
  @type weights :: list(float)
  @type bias :: float
  @type activation :: float
  @type activation_function :: (float -> float)
  @type t :: %NeuralNetwork.Neuron{
    weights: list(float), # List of weights for each input e.g. [-2.0, 1.0, 0.5]
    bias: bias # Bias for the neuron e.g. 5.5
  }

  @spec new(weights, bias) :: t
  def new(weights, bias) do
    %NeuralNetwork.Neuron{
      weights: weights,
      bias: bias
    }
  end

  @spec activate(t, inputs, activation_function) :: activation
  def activate(%NeuralNetwork.Neuron{weights: weights, bias: bias},
            inputs, activation_function \\ &Math.sigmoid/1)
  when length(inputs) == length(weights) do
    size = length(inputs)
    sum = 0..(size-1)
      |> Enum.map(fn i -> Enum.at(weights, i) * Enum.at(inputs, i) end)
      |> Enum.sum
    activation_function.(sum + bias)
  end
end
