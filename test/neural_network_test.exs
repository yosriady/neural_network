defmodule NeuralNetworkTest do
  use ExUnit.Case
  doctest NeuralNetwork

  test "Basic neural network" do
    sizes = [2,3,1]
    n = NeuralNetwork.Network.new(sizes)
    activations = NeuralNetwork.Network.feedforward(n, [0,1])
    assert length(activations) == length(sizes)

    second_layer_activations = Enum.at(activations, 1)
    assert length(second_layer_activations) == Enum.at(sizes, 1)
  end
end
