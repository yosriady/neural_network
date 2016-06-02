defmodule NeuralNetworkTest do
  use ExUnit.Case
  doctest NeuralNetwork

  test "Basic neural network" do
    sizes = [2,3,1]
    n = NeuralNetwork.Network.new(sizes)
    output = NeuralNetwork.Network.feedforward(n, [0,1])
    assert length(output) == Enum.at(sizes, -1)
  end
end
