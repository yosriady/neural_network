defmodule NeuronTest do
  use ExUnit.Case
  doctest NeuralNetwork.Neuron

  alias NeuralNetwork.Neuron

  test "Invalid size Neuron" do
    weights = [-2.0, -2.0]
    size = length(weights) - 1
    bias = 3.0
    assert_raise(FunctionClauseError, (fn -> Neuron.new(size, weights, bias) end))
  end

  test "Valid Neuron with invalid input size" do
    weights = [-2.0, -2.0]
    size = length(weights)
    bias = 3.0
    p = Neuron.new(size, weights, bias)
    assert_raise(FunctionClauseError, (fn -> Neuron.output(p, [0.0]) end))
  end

  test "Valid Neuron with valid inputs" do
    weights = [-2.0, -2.0]
    size = length(weights)
    bias = 3.0
    p = Neuron.new(size, weights, bias)
    assert Neuron.output(p, [0.0,0.0]) == 0.9525741268224331
    assert Neuron.output(p, [1.0,1.0]) == 0.2689414213699951
  end
end
