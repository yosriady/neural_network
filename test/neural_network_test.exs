defmodule NeuralNetworkTest do
  use ExUnit.Case
  doctest NeuralNetwork

  alias NeuralNetwork.Perceptron

  test "Invalid size Perceptron" do
    weights = [-2.0, -2.0]
    size = length(weights) - 1
    bias = 3.0
    assert_raise(FunctionClauseError, (fn -> Perceptron.new(size, weights, bias) end))
  end

  test "Valid Perceptron with invalid input size" do
    weights = [-2.0, -2.0]
    size = length(weights)
    bias = 3.0
    p = Perceptron.new(size, weights, bias)
    assert_raise(FunctionClauseError, (fn -> Perceptron.output(p, [0.0]) end))
  end

  test "Valid Perceptron with valid inputs" do
    weights = [-2.0, -2.0]
    size = length(weights)
    bias = 3.0
    p = Perceptron.new(size, weights, bias)
    assert Perceptron.output(p, [0.0,0.0]) == 3
    assert Perceptron.output(p, [1.0,1.0]) == -1
  end
end
