defmodule NeuronTest do
  use ExUnit.Case
  doctest NeuralNetwork.Neuron

  alias NeuralNetwork.Neuron

  test "Valid Neuron with invalid input size" do
    weights = [-2.0, -2.0]
    bias = 3.0
    p = Neuron.new( weights, bias)
    assert_raise(FunctionClauseError, (fn -> Neuron.activate(p, [0.0]) end))
  end

  test "Valid Neuron with valid inputs" do
    weights = [-2.0, -2.0]
    bias = 3.0
    p = Neuron.new(weights, bias)
    assert Neuron.activate(p, [0.0,0.0]) == 0.9525741268224331
    assert Neuron.activate(p, [1.0,1.0]) == 0.2689414213699951
  end
end
