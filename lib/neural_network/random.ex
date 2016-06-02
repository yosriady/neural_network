defmodule NeuralNetwork.Random do
  @moduledoc """
    This module implements functions to generate gaussian numbers.
  """

  @doc """
    randn returns an array of size x high * y wide,
    where each is generated from a Gaussian distribution with mean 0 and standard deviation 1
  """
  def randn(x, y) do
    0..(x-1)
      |> Enum.map(fn _ -> gauss(y) end)
  end

  @doc """
    Generates an array of size size, generated from a gaussian distribution
    of mean 0 and standard deviation 1.
  """
  def gauss(size) do
    0..(size-1)
      |> Enum.map(fn _ -> Statistics.Distributions.Normal.rand() end)
  end
end
