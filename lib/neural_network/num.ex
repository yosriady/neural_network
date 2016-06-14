defmodule NeuralNetwork.Num do
  @moduledoc """
    Drop-in replacement for NumPy-like methods.
  """

  @doc """
    returns a matrix of zeros, used as a scaffold for backpropagation
  """
  def zeros(x, y) do
    0..(x-1)
      |> Enum.map(fn _ ->
        (0..(y-1) |> Enum.map(fn _ -> 0
      end))
    end)
  end

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
