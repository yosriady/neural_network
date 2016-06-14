defmodule NeuralNetwork.Num do
  @moduledoc """
    Drop-in replacement for NumPy-like methods.
  """

  @doc """
    returns a x high * y wide matrix of zeros, used as a scaffold for backpropagation
  """
  def zeros(x, y) do
    0..(x-1)
      |> Enum.map(fn _ -> zeros(y) end)
  end

  @doc """
    returns a list of zeros, with length x
  """
  def zeros(x) do
    0..(x-1)
      |> Enum.map(fn _ -> 0 end)
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

  @doc """
    Merges in two lists, applies a function on each of their contents and return a new list.
  """
  @spec merge_lists(list(term), list(term), ({term, term} -> term)) :: list(term)
  def merge_lists(a, b, fun) do
    Enum.zip(a, b)
      |> Enum.map(fun)
  end

  @doc """
    Merges in two matrices, applies a function on each of their contents and return a new matrix.
  """
  @spec merge_matrices(list(list(term)), list(list(term)), ({term, term} -> term)) :: list(list(term))
  def merge_matrices(a, b, fun) do
    Enum.zip(a, b)
      |> Enum.map(fn {h, t} -> merge_lists(h, t, fun) end)
  end

  @doc """
    Merges in two lists of matrices, applies a function on each of ither contents and return a new list of matrices.
  """
  @spec merge_lists_of_matrices(list(list(list(term))), list(list(list(term))), ({term, term} -> term)) :: list(list(list(term)))
  def merge_lists_of_matrices(a, b, fun) do
      Enum.zip(a, b)
        |> Enum.map(fn {h, t} -> merge_matrices(h, t, fun) end)
  end
end
