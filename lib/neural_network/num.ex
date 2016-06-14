defmodule Num do
  @doc """
    returns a matrix of zeros, used as a scaffold for backpropagation
  """
  def zeros(x, y) do
    0..(x-1)
      |> 0..(y-1) |> Enum.map(fn _ -> 0 end)
  end

end
