cask "mlx-core" do
  version "26.10.1"
  sha256 "f2d9759bcc132aee57bd78f95b35ddf2a266c48097c049e663653ae3d10ab4ef"

  url "https://github.com/ddalcu/mlx-serve/releases/download/v#{version}/MLX-Serve.dmg"
  name "MLX-Serve"
  desc "Native LLM server for Apple Silicon with OpenAI & Anthropic compatible APIs"
  homepage "https://github.com/ddalcu/mlx-serve"

  depends_on macos: :tahoe
  depends_on arch: :arm64

  app "MLX-Serve.app"

  zap trash: [
    "~/.mlx-serve",
  ]
end
