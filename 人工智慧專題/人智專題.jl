using ExplainableAI
using Flux
using Metalhead                         
using HTTP, FileIO, ImageMagick         

# Load model
model = VGG(16, pretrain=true).layers
model = strip_softmax(flatten_chain(model))

# Load input
url = HTTP.URI("https://raw.githubusercontent.com/adrhill/ExplainableAI.jl/gh-pages/assets/heatmaps/castle.jpg")
img = load(url)
input = preprocess_imagenet(img)
input = reshape(input, 224, 224, 3, :)  # reshape to WHCN format

# Run XAI method
analyzer = LRP(model)
expl = analyze(input, analyzer)         # or: expl = analyzer(input)

# Show heatmap
heatmap(expl)

# Or analyze & show heatmap directly
heatmap(input, analyzer)