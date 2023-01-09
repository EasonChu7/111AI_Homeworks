using ExplainableAI
using Flux
using Metalhead                         
using HTTP, FileIO, ImageMagick  
using ImageShow        

#VGG DL Model
model = VGG().layers
model = strip_softmax(flatten_chain(model))

# 載入圖片
url = HTTP.URI("https://image.cache.storm.mg/styles/smg-800x533-fp/s3/media/image/2020/11/07/20201107-092915_U13380_M651499_4ac4.jpg?itok=6KFZde7p")
img = load(url)
input = preprocess_imagenet(img)
input = reshape(input, 224, 224, 3, :)  # 轉成WHCN格式

#LRP
analyzer = LRP(model)
expl = analyze(input, analyzer)   
heatmap(expl)
analyze(input, analyzer) 

#LRP+EpsilonAlpha2Beta1
composite = EpsilonAlpha2Beta1()
analyzer_2 = LRP(model,composite)
expl_2 = analyze(input, analyzer_2)   
heatmap(expl_2)

#LRP+EpsilonPlusFlat
composite = EpsilonPlusFlat()
analyzer_3 = LRP(model,composite)
expl_3 = analyze(input, analyzer_3)   
heatmap(expl_3)