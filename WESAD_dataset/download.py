import kagglehub

# 下载数据集
path = kagglehub.dataset_download("orvile/wesad-wearable-stress-affect-detection-dataset")
print("数据下载完成，存放路径为:", path)