from tensorflow.keras.models import load_model

# 请确保路径是您保存的模型的正确路径
model_path = 'models/mobilenet_fv.h5'

# 加载模型
model = load_model(model_path)

# 打印模型概要
model.summary()
