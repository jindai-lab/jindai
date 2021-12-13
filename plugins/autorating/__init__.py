import os
from PIL import Image
import config

from .inference_model import InferenceModel, load_state

model = None


def predict(i):
    global model
    if model is None:
        state = load_state(os.path.join(config.rootpath, 'models_data', 'autorating_best.pth'))
        model = InferenceModel(state)
       
    t = Image.open(i.read_image())
    return float(model.predict_from_pil_image(t).numpy()[0])
