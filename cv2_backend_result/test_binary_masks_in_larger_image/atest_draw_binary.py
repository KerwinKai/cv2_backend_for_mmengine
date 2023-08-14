from mmengine.visualization import Visualizer
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
if __name__ == '__main__':
    function = 'test_binary_mask_in larger_image'
    backends = ['matplotlib', 'cv2']
    image = Image.open('images/cat_and_dog.png')
    image = np.array(image)
    binary_mask = np.random.randint(0, 2, size=(3, 224, 224)).astype(bool)
    for backend in backends:
        vis = Visualizer(image=np.array(image),
                         backend=backend)
        vis.draw_binary_masks(binary_mask, colors=['r', 'g', 'b'])
        result_img = vis.get_image()
        save_name = f'./images/{function}_{backend}.png'
        if backend == 'matplotlib':
            plt.imshow(result_img)
            plt.axis('off')
            plt.savefig(save_name, bbox_inches='tight', pad_inches=0)
        else:
            result_img = cv2.cvtColor(
                result_img,
                cv2.COLOR_RGB2BGR)
            cv2.imwrite(save_name, result_img)