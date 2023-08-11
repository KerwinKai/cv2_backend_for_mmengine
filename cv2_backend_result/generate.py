from mmengine.visualization import Visualizer
import matplotlib.pyplot as plt
import numpy as np
import cv2
if __name__ == '__main__':
    image = np.zeros((500, 500, 3))
    backends = ['matplotlib', 'cv2']
    # test draw_points
    function = 'draw_points'
    for backend in backends:
        vis = Visualizer(image=image,
                         backend=backend)
        vis.draw_points(
            np.array([[100, 300], [200, 400]]),
            colors=['g', 'r']
        )
        result_img = vis.get_image()
        save_name = f'./images/{function}_{backend}.png'
        if backend == 'matplotlib':
            plt.imshow(result_img)
            plt.axis('off')
            plt.savefig(save_name, bbox_inches='tight', pad_inches = 0)
        else:
            cv2.imwrite(save_name, result_img)

    # test draw_bboxes
    function = 'draw_bboxes'
    for backend in backends:
        vis = Visualizer(image=image,
                         backend=backend)
        vis.draw_bboxes(bboxes=np.array([[100, 100, 200, 200], [200, 200, 300, 300]]),
                        edge_colors=['g', 'r'])
        result_img = vis.get_image()
        save_name = f'./images/{function}_{backend}.png'
        if backend == 'matplotlib':
            plt.imshow(result_img)
            plt.axis('off')
            plt.savefig(save_name, bbox_inches='tight', pad_inches=0)
        else:
            cv2.imwrite(save_name, result_img)

    # test draw_lines
    function = 'draw_lines'
    for backend in backends:
        vis = Visualizer(image=image,
                         backend=backend)
        vis.draw_lines(x_datas=np.array([[100, 300], [200, 400]]), y_datas=np.array([[100, 300], [200, 400]]),
                       colors=['w', 'r'], line_widths=[1, 2])
        result_img = vis.get_image()
        save_name = f'./images/{function}_{backend}.png'
        if backend == 'matplotlib':
            plt.imshow(result_img)
            plt.axis('off')
            plt.savefig(save_name, bbox_inches='tight', pad_inches=0)
        else:
            cv2.imwrite(save_name, result_img)

    # test draw_texts
    function = 'draw_texts'
    for backend in backends:
        vis = Visualizer(image=image,
                         backend=backend)
        vis.draw_texts(texts=['MMEngine', 'OpenMMLab'], positions=np.array([[200, 200], [350, 230]]),colors=['b', 'w'],
                       bboxes=dict(facecolor='r', alpha=0.2))
        result_img = vis.get_image()
        save_name = f'./images/{function}_{backend}.png'
        if backend == 'matplotlib':
            plt.imshow(result_img)
            plt.axis('off')
            plt.savefig(save_name, bbox_inches='tight', pad_inches=0)
        else:
            cv2.imwrite(save_name, result_img)

    # test draw_circles
    function = 'draw_circles'
    for backend in backends:
        vis = Visualizer(image=image,
                         backend=backend)
        vis.draw_circles(center=np.array([[200, 200], [300, 500]]), radius=np.array([100, 200]), edge_colors=['g', 'r'])
        result_img = vis.get_image()
        save_name = f'./images/{function}_{backend}.png'
        if backend == 'matplotlib':
            plt.imshow(result_img)
            plt.axis('off')
            plt.savefig(save_name, bbox_inches='tight', pad_inches=0)
        else:
            cv2.imwrite(save_name, result_img)

    # test draw_polygons
    function = 'draw_polygons'
    for backend in backends:
        vis = Visualizer(image=image,
                         backend=backend)
        squares = [np.array([[0, 0], [100, 0], [100, 100], [0, 100]]), np.array([[0, 0], [50, 0], [50, 50], [0, 50]])]
        vis.draw_polygons(polygons=squares, edge_colors=['g', 'r'])
        result_img = vis.get_image()
        save_name = f'./images/{function}_{backend}.png'
        if backend == 'matplotlib':
            plt.imshow(result_img)
            plt.axis('off')
            plt.savefig(save_name, bbox_inches='tight', pad_inches=0)
        else:
            cv2.imwrite(save_name, result_img)

    # test draw_binary_masks
    function = 'draw_binary_masks'
    image = np.random.randint(
        0, 256, size=(10, 10, 3)).astype('uint8')
    binary_mask = np.random.randint(0, 2, size=(2, 10, 10)).astype(bool)
    for backend in backends:
        vis = Visualizer(image=image,
                         backend=backend)
        vis.draw_binary_masks(binary_mask, colors=['r', 'g'])
        result_img = vis.get_image()
        save_name = f'./images/{function}_{backend}.png'
        if backend == 'matplotlib':
            plt.imshow(result_img)
            plt.axis('off')
            plt.savefig(save_name, bbox_inches='tight', pad_inches=0)
        else:
            cv2.imwrite(save_name, result_img)

'''
vis.draw_binary_masks(binary_mask, alpha=0.6)
'''