---
layout:     post
title:      "人脸关键点对应Mask的三角剖分,用于美颜中mask区域对齐原图"
subtitle:   "得到所有的三角形顶点坐标，并得到索引用于glDrawElements渲染位置"
date:       2024-11-30 14:45:00
author:     "CS"
header-img: "img/post-bg-2015.jpg"
catalog: true
tags:
    - landmarks 
    - glDrawElements
    - Delaunay
    - 三角剖分
---

我的主页：[https://www.csblog.site/about/](https://www.csblog.site/about/)

## 1. 引言

在一些安卓相机app中大量使用了OpenGL来实现各种特效，有着比较好的效果。其中人脸中五官的各个区域分别处理使用到了不同的mask掩膜。
在使用OpenGL将mask贴到人脸上，对齐原图的时候，需要得到mask的三角剖分，得到所有的三角形顶点坐标，并得到索引用于glDrawElements渲染位置。然而,mask剖分后有着大量的三角形,每次切换不同关键点数时，都需要重新计算三角剖分，这样的计算量是不可接受的。
因此，我决定使用本脚本代码,得到mask的Delanay三角剖分，得到所有的三角形顶点坐标，并得到索引用于glDrawElements渲染位置。然后直接打印出来,直接在渲染时使用即可。

项目地址：[https://github.com/CSsaan/GitPod_Python/tree/main/openGLPython](https://github.com/CSsaan/GitPod_Python/tree/main/openGLPython)

### 1.1 计算坐标实现代码

**计算渲染坐标与索引时，提前准备:**

- 加载各个mask图(我使用的eyes.png\lip.jpg\facemask.jpg),设置对应路径;
- 根据使用的关键点模型,设置对应的关键点坐标:将mask图上对应关键点的坐标除以图片的宽高，得到归一化坐标;
- main函数中设置参数:
  - chosen_name = "lip": 选择对应区域的mask：例如 eyes \ lip \ face \ ...;
  - upscale = 2 : 图像放大倍率（默认1）;
  - animate = False : 动态显示绘制三角剖分动画;
- 最终, 运行会打印出所有的三角形顶点坐标gl_Position+textureUV: (x, y, 0.0, u, v)，以及索引用于glDrawElements渲染的索引indices。

<table>
  <tr>
    <td><img src="https://github.com/CSsaan/CSsaan.github.io/blob/main/_posts/csdn_md/img/eyes.png?raw=true"/></td>
    <td><img src="https://github.com/CSsaan/CSsaan.github.io/blob/main/_posts/csdn_md/img/lip.jpg?raw=true"/></td>
    <td><img src="https://github.com/CSsaan/CSsaan.github.io/blob/main/_posts/csdn_md/img/facemask.jpg?raw=true"/></td>
  </tr>
</table>

```python
import numpy as np
import cv2
import random

# -------------------- data ----------------------
# mask图
EYES_IMG_PATH = r'D:\AndroidStudioProject\SOBeauty\Beauty\3_CameraSpecialEffect-FBOFaceReshape\app\src\main\assets\resource\eyes.png'
LIP_IMG_PATH = r'D:\AndroidStudioProject\SOBeauty\Beauty\3_CameraSpecialEffect-FBOFaceReshape\app\src\main\assets\resource\lip.jpg'
FACE_IMG_PATH = r'D:\AndroidStudioProject\SOBeauty\Beauty\3_CameraSpecialEffect-FBOFaceReshape\app\src\main\assets\resource\facemask.jpg'
# mask上对应的关键点归一化坐标
# 嘴唇(8个关键点)
LIP_POINTS = [
    (0.154639, 0.378788), (0.398625, 0.196970), (0.512027, 0.287879), (0.611684, 0.212121), (0.872852, 0.378788), 
    (0.639176, 0.848485), (0.522337, 0.846364), (0.398625, 0.843333)]
# 眼睛(16个关键点)
EYES_POINTS = [
    (0.102, 0.465), (0.175, 0.301), (0.370, 0.310), (0.446, 0.603), (0.353, 0.732), (0.197, 0.689), (0.566, 0.629), (0.659, 0.336), 
    (0.802, 0.318), (0.884, 0.465), (0.812, 0.681), (0.681, 0.750), (0.273, 0.241), (0.275, 0.758), (0.721, 0.275), (0.739, 0.758)]
# 脸(79个关键点)
FACE_POINTS = [
    (0.141, 0.508), (0.144, 0.536), (0.147, 0.570), (0.150, 0.603), (0.160, 0.645), (0.166, 0.689), (0.181, 0.724), (0.197, 0.756), (0.225, 0.782), (0.256, 0.810), 
    (0.272, 0.835), (0.304, 0.860), (0.344, 0.888), (0.378, 0.912), (0.419, 0.935), (0.469, 0.946), (0.516, 0.954), (0.559, 0.942), (0.594, 0.932), (0.625, 0.918), 
    (0.662, 0.896), (0.696, 0.872), (0.728, 0.842), (0.762, 0.810), (0.781, 0.774), (0.806, 0.738), (0.819, 0.705), (0.834, 0.672), (0.844, 0.638), (0.853, 0.606), 
    (0.856, 0.578), (0.859, 0.540), (0.862, 0.497), (0.219, 0.446), (0.266, 0.425), (0.319, 0.422), (0.362, 0.427), (0.382, 0.432), (0.397, 0.450), (0.360, 0.452), 
    (0.319, 0.450), (0.262, 0.448), (0.588, 0.441), (0.638, 0.427), (0.684, 0.422), (0.738, 0.425), (0.775, 0.444), (0.740, 0.450), (0.690, 0.448), (0.644, 0.452), 
    (0.588, 0.452), (0.262, 0.520), (0.288, 0.501), (0.325, 0.497), (0.360, 0.508), (0.378, 0.534), (0.350, 0.542), (0.325, 0.540), (0.284, 0.534), (0.606, 0.534), 
    (0.622, 0.515), (0.656, 0.503), (0.694, 0.503), (0.725, 0.522), (0.700, 0.538), (0.662, 0.542), (0.622, 0.542), (0.384, 0.786), (0.432, 0.770), (0.466, 0.766), 
    (0.491, 0.770), (0.518, 0.766), (0.556, 0.773), (0.603, 0.788), (0.588, 0.812), (0.550, 0.838), (0.497, 0.844), (0.447, 0.835), (0.412, 0.816)]
# ------------------------------------------------

def load_image_and_get_size(choice, upscale=1):
    switch = {
        "lip": LIP_IMG_PATH,
        "eyes": EYES_IMG_PATH,
        "face": FACE_IMG_PATH,
        # 可以根据需要继续添加其他选择
    }
    image_path = switch.get(choice)
    if image_path is None:
        raise ValueError(f"Unrecognized choice: {choice}")
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to read image from path: {image_path}")
    img = cv2.resize(img, (img.shape[1]*upscale, img.shape[0]*upscale))
    height, width = img.shape[:2]
    return img, width, height

def get_points(widthHeight, choice="lip"):
    switch = {
        "lip": LIP_POINTS,
        "eyes": EYES_POINTS,
        "face": FACE_POINTS,
        # 可以根据需要继续添加其他选择
    }
    points = switch.get(choice, (None, None))
    if points is None:
        raise ValueError(f"Unrecognized choice: {choice}")
    if len(points) == 0:
        raise ValueError(f"Length of points is 0.")
    # points = list(map(lambda i: (round(x_data[i]*widthHeight[0]), round(y_data[i]*widthHeight[1])), range(len(x_data))))
    unscale_points = [(round(point[0]*widthHeight[0]), round(point[1]*widthHeight[1])) for point in points]
    return unscale_points


#Check if a point is insied a rectangle
def rect_contains(rect, point):
    """
    Checks if a point is inside a rectangle.

    Args:
        rect (tuple): A tuple containing the coordinates of the rectangle
            (x1, y1, x2, y2) where (x1, y1) is the top-left corner and
            (x2, y2) is the bottom-right corner.
        point (tuple): A tuple containing the coordinates of the point
            (x, y).

    Returns:
        bool: True if the point is inside the rectangle, False otherwise.
    """
    if rect is None or point is None:
        raise ValueError("rect and point cannot be None")
    if point[0] < rect[0]:
        return False
    elif point[1] < rect[1]:
        return False
    elif point[0] > rect[2]:
        return False
    elif point[1] > rect[3]:
        return False
    return True

# Draw a point
def draw_point(img, p, color):
    cv2.circle(img, p, 2, color)

def crop_triangle(image, vertices):
    """
    Crops a triangle from the image.

    Args:
        image (numpy.ndarray): The image to crop from.
        vertices (numpy.ndarray): The vertices of the triangle to crop.

    Returns:
        tuple: A tuple containing the cropped triangle and the IoU of the cropped triangle with the original triangle.
    """
    if image is None:
        raise ValueError("image cannot be None")
    if vertices is None:
        raise ValueError("vertices cannot be None")
    if len(vertices) != 3:
        raise ValueError("vertices must contain 3 points")
    if vertices.dtype != np.int32:
        raise ValueError("vertices must be of type np.int32")
    # 三角形区域全1
    triangle_area = np.array([[vertices[0][0],vertices[0][1]], [vertices[1][0],vertices[1][1]], [vertices[2][0],vertices[2][1]]], np.int32)
    triangle_area = triangle_area.reshape((-1, 1, 2))
    mask = np.zeros_like(image, dtype=np.uint8)
    cv2.fillPoly(mask, [triangle_area], (255, 255, 255))
    _, ori_binary = cv2.threshold(mask, 125, 1, cv2.THRESH_BINARY)
    # 三角形区域mask占比
    vertices = vertices.reshape((-1, 1, 2))
    mask = np.zeros_like(image, dtype=np.uint8)
    cv2.fillPoly(mask, [vertices], (255, 255, 255))
    cropped_image = cv2.bitwise_and(image, mask)
    _, tri_binary = cv2.threshold(cropped_image, 100, 1, cv2.THRESH_BINARY)
    iou = np.sum(tri_binary) / np.sum(ori_binary)
    return cropped_image, iou

#Draw delaunay triangles
def draw_delaunay(img, subdiv, delaunay_color, points, points_id_xy, print_index=False):
    """
    Draws a Delaunay triangulation.

    Args:
        img (numpy.ndarray): The image to draw on.
        subdiv (cv2.Subdiv2D): The subdivision from which to draw the triangulation.
        delaunay_color (tuple): The color to draw the triangulation with.
        points (list): The list of points to draw the triangulation with.
        points_id_xy (list): The list of points' id and xy coordinates.
        print_index (bool): Whether to print the indices of the points in the triangulation.

    Returns:
        None
    """
    if img is None:
        raise ValueError("img cannot be None")
    if subdiv is None:
        raise ValueError("subdiv cannot be None")
    if points is None:
        raise ValueError("points cannot be None")
    if points_id_xy is None:
        raise ValueError("points_id_xy cannot be None")
    if len(points) != len(points_id_xy):
        raise ValueError("The length of points and points_id_xy must be the same")
    if print_index:
        print("OpenGL的glDrawElements所有三角形的indices:")
    trangleList = subdiv.getTriangleList()
    size = img.shape
    r = (0, 0, size[1], size[0])
    for t in trangleList:
        pt1 = (int(t[0]), int(t[1]))
        pt2 = (int(t[2]), int(t[3]))
        pt3 = (int(t[4]), int(t[5]))
        if (rect_contains(r, pt1) and rect_contains(r, pt2) and rect_contains(r, pt3)):
            cv2.line(img, pt1, pt2, delaunay_color, 1)
            cv2.line(img, pt2, pt3, delaunay_color, 1)
            cv2.line(img, pt3, pt1, delaunay_color, 1)
        p1 = points.index(pt1)
        p2 = points.index(pt2)
        p3 = points.index(pt3)
        if(print_index):
            # 判断三角形区域是否为mask
            if(points_id_xy is not None):
                # print("now Triangle: ", points_id_xy[p1], points_id_xy[p2], points_id_xy[p3])
                pts = np.array([[points_id_xy[p1][0], points_id_xy[p1][1]], [points_id_xy[p2][0], points_id_xy[p2][1]], [points_id_xy[p3][0], points_id_xy[p3][1]]], np.int32) # 三角形顶点
                cropped_triangle, iou = crop_triangle(img, pts) # 裁剪三角形
                # print("iou: ",iou)
                if (iou > 0.5):
                    print(f"{p1:3d}, {p2:3d}, {p3:3d},")
        else:
            print("pass")
    print("已经自动排除非mask区域的三角形")


# Draw voronoi diagram
def draw_voronoi(img, subdiv):
    """
    Draws a Voronoi diagram.

    Args:
        img (numpy.ndarray): The image to draw on.
        subdiv (cv2.Subdiv2D): The subdivision from which to draw the Voronoi diagram.

    Returns:
        None
    """
    if img is None:
        raise ValueError("img cannot be None")
    if subdiv is None:
        raise ValueError("subdiv cannot be None")
    (facets, centers) = subdiv.getVoronoiFacetList([])
    if facets is None or centers is None:
        raise ValueError("getVoronoiFacetList returned None")
    for i in range(0, len(facets)):
        ifacet_arr = []
        for f in facets[i]:
            ifacet_arr.append(f)
        ifacet = np.array(ifacet_arr, np.int_)
        if ifacet is None:
            raise ValueError("getVoronoiFacetList returned None")
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        try:
            cv2.fillConvexPoly(img, ifacet, color)
        except Exception as e:
            raise ValueError(f"Failed to fill convex polygon: {e}")
        ifacets = np.array([ifacet])
        try:
            cv2.polylines(img, ifacets, True, (0, 0, 0), 1)
        except Exception as e:
            raise ValueError(f"Failed to draw polylines: {e}")
        try:
            cv2.circle(img, (int(centers[i][0]), int(centers[i][1])), 3, (0, 0, 0))
        except Exception as e:
            raise ValueError(f"Failed to draw circle: {e}")

def get_gradient_color(i, total):
    color_r = int(255 * i / total) if i % 2 == 1 else 255-int(255 * i / total)
    color_g = int(255 * (total - i) / total) if i % 2 == 1 else 255-int(255 * (total - i) / total)
    color_b = int(255 * i / total)
    return (color_r, color_g, color_b)

# Draw number
def draw_number(img, subdiv):
    """
    Draws the Voronoi diagram with point IDs.

    Args:
        img (numpy.ndarray): The image to draw on.
        subdiv (cv2.Subdiv2D): The subdivision from which to draw the Voronoi diagram.

    Returns:
        dict: A dictionary mapping point IDs to their corresponding coordinates.
    """
    if img is None:
        raise ValueError("img cannot be None")
    if subdiv is None:
        raise ValueError("subdiv cannot be None")
    result = {}  # 创建一个空字典 id:(x,y)
    height, width = img.shape[:2]
    (facets, centers) = subdiv.getVoronoiFacetList([])
    if centers is None:
        raise ValueError("getVoronoiFacetList returned None")
    for i in range(0, len(centers)):
        assert centers[i] is not None
        assert centers[i][0] is not None
        assert centers[i][1] is not None
        x = int(centers[i][0])
        y = int(centers[i][1])
        # 图片 添加的文字 位置 字体 字体大小 字体颜色 字体粗细
        try:
            print(f"pointID:{i:2d}, gl_Position+textureUV:({centers[i][0]/width*2.0-1.0:.3f}, {centers[i][1]/height*2.0-1.0:.3f}, 0.0,    {centers[i][0]/width:.3f}, {1.0-centers[i][1]/height:.3f})")
            cv2.putText(img, str(i)+f"({x/width:.3f},{y/height:.3f})", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.4, color=get_gradient_color(i,len(centers)), thickness=1, lineType=cv2.LINE_AA)
        except Exception as e:
            raise ValueError(f"Failed to draw text: {e}")
        result[i] = (x, y)
    return result



if __name__ == '__main__':

    # --------------- config params ------------------
    chosen_name = "lip" # 选择对应区域的mask：例如 eyes \ lip \ face \ ...
    upscale = 2          # 图像放大倍率（默认1）
    animate = False      # 动态显示绘制三角剖分动画
    # ------------------------------------------------

    # 加载图像
    chosen_image, image_width, image_height = load_image_and_get_size(chosen_name, upscale)
    assert chosen_image is not None, "chosen_image加载图像失败, 为None."
    print("成功加载图像 - 宽度：{}，高度：{}.".format(image_width, image_height))
    # 加载关键点
    chosen_points = get_points((image_width, image_height), chosen_name)
    print("all points: ", chosen_points)

    # 绘制三角剖分图
    img = chosen_image.copy()
    img_orig = img.copy()
    #Rectangle to be used with Subdiv2D
    size = img.shape
    rect = (0, 0, size[1], size[0])
    #Create an instance of Subdiv2d
    subdiv = cv2.Subdiv2D(rect)
    #Insert points into subdiv
    for p in chosen_points:
        subdiv.insert(p)
        #Show animate # 动态绘制
        if animate:
            img_copy = img_orig.copy()
            #Draw delaunay triangles
            draw_delaunay(img_copy, subdiv, (0,0,255), chosen_points, None, print_index=False)
            cv2.imshow("三角剖分(Delaunay Triangulation)", img_copy)
            cv2.waitKey(50) 
    #Draw delaunary triangles
    points_id_xy = draw_number(img, subdiv)
    # 在图像上绘制每个点
    for i in range(len(points_id_xy)):
        cv2.circle(chosen_image, points_id_xy[i], 2, (0, 255, 0), -1)
        cv2.putText(chosen_image, str(i), points_id_xy[i], cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.4, color=get_gradient_color(i,len(points_id_xy)), thickness=1, lineType=cv2.LINE_AA)
    cv2.imshow('Image with points', chosen_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print("points_id-xy: ", points_id_xy)
    draw_delaunay(img, subdiv, (0,0,255), chosen_points, points_id_xy, print_index=True)
    #Draw points
    for p in chosen_points:
        draw_point(img, p, (0,0,255))
    #Allocate space for Voroni Diagram
    img_voronoi = np.zeros(img.shape, dtype = img.dtype)
    #Draw Voonoi diagram
    draw_voronoi(img_voronoi, subdiv)
    #Show results
    cv2.imshow("三角剖分(Delaunay Triangulation)", img)
    cv2.imshow("维诺图(Voronoi Diagram)", img_voronoi)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print("done.")
```

**运行后打印结果:**

```text
成功加载图像 - 宽度：{}，高度：{}.
all points:  [(90, 100), (232, 52), (298, 76), (356, 56), (508, 100), (372, 224), (304, 223), (232, 223)]
pointID: 0, gl_Position+textureUV:(-0.691, -0.242, 0.0,    0.155, 0.621)
pointID: 1, gl_Position+textureUV:(-0.203, -0.606, 0.0,    0.399, 0.803)
pointID: 2, gl_Position+textureUV:(0.024, -0.424, 0.0,    0.512, 0.712)
pointID: 3, gl_Position+textureUV:(0.223, -0.576, 0.0,    0.612, 0.788)
pointID: 4, gl_Position+textureUV:(0.746, -0.242, 0.0,    0.873, 0.621)
pointID: 5, gl_Position+textureUV:(0.278, 0.697, 0.0,    0.639, 0.152)
pointID: 6, gl_Position+textureUV:(0.045, 0.689, 0.0,    0.522, 0.155)
pointID: 7, gl_Position+textureUV:(-0.203, 0.689, 0.0,    0.399, 0.155)
points_id-xy:  {0: (90, 100), 1: (232, 52), 2: (298, 76), 3: (356, 56), 4: (508, 100), 5: (372, 224), 6: (304, 223), 7: (232, 223)}
OpenGL的glDrawElements所有三角形的indices:
  2,   6,   7,
  6,   2,   5,
  0,   1,   7,
  2,   7,   1,
  3,   5,   2,
  5,   3,   4,
done.
```

其中所有三角形的顶点数组为:
[-0.690, -0.242, 0.0,  0.154639, 0.378788,
 -0.202, -0.606, 0.0,  0.398625, 0.196970,
  0.024, -0.424, 0.0,  0.512027, 0.287879,
  0.223, -0.575, 0.0,  0.611684, 0.212121,
  0.740, -0.242, 0.0,  0.872852, 0.378788,
  0.278,  0.696, 0.0,  0.639176, 0.848485,
  0.040,  0.272, 0.0,  0.522337, 0.636364,
 -0.202,  0.666, 0.0,  0.398625, 0.833333 ]
三角形的索引为:
[0, 1, 7,
 7, 1, 6,
 1, 6, 2,
 2, 6, 3,
 5, 6, 3,
 5, 3, 4 ]

### 1.2 渲染mask代码

**根据三角形坐标和index索引渲染mask:**

```python
import numpy as np
from OpenGL.GL import *
from OpenGL.arrays.vbo import VBO
from utils.shader import Shader
from utils.texture import Texture
from utils.window import Window
from utils.myUtils import ensure_directory_exists
from utils.framebuffer import FBO
import argparse
import cv2
import os
import nanogui

POINTS_NUM = 8


parser = argparse.ArgumentParser()
parser.add_argument('--project_name', default="Test CS", type=str, help="Window's name")
parser.add_argument('--inputVideo_path', default="./resource/toothMask.mp4", type=str, help='input a video to render frames')
parser.add_argument('--inputMask_path', default="./resource/640/6.avi", type=str, help='input a ai mask result video to render frames')
parser.add_argument('--save_video', default=False, type=bool, help='if save frames to a video')
parser.add_argument('--saveVideo_path', default="./result/640-2.mp4", type=str, help='save frames to a video')
parser.add_argument('--concat_ori_result', default=False, type=bool, help='concat origin & result') 
parser.add_argument('--save_frames', default=False, type=bool, help='if save frames to a folder')
parser.add_argument('--saveFrames_path', default="./result/frames", type=str, help='save frames to a folder')
parser.add_argument('--show_on_screen', default=True, type=bool, help='show result on screen')
args = parser.parse_args()

ensure_directory_exists(os.path.dirname(args.saveVideo_path))
ensure_directory_exists(args.saveFrames_path)

# 帧数
frame_n = 1
cap = cv2.VideoCapture(args.inputVideo_path)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

cap_aimask = cv2.VideoCapture(args.inputMask_path)

# 保存视频
window_w, window_h = video_width, video_height # video_width//2, video_height//2
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(args.saveVideo_path, fourcc, fps, (window_w*2 if args.concat_ori_result else window_w, window_h), True)

# 创建窗口
print(f"window size:[{window_w},{window_h}]")
w = Window(window_w, window_h, args.project_name)

# Program segment
shader = Shader("./shaders/ToothWhiten/toothMask.vert", "./shaders/ToothWhiten/toothMask.frag") # dilate  Green_segmentation
shader_2D = Shader("./shaders/base.vert", "./shaders/base.frag") # normal 2D

# 顶点数据
vert2D = np.array(
    [-1.0, -1.0, 0.0,  0.0, 1.0,
      1.0, -1.0, 0.0,  1.0, 1.0, 
      1.0,  1.0, 0.0,  1.0, 0.0, 
     -1.0, -1.0, 0.0,  0.0, 1.0,
      1.0,  1.0, 0.0,  1.0, 0.0, 
     -1.0,  1.0, 0.0,  0.0, 0.0	], dtype=np.float32)
triangle = np.array(
    [-0.690, -0.242, 0.0,  0.154639, 0.378788,
     -0.202, -0.606, 0.0,  0.398625, 0.196970,
      0.024, -0.424, 0.0,  0.512027, 0.287879,
      0.223, -0.575, 0.0,  0.611684, 0.212121,
      0.740, -0.242, 0.0,  0.872852, 0.378788,
      0.278,  0.696, 0.0,  0.639176, 0.848485,
      0.040,  0.272, 0.0,  0.522337, 0.636364,
     -0.202,  0.666, 0.0,  0.398625, 0.833333 ], dtype=np.float32)
assert triangle.nbytes % (POINTS_NUM*5) == 0, "不能被整除"
every_size = triangle.nbytes//(POINTS_NUM*5)
print(triangle.nbytes, every_size)

# VAO & VBO
vao = glGenVertexArrays(1)
glBindVertexArray(vao)
vbo = VBO(triangle, GL_STATIC_DRAW)
vbo.bind()
shader.setAttrib(0, 3, GL_FLOAT, every_size*5, 0)
shader.setAttrib(1, 2, GL_FLOAT, every_size*5, every_size*3)


vao2D = glGenVertexArrays(1)
glBindVertexArray(vao2D)
vbo2D = VBO(vert2D, GL_STATIC_DRAW)
vbo2D.bind()
shader_2D.setAttrib(0, 3, GL_FLOAT, every_size*5, 0)
shader_2D.setAttrib(1, 2, GL_FLOAT, every_size*5, every_size*3)

# 创建Fbo&Texture [shader result]
fbo_green = FBO(window_w, window_h)
# 创建Fbo
fbo_2d = FBO(window_w, window_h)

# 创建纹理
tex = Texture(idx=0, texType=GL_TEXTURE_2D, imgType=GL_RGB, innerType=GL_RGB, dataType=GL_UNSIGNED_BYTE, w=video_width, h=video_height)
tex_background = Texture(idx=1, imgPath="./resource/sight.jpg")

def quite_cap(self):
    cap.release()
    cap_aimask.release()

# 渲染循环
def render():
    glDisable(GL_CULL_FACE)
    # 读取每一帧
    # img = cv2.imread("./resource/sight.jpg")
    global frame_n
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_n)
    ret, img = cap.read()
    cap_aimask.set(cv2.CAP_PROP_POS_FRAMES, frame_n)
    ret_aimask, img_aimask = cap_aimask.read()
    if not ret and not ret_aimask:
        raise ValueError("无法读取视频帧")
    frame_n += 1
    print("\r" + f'{frame_n}/{total_frames}')

    # -----------------------------------------------
    # update VBO
    increment = np.array(
        # ( x      y      z )  (  x      1-y )
        [-0.690, -0.242, 0.0,  0.154639, 0.621,
         -0.202, -0.606, 0.0,  0.398625, 0.803,
          0.024, -0.424, 0.0,  0.512027, 0.712,
          0.223, -0.575, 0.0,  0.611684, 0.787,
          0.740, -0.242, 0.0,  0.872852, 0.621,
          0.278,  0.696, 0.0,  0.639176, 0.151,
          0.040,  0.692, 0.0,  0.522337, 0.153,
         -0.202,  0.696, 0.0,  0.398625, 0.156 ], dtype=np.float32)
    vbo.set_array(increment)
    vbo.bind()
    # draw framebuffer [Green]
    fbo_green.bind()
    shader.use()
    glBindVertexArray(vao)
    tex.updateTex(shader, "tex", img) # 原视频纹理
    tex_background.useTex(shader, "tex_background") # 背景
    shader.setUniform("strenth", 0.9)
    shader.setUniform("gpow", 0.5)
    
    indices = np.array(
        [0, 1, 7,
         7, 1, 6,
         1, 6, 2,
         2, 6, 3,
         5, 6, 3,
         5, 3, 4 ], dtype=np.int8)

    glDrawElements(GL_TRIANGLES, indices.nbytes, GL_UNSIGNED_SHORT, indices)
    # glDrawArrays(GL_TRIANGLES, 0, 6)
    glBindTexture(GL_TEXTURE_2D, GL_NONE)
    glBindVertexArray(0)
    glUseProgram(GL_NONE)
    fbo_green.unbind()
    # -----------------------------------------------
    # draw normal 2D on screen
    if(not args.show_on_screen):
        fbo_2d.bind()
    shader_2D.use()
    glBindVertexArray(vao2D)
    glActiveTexture(GL_TEXTURE0)
    glBindTexture(GL_TEXTURE_2D, fbo_green.uTexture)
    shader_2D.setUniform("tex", 0)
    glDrawArrays(GL_TRIANGLES, 0, 6)
    glBindTexture(GL_TEXTURE_2D, GL_NONE)
    glBindVertexArray(0)
    glUseProgram(GL_NONE)
    
    # save
    if(args.save_video or args.save_frames):
        data = glReadPixels(0, 0, window_w, window_h, GL_BGR, GL_UNSIGNED_BYTE)  # 注意这里使用BGR通道顺序
        image = np.frombuffer(data, dtype=np.uint8).reshape(window_h, window_w, 3)
        image = cv2.flip(image, 0)

        if(args.concat_ori_result):
            img = cv2.resize(img, (image.shape[1], image.shape[0]))
            result = cv2.hconcat([img, image])
        else:
            result = image

        try:
            if(args.save_frames):
                cv2.imwrite(f"{args.saveFrames_path}/output_{frame_n-1}.png", result)
            if(args.save_video):
                video_writer.write(result)
        except Exception as e:
            print(e)
            video_writer.release()
    if(not args.show_on_screen):
        fbo_2d.unbind()

if __name__ == '__main__':
    w.loop(render)

```