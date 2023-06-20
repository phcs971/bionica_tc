import cv2
from skimage import measure
from mayavi import mlab
import numpy as np
import meshio
import os
from math import pi, cos, sin, inf

START_RADIUS = 225
DISCONTINUITIES_THRESHOLD = .5
DELTA_ANGLE = 2*pi/150

def distance(point1, point2):
    return ((point2[0]-point1[0])**2 + (point2[1]-point1[1])**2)**(1/2)

def find_countours(thresh):
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def draw_contours(image, contours):
    cv2.drawContours(image, contours, -1, (0, 255, 0), 3)

def find_mean_radius(image, contours):
    center = [int(image.shape[1]/2), int(image.shape[0]/2)]
    mean_radius = 0
    n = 0
    for contour in contours:
        for point in contour:
            point = point[0]
            mean_radius += distance(center, point)
            n += 1
    return mean_radius / n

def find_discontinuities(image, contours):
    angle = 0
    discontinuities = []
    center = [int(image.shape[1]/2), int(image.shape[0]/2)]
    while angle < 2*pi:
        x = int(center[0] + START_RADIUS * cos(angle))
        y = int(center[1] + START_RADIUS * sin(angle))

        d = distance(center, (x, y))
        point_in_line = False
        for contour in contours:
            for point in contour:
                point = point[0]
                dc = distance(center, point)
                de = distance((x, y), point)
                if (dc+de) - d < DISCONTINUITIES_THRESHOLD:
                    point_in_line = True
                    break
            if point_in_line:
                break
        
        if not point_in_line:
            discontinuities.append(angle)

        angle += DELTA_ANGLE
    return discontinuities

def filter_discontinuities(discontinuities):
    groups = [[discontinuities[0]]]
    for discontinuity in discontinuities[1:]:
        added = False
        for group in groups:
            closest_angle_dist = inf
            for angle in group:
                dist = abs(angle - discontinuity)
                if dist < closest_angle_dist:
                    closest_angle_dist = dist
            if closest_angle_dist <= DELTA_ANGLE * 1.1:
                group.append(discontinuity)
                added = True
                break
        if not added:
            groups.append([discontinuity])

    max_group = groups[0]
    for group in groups[1:]:
        if len(group) > len(max_group):
            max_group = group
    return max_group

def draw_discontinuities(image, discontinuities, radius):
    center = [int(image.shape[1]/2), int(image.shape[0]/2)]
    # for angle in [discontinuities[0], discontinuities[-1]]:
    for angle in discontinuities:
        x = int(center[0] + radius * cos(angle))
        y = int(center[1] + radius * sin(angle))
        cv2.circle(image, (x,y), 5, (255, 0, 0), -1)

def find_corners(image, min_distance):
    corners = cv2.goodFeaturesToTrack(image, 100, 0.225, min_distance)
    corners = np.intp(corners)
    return corners

def filter_corners(image, corners, discontinuities, radius):
    center = [int(image.shape[1]/2), int(image.shape[0]/2)]
    start_angle = discontinuities[0]
    start_point = [int(center[0] + radius * cos(start_angle)), int(center[1] + radius * sin(start_angle))]
    end_angle = discontinuities[-1]
    end_point = [int(center[0] + radius * cos(end_angle)), int(center[1] + radius * sin(end_angle))]


    #get the two closest corners to the start point
    closest_corners_start = [corners[0], corners[1]]
    closest_corners_start_dist = [distance(start_point, corners[0][0]), distance(start_point, corners[1][0])]
    for corner in corners[2:]:
        dist = distance(start_point, corner[0])
        if dist < closest_corners_start_dist[0] and closest_corners_start_dist[0] > closest_corners_start_dist[1]:
            closest_corners_start_dist[0] = dist
            closest_corners_start[0] = corner
        elif dist < closest_corners_start_dist[1]:
            closest_corners_start_dist[1] = dist
            closest_corners_start[1] = corner

    #get the two closest corners to the end point
    closest_corners_end = [corners[0], corners[1]]
    closest_corners_end_dist = [distance(end_point, corners[0][0]), distance(end_point, corners[1][0])]
    for corner in corners[2:]:
        dist = distance(end_point, corner[0])
        if dist < closest_corners_end_dist[0]  and closest_corners_end_dist[0] > closest_corners_end_dist[1]:
            closest_corners_end_dist[0] = dist
            closest_corners_end[0] = corner
        elif dist < closest_corners_end_dist[1]:
            closest_corners_end_dist[1] = dist
            closest_corners_end[1] = corner

    return (closest_corners_start, closest_corners_end)

def draw_corners(image, corners, color):
    for corner in corners:
        x, y = corner.ravel()
        cv2.circle(image, (x,y), 5, color, -1)

def reconstruct_horizontal_flip(image, index):
    copy = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    contours = find_countours(thresh)
    # draw_contours(image, contours)

    radius = find_mean_radius(image, contours)

    discontinuities = find_discontinuities(image, contours)
    discontinuities = filter_discontinuities(discontinuities)
    draw_discontinuities(image, discontinuities, radius)

    corners = find_corners(thresh, 8 if index < 16 else 15)
    (cornersA, cornersB) = filter_corners(image, corners, discontinuities, radius)
    draw_corners(image, cornersA,(0, 0, 255))
    draw_corners(image, cornersB, (0, 0, 255))

    
    prosthetic = cv2.flip(copy.copy(), 1)
    size = prosthetic.shape

    prosthetic[:, 0:int(size[1]/2)] = 0
    prosthetic[cornersB[0][0][1]:, :] = 0

    fixed = cv2.add(copy.copy(), prosthetic.copy())

    return image, prosthetic, fixed

def reconstruct_top_circle(image, prosthetic, index):
    copy = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    contours = find_countours(thresh)
    draw_contours(image, contours)

    radius = find_mean_radius(image, contours)

    discontinuities = find_discontinuities(image, contours)
    discontinuities = filter_discontinuities(discontinuities)
    draw_discontinuities(image, discontinuities, radius)

    corners = find_corners(thresh, 8 if index < 16 else 15)
    (cornersA, cornersB) = filter_corners(image, corners, discontinuities, radius)
    draw_corners(image, cornersA,(0, 0, 255))
    draw_corners(image, cornersB, (0, 0, 255))

    center_down = 3 - (index-2)/20

    center = [int(image.shape[1]/2), int(image.shape[0]/center_down)]

    mA = (cornersA[0][0][1] - cornersA[1][0][1]) / (cornersA[0][0][0] - cornersA[1][0][0])
    nA = cornersA[0][0][1] - mA * cornersA[0][0][0]

    mB = (cornersB[0][0][1] - cornersB[1][0][1]) / (cornersB[0][0][0] - cornersB[1][0][0])
    nB = cornersB[0][0][1] - mB * cornersB[0][0][0]

    radiuses = [distance(center, cornersA[0][0]), distance(center, cornersA[1][0]), distance(center, cornersB[0][0]), distance(center, cornersB[1][0])]
    inner_radius = min(radiuses)
    outer_radius = max(radiuses)

    for y in range(len(prosthetic)):
        if y > center[1]:
            continue
        for x in range(len(prosthetic[y])):
            if y < mA * x + nA:
                if y < mB * x + nB:
                    dist = distance(center, [x, y])
                    if dist > inner_radius:
                        if dist < outer_radius:
                            prosthetic[y][x] = [255, 255, 255]
            

    fixed = cv2.add(copy.copy(), prosthetic.copy())
    return image, prosthetic, fixed

def preprocess(image_path):
    """
    Realiza o pré-processamento da imagem.

    :param image_path: Caminho da imagem.  
    :return: Imagem pré-processada.  
    """
    # image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = image_path
    
    # Aplicar filtro de suavização para reduzir o ruído
    image_smoothed = cv2.GaussianBlur(image, (5, 5), 0)
    
    # Aplicar limiarização para segmentar os ossos
    _, image_thresholded = cv2.threshold(image_smoothed, 150, 255, cv2.THRESH_BINARY)
    
    # Realizar operações morfológicas para remover ruídos e preencher lacunas
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    image_closed = cv2.morphologyEx(image_thresholded, cv2.MORPH_CLOSE, kernel)
    
    # Remover objetos indesejados através da abertura
    image_opened = cv2.morphologyEx(image_closed, cv2.MORPH_OPEN, kernel)
    
    # Realizar correção de intensidade usando equalização de histograma
    image_equalized = cv2.equalizeHist(image_opened)
    
    return image_equalized

def reconstruct_3d_model(slices, thresh= 0.5, spacing=(1.0, 1.0, 1.0)):
    """
    Faz a reconstrução do modelo 3D a partir de uma lista de imagens

    slices = Lista de imagens
    thresh = Valor de limiar para segmentação
    spacing = Espaçamento entre os pixels (M, N, P)
    """
    volume = np.stack(slices, axis=0)

    segmented = volume > thresh

    # Extrai a superfície da região segmentada
    verts, faces, _, _ = measure.marching_cubes(segmented, spacing=spacing, method="lorensen")

    # Cria um objeto para os pontos do modelo 3D
    fig = mlab.figure(bgcolor=(1, 1, 1), size=(800, 600))

    # Melhora a qualidade da superfície
    glyphs = mlab.triangular_mesh(verts[:, 0], verts[:, 1], verts[:, 2], faces)
    glyphs.actor.property.interpolation = 'phong'
    glyphs.actor.property.backface_culling = True

    # Atualiza a figura com o novo objeto
    fig = mlab.gcf()

    mlab.view(azimuth=45, elevation=30)
    mlab.show()

    # Retorna os pontos e as faces do modelo 3D
    return fig.children[0].mlab_source.points, fig.children[0].mlab_source.triangles

def main():
    files = [f"assets/{file}" for file in os.listdir("assets")]
    files.sort()
    images = [cv2.imread(file) for file in files]

    # 5, 9

    start_image = 2 #2
    end_image = 22 #22

    prosthetic_slice = []
    for index, image in enumerate(images[start_image:end_image]):
        index += start_image

        ind = index + 1
        pretty_index = str(ind) if ind > 9 else f"0{ind}"

        image = image[30:542, 112:594]

        half_printed, half_prosthetic, half_fixed = reconstruct_horizontal_flip(image.copy(), index)
        printed, prosthetic, fixed = reconstruct_top_circle(half_fixed.copy(), half_prosthetic.copy(), index)

        result = np.hstack((image, fixed))

        cv2.imshow(f"image_{pretty_index}", result)
        cv2.waitKey(0)

        cv2.imwrite(f"output/slice_{pretty_index}.png", result)
        cv2.imwrite(f"prosthetic/slice_{pretty_index}.png", prosthetic)

    # Cria os slices do crânio
    skull_slices = []
    for path in files:
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        image = preprocess(image[30:542, 112:594])
        skull_slices.append(image)

    # Cria os slices da prótese
    prosthetic_slice = []
    for path in os.listdir("prosthetic"):
        image = cv2.imread(f"prosthetic/{path}", cv2.IMREAD_GRAYSCALE)
        image = preprocess(image)
        prosthetic_slice.append(image)

    # Reconstrói o modelo 3D do crânio e da prótese
    skull_verts, skull_faces = reconstruct_3d_model(skull_slices, spacing=(2.0, 1.0, 1.0))
    prosthetic_verts, prosthetic_faces = reconstruct_3d_model(prosthetic_slice, spacing=(2.0, 1.0, 1.0))
    
    # Salva os modelos 3D em arquivos STL
    meshio.write_points_cells("model/model.stl", skull_verts, {"triangle": skull_faces})
    meshio.write_points_cells("model/prosthetic.stl", prosthetic_verts, {"triangle": prosthetic_faces})

 
if __name__ == "__main__":
    main()