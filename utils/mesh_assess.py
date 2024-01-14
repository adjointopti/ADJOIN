import torch.nn.functional as F
import torch


def mesh_assess_obj(obj_dir1, obj_dir2):
    vertices1 = read_vertices_from_dir(obj_dir1)
    vertices2 = read_vertices_from_dir(obj_dir2)
    return mesh_assess(vertices1, vertices2)


def mesh_assess(vertices1, vertices2):
    if vertices1.shape != vertices2.shape:
        raise Exception('The meshes should have the same shape!!')
    l1 = F.l1_loss(vertices1, vertices2, reduction='mean')
    l2 = F.mse_loss(vertices1, vertices2, reduction='mean')
    return l1, l2


def read_vertices_from_dir(obj_dir):
    lines = [l.strip() for l in open(obj_dir)]
    V = []
    for l in lines:
        words = [w for w in l.split(' ') if w != '']
        if len(words) == 0:
            continue
        if words[0] == 'v':
            V.append([float(words[1]), float(words[2]), float(words[3])])
    return torch.tensor(V)
