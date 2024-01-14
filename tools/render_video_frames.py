"""
Author: Eckert ZHANG
Date: 2021-11-16 01:35:26
LastEditTime: 2023-01-16 01:04:12
LastEditors: Eckert ZHANG
FilePath: /Texture_code_v1.6/tools/render_video_frames.py
Description: 
"""
import glob, argparse, os, torch, math, sys, pdb
import numpy as np
import matplotlib.pyplot as plt
import PIL
from pytorch3d.io import load_ply_rgb, load_objs_as_meshes
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    PointLights,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    # SoftPhongShader,
    TexturesVertex)
import fresnel
from math import sqrt, acos, pi
import matplotlib, matplotlib.cm

libpath = os.path.dirname(os.path.abspath(__file__))
sys.path.append(libpath + '/..')
sys.path.insert(1, '..')
from models.generator import OpenGLRealPerspectiveCameras, SimpleShader, SoftPhongShader


def vect_normalized(vect):
    dim_vect = len(vect)
    sum_square = 0
    for ii in range(dim_vect):
        sum_square += vect[ii]**2
    magnitude = sqrt(sum_square)
    vect_norm = vect / magnitude
    return vect_norm


def vect_length(vect):
    dim_vect = len(vect)
    sum_square = 0
    for ii in range(dim_vect):
        sum_square += vect[ii]**2
    magnitude = sqrt(sum_square)
    return magnitude


def gen_fresnel_mesh(verts, faces):
    '''
    The mesh geometry defines a generic triangle mesh. 
    Define a mesh with an 3Tx3 array where T is the number of triangles. 
    Triangles must be specified with a counter clockwise winding.
    '''
    fresnel_mesh = []
    num_f = faces.shape[0]
    for ii in range(num_f):
        ff = faces[ii]
        for jj in range(3):
            vert = verts[ff[jj]]
            fresnel_mesh.append(vert)
    return np.array(fresnel_mesh)


parser = argparse.ArgumentParser()
parser.add_argument(
    "--mesh_dir",
    default=
    '/home/jingbo/Results_texture_opti/ply_intrinsic3d/M20/194725/results/intrinsic3d/mesh_g0_p0.ply',
    type=str)
parser.add_argument("--mesh_stand_dir", default=None, type=str)
parser.add_argument("--name", default='194725', type=str)
parser.add_argument("--type", default='object', type=str)  # object or scene
parser.add_argument("--peak_point",
                    default=[0.00265, -0.09833, 0.33866],
                    nargs='+',
                    type=float)
parser.add_argument("--fore_point",
                    default=[0.00973, 0.03197, 0.36692],
                    nargs='+',
                    type=float)
parser.add_argument("--right_point",
                    default=[0.00973, 0.03197, 0.36692],
                    nargs='+',
                    type=float)
parser.add_argument("--frames", default=200, type=int)
parser.add_argument("--save_dir", default='./video', type=str)
parser.add_argument("--method", default='Intrinsic3d', type=str)
opt = parser.parse_args()
opt.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# pdb.set_trace()
opt.peak_point = np.array(opt.peak_point)
opt.fore_point = np.array(opt.fore_point)
opt.right_point = np.array(opt.right_point)

name = opt.name  #opt.mesh_dir.split('/')[-1].split('.')[0]
mesh_type = opt.mesh_dir.split('/')[-1].split('.')[-1]
output_dir = os.path.join(opt.save_dir, name + '/' + opt.method)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

if mesh_type == 'obj':
    mesh = load_objs_as_meshes([opt.mesh_dir],
                               device=opt.device,
                               load_textures=True)
    verts = mesh.verts_list()[0]
    faces = mesh.faces_list()[0]
elif mesh_type == 'ply':
    ply = load_ply_rgb(opt.mesh_dir)
    verts = ply[0]
    faces = ply[1]
    features = torch.from_numpy(ply[2])
    tex = TexturesVertex(verts_features=[features])
    mesh = Meshes(verts=[verts], faces=[faces], textures=tex).to(opt.device)
else:
    raise Exception('The type of mesh is wrong!')

# -------------------------------------------------------
# fresnel_mesh = gen_fresnel_mesh(verts.cpu().numpy(), faces.cpu().numpy())
# scene1 = fresnel.Scene()
# scene1.lights = fresnel.light.cloudy()
# geometry = fresnel.geometry.Mesh(scene1, vertices=fresnel_mesh, N=1)
# mapper = matplotlib.cm.ScalarMappable(norm = matplotlib.colors.Normalize(vmin=-0.08, vmax=0.05, clip=True),
#                                       cmap = matplotlib.cm.get_cmap(name='viridis'))
# geometry.color[:] = fresnel.color.linear(mapper.to_rgba(fresnel_mesh[:,1]))
# geometry.material = fresnel.material.Material(color=fresnel.color.linear([0.25,0.5,0.9]),
#                                                   roughness=0.8,
#                                                   specular=0.5,
#                                                   primitive_color_mix=1.0,
#                                                   solid=1)
# geometry.outline_material = fresnel.material.Material(
#         color=fresnel.color.linear([0, 1.0, 0]),
#         roughness=0.8,
#         specular=0.5,
#         primitive_color_mix=0.0,
#         solid=1)
# geometry.outline_width = 0.0001
# -------------------------------------------------------

if opt.type == 'object':
    if opt.mesh_stand_dir is None:
        raise Exception('For object model, mesh_stand_dir is needed!')
    else:
        mesh_stand = load_objs_as_meshes([opt.mesh_stand_dir],
                                         device=opt.device,
                                         load_textures=True)
        verts = mesh_stand.verts_list()[0]

positions = []
if opt.type == 'object':
    verts_mean = verts.mean(dim=0).cpu().numpy()
    up_direct = vect_normalized(opt.peak_point - verts_mean)
    fore_direct0 = opt.fore_point - verts_mean
    fore_direct = vect_normalized(fore_direct0 -
                                  np.dot(up_direct, fore_direct0) * up_direct)
    au_direct = np.cross(up_direct, fore_direct)
    # find the farmost point
    dist = sqrt(np.sum(
        (verts.cpu().numpy() - verts_mean)**2, axis=1).max()) * 2
    # genearte camera_poses
    delta_angle = 2 * pi / opt.frames
    for ii in range(opt.frames):
        ang = delta_angle * ii
        pos = verts_mean + dist * (math.cos(ang) * fore_direct +
                                   math.sin(ang) * au_direct)
        positions.append(pos)
elif opt.type == 'scene':
    if opt.right_point is None:
        raise Exception('The right point is needed!')
    else:
        verts_mean = verts.mean(dim=0).cpu().numpy()
        # dist = sqrt(np.sum((verts.cpu().numpy()-verts_mean)**2, axis = 1).max()) * 2
        dist = vect_length(opt.peak_point - opt.fore_point) * 2
        up_direct = vect_normalized(opt.peak_point - opt.fore_point)
        right_direct = vect_normalized(opt.right_point - opt.fore_point)
        length_line = vect_length(opt.right_point - opt.fore_point)
        au_direct = np.cross(right_direct, up_direct)
        delta_line = length_line / opt.frames
        positions_at = []
        for ii in range(opt.frames):
            lenth = delta_line * ii
            pos = opt.fore_point + dist * au_direct + lenth * right_direct
            positions_at.append(opt.fore_point + lenth * right_direct)
            positions.append(pos)

# rendering
for ii in range(opt.frames):
    if opt.type == 'scene':
        verts_mean = positions_at[ii]
    R, T = look_at_view_transform(dist=dist,
                                  eye=torch.tensor(positions[ii],
                                                   dtype=torch.float32)[None],
                                  at=torch.tensor(verts_mean,
                                                  dtype=torch.float32)[None],
                                  up=torch.tensor(up_direct,
                                                  dtype=torch.float32)[None])
    edge_len = 640
    cameras = OpenGLRealPerspectiveCameras(device=opt.device,
                                           focal_length=496,
                                           principal_point=((320, 320), ),
                                           R=R,
                                           T=T,
                                           w=edge_len,
                                           h=edge_len,
                                           x0=0,
                                           y0=0)
    raster_settings = RasterizationSettings(
        image_size=edge_len,
        blur_radius=0.0,
        faces_per_pixel=1,
    )

    if mesh_type == 'obj':
        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(cameras=cameras,
                                      raster_settings=raster_settings),
            shader=SimpleShader(device=opt.device,
                                # zbuf_output=True
                                ))
    elif mesh_type == 'ply':
        lights = PointLights(
            location=[[0.0, 0.0, -3.0]],
            ambient_color=((1, 1, 1), ),
            diffuse_color=((0, 0, 0), ),
            specular_color=((0, 0, 0), ),
            device=opt.device,
        )
        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(cameras=cameras,
                                      raster_settings=raster_settings),
            shader=SoftPhongShader(
                device=opt.device,
                cameras=cameras,
                lights=lights,
                # zbuf_output=True
            ))

    images = renderer(mesh.extend(1))
    # images, zbufs = renderer(mesh.extend(1))
    color_render = (images[0, ..., :3] *
                    255).cpu().detach().numpy().astype('uint8')
    silhouette = images[0, ..., 3:4]
    silhouette = (torch.cat((silhouette, silhouette, silhouette), dim=2) *
                  255).cpu().detach().numpy().astype('uint8')
    mask_render = images[0, ..., 3:4].clone()
    mask_render[mask_render != 0] = 1
    mask3_r = torch.cat((mask_render, mask_render, mask_render),
                        dim=2).detach().cpu().numpy()
    # zbuf = torch.cat((zbufs[0], zbufs[0], zbufs[0]), dim=2).cpu().detach().numpy()
    # zbuf = ((np.clip(zbuf * mask3_r, 0, 1) + (1 - mask3_r))*255).astype('uint8')
    silhouette = (silhouette * mask3_r + (1 - mask3_r) * 255).astype('uint8')
    img_r_adverse = (color_render * mask3_r +
                     (1 - mask3_r) * 255).astype('uint8')

    if not os.path.exists(os.path.join(output_dir, 'color')):
        os.makedirs(os.path.join(output_dir, 'color'))
    plt.imsave(os.path.join(output_dir, 'color/' + '%05d.png' % ii),
               img_r_adverse)
    if not os.path.exists(os.path.join(output_dir, 'mesh')):
        os.makedirs(os.path.join(output_dir, 'mesh'))
    plt.imsave(os.path.join(output_dir, 'mesh/' + '%05d.png' % ii), silhouette)
# -------------------------------------------------------
# scene1.camera = fresnel.camera.orthographic(position=positions[ii],
#                                             look_at=verts_mean,
#                                             up=up_direct,
#                                             height=0.25)
# scene1.lights = fresnel.light.cloudy()
# # bb = fresnel.preview(scene1, w=640, h=640)
# aa = fresnel.pathtrace(scene1, w=640, h=640, samples=200)
# if not os.path.exists(os.path.join(output_dir, 'mesh2')):
#     os.makedirs(os.path.join(output_dir, 'mesh2'))
# PIL.Image.fromarray(aa[:], mode='RGBA').save(os.path.join(output_dir, 'mesh2/'+'%05d.png'%ii))
# -------------------------------------------------------

print('Finish frames of %s produced by %s' % (name, opt.method))
