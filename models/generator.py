"""
Author: Eckert ZHANG
Date: 2021-11-16 01:34:42
LastEditTime: 2021-11-16 21:28:35
LastEditors: Eckert ZHANG
FilePath: /Texture_code_v1.6/models/generator.py
Description: 
"""
import cv2
import torch
import torch.nn as nn
from PIL import Image
import os
import numpy as np
from pytorch3d.renderer.blending import softmax_rgb_blend, sigmoid_alpha_blend
from pytorch3d.renderer import (OpenGLPerspectiveCameras, PointLights,
                                get_world_to_view_transform, DirectionalLights,
                                Materials, RasterizationSettings, MeshRenderer,
                                MeshRasterizer, SoftSilhouetteShader,
                                phong_shading)
from tqdm import tqdm_notebook
import matplotlib.pyplot as plt
import matplotlib as mpl
import torch.nn.functional as F
from pytorch3d.renderer.utils import TensorProperties, convert_to_tensors_and_broadcast
from pytorch3d.transforms import Rotate, Transform3d, Translate
from typing import NamedTuple, Sequence


class Generator(torch.nn.Module):
    ## Render the image from a given mesh and world2cam matrix
    def __init__(self, opt):
        super(Generator, self).__init__()
        self.opt = opt
        self.device = opt.device
        self.intrinsic = np.loadtxt(opt.input_dir + '/intrinsic.txt')
        self.intrinsic = np.reshape(self.intrinsic, [16])
        if os.path.exists(opt.input_dir + '/intrinsic_depth.txt'):
            self.render_d = True
            self.intrinsic_d = np.loadtxt(opt.input_dir +
                                          '/intrinsic_depth.txt')
            self.intrinsic_d = np.reshape(self.intrinsic_d, [16])
            try:
                self.height_d = self.opt.hw_d['height']
                self.width_d = self.opt.hw_d['width']
            except:
                depth_src = np.load(opt.input_dir +
                                    '/00000_depth.npz')['arr_0']
                self.height_d, self.width_d = depth_src.shape
        else:
            self.render_d = False

    def forward(self, input_w2c, new_meshes):
        width = self.opt.hw['width']
        height = self.opt.hw['height']
        b = input_w2c.shape[0]
        if self.opt.rotation_z:
            Rz = torch.tensor(
                [[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                dtype=torch.float32)
            Rz = Rz.view(1, 4, 4).repeat(b, 1, 1).to(self.device)
            input_w2c = torch.bmm(Rz, input_w2c)
        R = input_w2c[:, :3, :3].transpose(1, 2)
        T = input_w2c[:, :3, 3]
        # R.register_hook(print)

        # camera intrinsic
        edge_len = max(width, height)
        cameras = OpenGLRealPerspectiveCameras(
            device=self.device,
            focal_length=self.intrinsic[0],
            principal_point=((self.intrinsic[2], self.intrinsic[6]), ),
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
        renderer = MeshRenderer(rasterizer=MeshRasterizer(
            cameras=cameras, raster_settings=raster_settings),
                                shader=SimpleShader(device=self.device,
                                                    zbuf_output=True))
        images, zbufs = renderer(new_meshes.extend(b))
        del renderer, raster_settings

        # Silhouette renderer
        sigma = 1e-5
        raster_settings_soft = RasterizationSettings(
            image_size=edge_len,
            blur_radius=np.log(1. / 1e-4 - 1.) * sigma,
            faces_per_pixel=20,
        )
        renderer_silhouette = MeshRenderer(rasterizer=MeshRasterizer(
            cameras=cameras, raster_settings=raster_settings_soft),
                                           shader=SoftSilhouetteShader())
        silhouette_images = renderer_silhouette(new_meshes.extend(b))
        if width < height:
            silhouette_images = silhouette_images[:, :, height - width:, :]
        elif width > height:
            silhouette_images = silhouette_images[:, :height, :, :]
        silhouette_images = silhouette_images[:, ..., 3:4]
        del renderer_silhouette, cameras, raster_settings_soft

        # images.register_hook(print)
        if width < height:
            images = images[:, :, height - width:, :]
            zbufs = zbufs[:, :, height - width:, :]
        elif width > height:
            images = images[:, :height, :, :]
            zbufs = zbufs[:, :height, :, :]

        if self.render_d:
            del zbufs
            edge_len_d = max(self.height_d, self.width_d)
            cameras = OpenGLRealPerspectiveCameras(
                device=self.device,
                focal_length=self.intrinsic_d[0],
                principal_point=((self.intrinsic_d[2], self.intrinsic_d[6]), ),
                R=R,
                T=T,
                w=edge_len_d,
                h=edge_len_d,
                x0=0,
                y0=0)
            raster_settings = RasterizationSettings(
                image_size=edge_len_d,
                blur_radius=0.0,
                faces_per_pixel=1,
            )
            renderer = MeshRenderer(rasterizer=MeshRasterizer(
                cameras=cameras, raster_settings=raster_settings),
                                    shader=SimpleShader(device=self.device,
                                                        zbuf_output=True))
            images_d, zbufs = renderer(new_meshes.extend(b))
            if self.width_d < self.height_d:
                images_d = images_d[:, :, self.height_d - self.width_d:, :]
                zbufs = zbufs[:, :, self.height_d - self.width_d:, :]
            elif self.width_d > self.height_d:
                images_d = images_d[:, :self.height_d, :, :]
                zbufs = zbufs[:, :self.height_d, :, :]
            mask_d = images_d[:, ..., 3:4].clone().detach()
            mask_d[mask_d != 0] = 1
            del renderer, cameras, raster_settings, images_d

        # silhouette_images.register_hook(print)

        # for ii in range(images.shape[0]):
        #     plt.figure()
        #     plt.imshow(images[ii, ..., :3].cpu().detach().numpy())
        #     plt.axis('off')
        #     plt.show()
        #
        # plt.figure()
        # plt.imshow(silhouette_images[ii, ..., 0].cpu().detach().numpy())
        # plt.axis('off')
        # plt.show()

        color_render = (images[:, ..., :3] * 2.0 - 1.0)
        mask_render = images[:, ..., 3:4].clone().detach()
        mask_render[mask_render != 0] = 1
        for i in range(3):
            color_render[..., i] *= mask_render[..., 0]
        if not self.render_d:
            mask_d = mask_render
        zbufs[..., 0] *= mask_d[..., 0]

        return color_render.permute(0, 3, 1, 2), mask_render.permute(
            0, 3, 1, 2), silhouette_images.permute(
                0, 3, 1, 2), zbufs[..., 0], mask_d.permute(0, 3, 1,
                                                           2)  # BCHW * 2 + BHW

    @staticmethod
    def save_texture(path, iteration, texture_map, n_round='none'):
        tmp = texture_map.clone()
        save_tex_path = os.path.join(path, 'training_textures')
        if not os.path.exists(save_tex_path):
            os.makedirs(save_tex_path)
        if n_round != 'none':
            cv2.imwrite(
                os.path.join(save_tex_path,
                             str(n_round) + '_' + str(iteration) + '.png'),
                cv2.cvtColor(
                    np.clip(tmp.cpu().detach().numpy(), 0, 1) * 255.,
                    cv2.COLOR_RGB2BGR))
        else:
            cv2.imwrite(
                os.path.join(save_tex_path,
                             str(iteration) + '.png'),
                cv2.cvtColor(
                    np.clip(tmp.cpu().detach().numpy(), 0, 1) * 255.,
                    cv2.COLOR_RGB2BGR))

    def save_mesh(self, path, iteration, new_meshes, n_round='none'):
        save_mesh_path = os.path.join(path, 'training_mesh')
        if not os.path.exists(save_mesh_path):
            os.makedirs(save_mesh_path)
        if n_round != 'none':
            save_objs(os.path.join(
                save_mesh_path,
                str(n_round) + '_' + str(iteration) + '.obj'),
                      self.opt.input_dir.split('/')[-1],
                      new_meshes.verts_packed(),
                      new_meshes.faces_packed(),
                      vnormal=new_meshes.verts_normals_packed(),
                      vtex=new_meshes.textures.verts_uvs_list()[0],
                      ftex=new_meshes.textures.faces_uvs_list()[0])
        else:
            save_objs(os.path.join(save_mesh_path,
                                   str(iteration) + '.obj'),
                      self.opt.input_dir.split('/')[-1],
                      new_meshes.verts_packed(),
                      new_meshes.faces_packed(),
                      vnormal=new_meshes.verts_normals_packed(),
                      vtex=new_meshes.textures.verts_uvs_list()[0],
                      ftex=new_meshes.textures.faces_uvs_list()[0])

    # def save_texture(self, path, iteration):
    #     tmp = self.textures[0, :, :, :].clone()
    #     if not os.path.exists(os.path.join(path, 'training_textures')):
    #         os.makedirs(os.path.join(path, 'training_textures'))
    #
    #     RGB_img = (tmp.permute(1, 2, 0).detach().cpu().numpy() + 1) / 2 * 255.
    #     cv2.imwrite(os.path.join(path, 'training_textures', str(iteration) + '.png'),
    #                 cv2.cvtColor(RGB_img, cv2.COLOR_RGB2BGR))  ## convert RGB to BGR


## ------------------------------------------------------------
## ------------------Self-defined Functions--------------------
## ------------------------------------------------------------

# def get_textures(texture_path):
#     texture_img = cv2.imread(texture_path, cv2.IMREAD_COLOR)
#     texture_img = texture_img[..., ::-1].copy()
#     texture_arr = np.array(texture_img)
#     textures = texture_arr.transpose(2, 0, 1)
#     texture_tensor = torch.tensor(textures).float()
#
#     ######################################################################
#     texture_tensor /= 255.  ## From [0,255] to [0,1]
#     texture_tensor = (texture_tensor - 0.5) / 0.5  ## From [0,1] to [-1,1]
#     ######################################################################
#     return texture_tensor.float().unsqueeze(0)  # 1*C*H*W

# def get_texture(texture_path):
#     texture_img = cv2.imread(texture_path, cv2.IMREAD_COLOR)
#     texture_img = texture_img[..., ::-1].copy()
#     texture = torch.tensor(texture_img) / 255.  ## From [0,255] to [0,1]
#     return texture


def save_objs(new_mesh_file, obj_name, verts, faces, vnormal, vtex, ftex):
    file = open(new_mesh_file, 'w')
    V0 = np.array(verts.cpu().detach(), dtype='float32')
    VN = np.array(vnormal.cpu().detach(), dtype='float32')
    VT = np.array(vtex.cpu().detach(), dtype='float32')
    F1 = np.array(faces.cpu().detach(), dtype='int32')
    FT = np.array(ftex.cpu().detach(), dtype='int32')
    lines_new = ""
    lines_new += '# Generated.\n'
    lines_new += 'mtllib %s.obj.mtl\n' % obj_name
    if len(V0):
        for i in range(V0.shape[0]):
            vert = ['%.6f' % V0[i, j] for j in range(V0.shape[1])]  #425
            normal = ['%.6f' % VN[i, j] for j in range(VN.shape[1])]
            lines_new += 'v %s\n' % ' '.join(vert)
            lines_new += 'vn %s\n' % ' '.join(normal)
        for i in range(VT.shape[0]):
            vert_t = ['%.6f' % VT[i, j] for j in range(VT.shape[1])]
            lines_new += 'vt %s\n' % ' '.join(vert_t)
    lines_new += 'usemtl material_0\n'
    if len(F1):
        F1 += np.ones(F1.shape, dtype='int32')
        FT += np.ones(FT.shape, dtype='int32')
        for i in range(F1.shape[0]):
            lines_new += 'f %d/%d/%d %d/%d/%d %d/%d/%d\n' % (
                F1[i, 0], FT[i, 0], F1[i, 0], F1[i, 1], FT[i, 1], F1[i, 1],
                F1[i, 2], FT[i, 2], F1[i, 2])
        # lines_new += 'f %s\n' % ' '.join(F[i])
    lines_new += '# End of file.\n'
    file.write(lines_new)
    file.close()


class SimpleShader(nn.Module):  ## A Simple Shader without Light setting
    def __init__(self,
                 device='cpu',
                 cameras=None,
                 materials=None,
                 blend_params=None,
                 zbuf_output=False):
        super().__init__()
        self.materials = (materials if materials is not None else Materials(
            device=device))
        self.cameras = cameras
        self.zbuf_output = zbuf_output
        self.blend_params = blend_params if blend_params is not None else _BlendParams(
        )

    def forward(self, fragments, meshes, **kwargs) -> torch.Tensor:
        blend_params = kwargs.get("blend_params", self.blend_params)
        pixel_colors = meshes.sample_textures(fragments)
        images = softmax_rgb_blend(pixel_colors, fragments, blend_params)
        # fragments.zbuf.register_hook(print)
        # images = sigmoid_alpha_blend(pixel_colors, fragments, blend_params)
        if self.zbuf_output:
            return images, fragments.zbuf
        else:
            return images


class _BlendParams(NamedTuple):
    sigma: float = 1e-5
    gamma: float = 1e-4
    background_color: Sequence = (0.0, 0.0, 0.0)


class SoftPhongShader(nn.Module):
    """
    Per pixel lighting - the lighting model is applied using the interpolated
    coordinates and normals for each pixel. The blending function returns the
    soft aggregated color using all the faces per pixel.

    To use the default values, simply initialize the shader with the desired
    device e.g.

    .. code-block::

        shader = SoftPhongShader(device=torch.device("cuda:0"))
    """
    def __init__(self,
                 device="cpu",
                 cameras=None,
                 lights=None,
                 materials=None,
                 blend_params=None,
                 zbuf_output=False):
        super().__init__()
        self.lights = lights if lights is not None else PointLights(
            device=device)
        self.materials = (materials if materials is not None else Materials(
            device=device))
        self.cameras = cameras
        self.blend_params = blend_params if blend_params is not None else _BlendParams(
        )
        self.zbuf_output = zbuf_output

    def to(self, device):
        # Manually move to device modules which are not subclasses of nn.Module
        self.cameras = self.cameras.to(device)
        self.materials = self.materials.to(device)
        self.lights = self.lights.to(device)

    def forward(self, fragments, meshes, **kwargs) -> torch.Tensor:
        cameras = kwargs.get("cameras", self.cameras)
        if cameras is None:
            msg = "Cameras must be specified either at initialization \
                or in the forward pass of SoftPhongShader"

            raise ValueError(msg)

        texels = meshes.sample_textures(fragments)
        lights = kwargs.get("lights", self.lights)
        materials = kwargs.get("materials", self.materials)
        blend_params = kwargs.get("blend_params", self.blend_params)
        colors = phong_shading(
            meshes=meshes,
            fragments=fragments,
            texels=texels,
            lights=lights,
            cameras=cameras,
            materials=materials,
        )
        znear = kwargs.get("znear", getattr(cameras, "znear", 1.0))
        zfar = kwargs.get("zfar", getattr(cameras, "zfar", 100.0))
        images = softmax_rgb_blend(colors,
                                   fragments,
                                   blend_params,
                                   znear=znear,
                                   zfar=zfar)
        if self.zbuf_output:
            return images, fragments.zbuf
        else:
            return images


r = np.expand_dims(np.eye(3), axis=0)  # (1, 3, 3)
t = np.expand_dims(np.zeros(3), axis=0)


class OpenGLRealPerspectiveCameras(TensorProperties):
    """
    A class which stores a batch of parameters to generate a batch of
    projection matrices using the OpenGL convention for a perspective camera.

    The extrinsics of the camera (R and T matrices) can also be set in the
    initializer or passed in to `get_full_projection_transform` to get
    the full transformation from world -> screen.

    The `transform_points` method calculates the full world -> screen transform
    and then applies it to the input points.

    The transforms can also be returned separately as Transform3d objects.
    """
    def __init__(
        self,
        focal_length=1.0,
        principal_point=((0.0, 0.0), ),
        R=r,
        T=t,
        znear=1,
        zfar=100.0,
        x0=0,
        y0=0,
        w=640,
        h=480,
        device="cpu",
    ):
        """
        __init__(self, znear, zfar, R, T, device) -> None  # noqa

        Args:
            znear: near clipping plane of the view frustrum.
            zfar: far clipping plane of the view frustrum.
            R: Rotation matrix of shape (N, 3, 3)
            T: Translation matrix of shape (N, 3)
            device: torch.device or string
        """
        # The initializer formats all inputs to torch tensors and broadcasts
        # all the inputs to have the same batch dimension where necessary.
        super().__init__(
            device=device,
            focal_length=focal_length,
            principal_point=principal_point,
            R=R,
            T=T,
            znear=znear,
            zfar=zfar,
            x0=x0,
            y0=y0,
            h=h,
            w=w,
        )

    def get_projection_transform(self, **kwargs) -> Transform3d:
        """
        Calculate the OpenGL perpective projection matrix with a symmetric
        viewing frustrum. Use column major order.

        Args:
            **kwargs: parameters for the projection can be passed in as keyword
                arguments to override the default values set in `__init__`.

        Return:
            P: a Transform3d object which represents a batch of projection
            matrices of shape (N, 3, 3)

        .. code-block:: python
            q = -(far + near)/(far - near)
            qn = -2*far*near/(far-near)

            P.T = [
                    [2*fx/w,     0,           0,  0],
                    [0,          -2*fy/h,     0,  0],
                    [(2*px-w)/w, (-2*py+h)/h, -q, 1],
                    [0,          0,           qn, 0],
                ]
                sometimes P[2,:] *= -1, P[1, :] *= -1
        """
        znear = kwargs.get("znear", self.znear)  # pyre-ignore[16]
        zfar = kwargs.get("zfar", self.zfar)  # pyre-ignore[16]
        x0 = kwargs.get("x0", self.x0)  # pyre-ignore[16]
        y0 = kwargs.get("y0", self.y0)  # pyre-ignore[16]
        w = kwargs.get("w", self.w)  # pyre-ignore[16]
        h = kwargs.get("h", self.h)  # pyre-ignore[16]
        principal_point = kwargs.get("principal_point",
                                     self.principal_point)  # pyre-ignore[16]
        focal_length = kwargs.get("focal_length",
                                  self.focal_length)  # pyre-ignore[16]

        if not torch.is_tensor(focal_length):
            focal_length = torch.tensor(focal_length, device=self.device)

        if len(focal_length.shape) in (0, 1) or focal_length.shape[1] == 1:
            fx = fy = focal_length
        else:
            fx, fy = focal_length.unbind(1)

        if not torch.is_tensor(principal_point):
            principal_point = torch.tensor(principal_point, device=self.device)
        px, py = principal_point.unbind(1)

        P = torch.zeros((self._N, 4, 4),
                        device=self.device,
                        dtype=torch.float32)
        ones = torch.ones((self._N), dtype=torch.float32, device=self.device)

        # NOTE: In OpenGL the projection matrix changes the handedness of the
        # coordinate frame. i.e the NDC space postive z direction is the
        # camera space negative z direction. This is because the sign of the z
        # in the projection matrix is set to -1.0.
        # In pytorch3d we maintain a right handed coordinate system throughout
        # so the so the z sign is 1.0.
        z_sign = 1.0
        # # define P.T directly
        # P[:, 0, 0] = 2.0 * fx / w
        # P[:, 1, 1] = -2.0 * fy / h
        # P[:, 2, 0] = -(-2 * px + w + 2 * x0) / w
        # P[:, 2, 1] = -(+2 * py - h + 2 * y0) / h
        # P[:, 2, 3] = z_sign * ones
        # define P.T directly #### Edited by Eckert
        P[:, 0, 0] = 2.0 * fx / w
        P[:, 1, 1] = 2.0 * fy / h
        P[:, 2, 0] = (1.0 - 2.0 * (px - x0) / (w - 1.0)) * -1
        P[:, 2, 1] = (2.0 * (py - y0) / (h - 1.0) - 1.0) * -1
        # P[:, 2, 0] = 1.0 - 2.0 * (px - x0) / w
        # P[:, 2, 1] = 2.0 * (py - y0) / h - 1.0
        P[:, 2, 3] = z_sign * ones

        # NOTE: This part of the matrix is for z renormalization in OpenGL
        # which maps the z to [-1, 1]. This won't work yet as the torch3d
        # rasterizer ignores faces which have z < 0.
        # P[:, 2, 2] = z_sign * (far + near) / (far - near)
        # P[:, 2, 3] = -2.0 * far * near / (far - near)
        # P[:, 2, 3] = z_sign * torch.ones((N))

        # NOTE: This maps the z coordinate from [0, 1] where z = 0 if the point
        # is at the near clipping plane and z = 1 when the point is at the far
        # clipping plane. This replaces the OpenGL z normalization to [-1, 1]
        # until rasterization is changed to clip at z = -1.
        P[:, 2, 2] = z_sign * zfar / (zfar - znear)
        P[:, 3, 2] = -(zfar * znear) / (zfar - znear)

        # print('z_sign = ', z_sign)
        # print('Projection Matrix:\n',P.transpose(1,2))

        # OpenGL uses column vectors so need to transpose the projection matrix
        # as torch3d uses row vectors.
        transform = Transform3d(device=self.device)
        transform._matrix = P
        return transform

    def clone(self):
        other = OpenGLRealPerspectiveCameras(device=self.device)
        return super().clone(other)

    def get_camera_center(self, **kwargs):
        """
        Return the 3D location of the camera optical center
        in the world coordinates.

        Args:
            **kwargs: parameters for the camera extrinsics can be passed in
                as keyword arguments to override the default values
                set in __init__.

        Setting T here will update the values set in init as this
        value may be needed later on in the rendering pipeline e.g. for
        lighting calculations.

        Returns:
            C: a batch of 3D locations of shape (N, 3) denoting
            the locations of the center of each camera in the batch.
        """
        w2v_trans = self.get_world_to_view_transform(**kwargs)
        P = w2v_trans.inverse().get_matrix()

        # the camera center is the translation component (the first 3 elements
        # of the last row) of the inverted world-to-view
        # transform (4x4 RT matrix)
        C = P[:, 3, :3]
        # print('The internal camera ceter:\n',C)
        return C

    def get_world_to_view_transform(self, **kwargs) -> Transform3d:
        """
        Return the world-to-view transform.

        Args:
            **kwargs: parameters for the camera extrinsics can be passed in
                as keyword arguments to override the default values
                set in __init__.

        Setting R and T here will update the values set in init as these
        values may be needed later on in the rendering pipeline e.g. for
        lighting calculations.

        Returns:
            T: a Transform3d object which represents a batch of transforms
            of shape (N, 3, 3)
        """
        R = self.R = kwargs.get("R", self.R)  # pyre-ignore[16]
        T = self.T = kwargs.get("T", self.T)  # pyre-ignore[16]
        if T.shape[0] != R.shape[0]:
            msg = "Expected R, T to have the same batch dimension; got %r, %r"
            raise ValueError(msg % (R.shape[0], T.shape[0]))
        if T.dim() != 2 or T.shape[1:] != (3, ):
            msg = "Expected T to have shape (N, 3); got %r"
            raise ValueError(msg % repr(T.shape))
        if R.dim() != 3 or R.shape[1:] != (3, 3):
            msg = "Expected R to have shape (N, 3, 3); got %r"
            raise ValueError(msg % R.shape)

        # Create a Transform3d object
        T = Translate(T, device=T.device)
        R = Rotate(R, device=R.device)
        world_to_view_transform = R.compose(T)
        return world_to_view_transform

    def get_full_projection_transform(self, **kwargs) -> Transform3d:
        """
        Return the full world-to-screen transform composing the
        world-to-view and view-to-screen transforms.

        Args:
            **kwargs: parameters for the projection transforms can be passed in
                as keyword arguments to override the default values
                set in __init__.

        Setting R and T here will update the values set in init as these
        values may be needed later on in the rendering pipeline e.g. for
        lighting calculations.

        Returns:
            T: a Transform3d object which represents a batch of transforms
            of shape (N, 3, 3)
        """
        self.R = kwargs.get("R", self.R)  # pyre-ignore[16]
        self.T = kwargs.get("T", self.T)  # pyre-ignore[16]

        world_to_view_transform = self.get_world_to_view_transform(R=self.R,
                                                                   T=self.T)
        view_to_screen_transform = self.get_projection_transform(**kwargs)
        return world_to_view_transform.compose(view_to_screen_transform)

    def transform_points(self, points, **kwargs) -> torch.Tensor:
        """
        Transform input points from world to screen space.

        Args:
            points: torch tensor of shape (..., 3).

        Returns
            new_points: transformed points with the same shape as the input.
        """
        world_to_screen_transform = self.get_full_projection_transform(
            **kwargs)
        return world_to_screen_transform.transform_points(points)
