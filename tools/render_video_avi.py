import cv2, glob, os, sys, argparse, pdb
import numpy as np
from moviepy.editor import *

libpath = os.path.dirname(os.path.abspath(__file__))
sys.path.append(libpath + '/..')

font=cv2.FONT_HERSHEY_COMPLEX#FONT_HERSHEY_SIMPLEX

def main():    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_type", default='synthetic', type=str)  # synthetic or real
    parser.add_argument("--img_type", default='color', type=str)  # color or mesh
    parser.add_argument("--data_path", default='/home/jingbo/Codes/TextureOpt/Texture_code_v1.5/video/194725', type=str)
    parser.add_argument("--save_path", default='/home/jingbo/Codes/TextureOpt/Texture_code_v1.5/video/194725', type=str)
    opt = parser.parse_args()
    data_path = opt.data_path
    save_path = opt.save_path
    if opt.data_type == 'synthetic':
        methods = ['Input', 'G2LTex', 'ATO', 'JTG', 'Ours', 'GT']
    elif opt.data_type == 'real':
        methods = ['Input', 'G2LTex', 'ATO', 'Intrinsic3d', 'JTG', 'Ours']
    else:
        raise Exception('The data_type is wrong!')
    
    img_type = opt.img_type
    name = data_path.split('/')[-1]
    imgs_dir = {}
    for method in methods:
        dirs = sorted(glob.glob(os.path.join(data_path, method+'/'+img_type+'/*.png')))
        imgs_dir[method] = dirs
    num = len(imgs_dir[method])

    fps = 30          # 视频帧率
    size = (640*3, 640*2) # 需要转为视频的图片的尺寸
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    video = cv2.VideoWriter(os.path.join(save_path, name+'_'+img_type+".avi"), cv2.VideoWriter_fourcc('I', '4', '2', '0'), fps, size)
    
    for i in range(num): 
        img1 = cv2.imread(imgs_dir[methods[0]][i])
        # pdb.set_trace()
        if name == 'Bricks' or name == 'Fountain' or name == 'Tomb_statuary' or name == 'lounge1':
            # img1[:70, ...] = 255
            # img1[70:, ...] = img1[:570, ...]
            # img1[:570, ...] = img1[70:, ...]
            # img1[:, 0:15, :] = 255
            # img1[:, 625:640, :] = 255
            img1[570:, :, :] = 255
        img1 = cv2.putText(img1,methods[0],(280,620),font,1.5,(0,0,0),4)

        img2 = cv2.imread(imgs_dir[methods[1]][i])
        if name == 'Bricks' or name == 'Fountain' or name == 'Tomb_statuary' or name == 'lounge1':
            # img2[:70, ...] = 255
            # img2[70:, ...] = img2[:570, ...]
            # img2[:570, ...] = img2[70:, ...]
            # img2[:, 0:15, :] = 255
            # img2[:, 625:640, :] = 255
            img2[570:, :, :] = 255
        img2 = cv2.putText(img2,methods[1],(280,620),font,1.5,(0,0,0),4)

        img3 = cv2.imread(imgs_dir[methods[2]][i])
        if name == 'Bricks' or name == 'Fountain' or name == 'Tomb_statuary' or name == 'lounge1':
            # img3[:70, ...] = 255
            # img3[70:, ...] = img3[:570, ...]
            # img3[:570, ...] = img3[70:, ...]
            # img3[:, 0:15, :] = 255
            # img3[:, 625:640, :] = 255
            img3[570:, :, :] = 255
        img3 = cv2.putText(img3,methods[2],(280,620),font,1.5,(0,0,0),4)

        img4 = cv2.imread(imgs_dir[methods[3]][i])
        if name == 'Bricks' or name == 'Fountain' or name == 'Tomb_statuary' or name == 'lounge1':
            # img4[:70, ...] = 255
            # img4[70:, ...] = img4[:570, ...]
            # img4[:570, ...] = img4[70:, ...]
            # img4[:, 0:15, :] = 255
            # img4[:, 625:640, :] = 255
            img4[570:, :, :] = 255
        img4 = cv2.putText(img4,methods[3],(280,620),font,1.5,(0,0,0),4)

        img5 = cv2.imread(imgs_dir[methods[4]][i])
        if name == 'Bricks' or name == 'Fountain' or name == 'Tomb_statuary' or name == 'lounge1':
            # img5[:70, ...] = 255
            # img5[70:, ...] = img5[:570, ...]
            # img5[:570, ...] = img5[70:, ...]
            # img5[:, 0:15, :] = 255
            # img5[:, 625:640, :] = 255
            img5[570:, :, :] = 255
        img5 = cv2.putText(img5,methods[4],(280,620),font,1.5,(0,0,0),4)

        img6 = cv2.imread(imgs_dir[methods[5]][i])
        if name == 'Bricks' or name == 'Fountain' or name == 'Tomb_statuary' or name == 'lounge1':
            # img6[:70, ...] = 255
            # img6[70:, ...] = img6[:570, ...]
            # img6[:570, ...] = img6[70:, ...]
            # img6[:, 0:15, :] = 255
            # img6[:, 625:640, :] = 255
            img6[570:, :, :] = 255
        img6 = cv2.putText(img6,methods[5],(280,620),font,1.5,(0,0,0),4)

        img_h1 = np.hstack((img1, img2, img3))
        img_h2 = np.hstack((img4, img5, img6))
        img = np.vstack((img_h1, img_h2))
        video.write(img)   
    video.release()
    cv2.destroyAllWindows()
    print('Finish video of %s in format  %s.' % (name, img_type))

def concat_video():
    list_name1 = ['yosee', 'weiNi', 'RedFlyDragon', 'luyiji']
    list_name2 = ['Tomb_statuary', 'Lion', '201437', '201617', 
    'chair02', 'chair14', 'chair16', 'chair21', 'lounge1', 'Bricks']
    path = '/home/jingbo/Results_texture_opti/Videos'
    title = True
    if title:
        fps = 30
        size = (640*3, 640*2)
        video = cv2.VideoWriter(os.path.join(path, '00title.avi'), cv2.VideoWriter_fourcc('I', '4', '2', '0'), fps, size)
        title_img = cv2.imread(os.path.join(path, '00title/title_video.png'))
        for i in range(70):
            video.write(title_img) 
        video.release()
        cv2.destroyAllWindows()

        video = cv2.VideoWriter(os.path.join(path, 'syn.avi'), cv2.VideoWriter_fourcc('I', '4', '2', '0'), fps, size)
        title_img = cv2.imread(os.path.join(path, '00title/syn.png'))
        for i in range(40):
            video.write(title_img) 
        video.release()
        cv2.destroyAllWindows()

        video = cv2.VideoWriter(os.path.join(path, 'real.avi'), cv2.VideoWriter_fourcc('I', '4', '2', '0'), fps, size)
        title_img = cv2.imread(os.path.join(path, '00title/real.png'))
        for i in range(40):
            video.write(title_img) 
        video.release()
        cv2.destroyAllWindows()
    L = []
    video = VideoFileClip(os.path.join(path, '00title.avi'))
    L.append(video)
    video = VideoFileClip(os.path.join(path, 'syn.avi'))
    L.append(video)
    for name in list_name1:
        video = VideoFileClip(os.path.join(path, '%s_color.avi'%name))
        L.append(video)
        # video = VideoFileClip(os.path.join(path, '%s_mesh.avi'%name))
        # L.append(video)
    video = VideoFileClip(os.path.join(path, 'real.avi'))
    L.append(video)
    for name in list_name2:
        video = VideoFileClip(os.path.join(path, '%s_color.avi'%name))
        L.append(video)
        # video = VideoFileClip(os.path.join(path, '%s_mesh.avi'%name))
        # L.append(video)
    final_video = concatenate_videoclips(L)
    final_video.to_videofile(os.path.join(path, 'final_video_nomesh.mp4'), fps=30, remove_temp=False)

   
if __name__ == "__main__":
    # main()
    concat_video()
    