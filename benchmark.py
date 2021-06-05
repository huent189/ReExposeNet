import cv2
import glob
import argparse
import numpy as np
# import sewar.full_ref
import piq
import os
import torch
from keras.models import load_model
import loss_definition
# import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
def single_validate(img, img_gt, model):
    width = img.shape[1]
    height = img.shape[0]
    width = width // 8 * 8
    height = height // 8 * 8
    img = cv2.resize(img, (width, height))
    img_gt = cv2.resize(img_gt, (width, height))
    shape = img.shape
    # print(img.shape)
    img = np.reshape(img, (1, height,width, shape[2]))
    img = img/255
    im_pred = np.clip(model.predict(img), .0, 1.)
    y_pred = torch.from_numpy(im_pred).permute(0,3,1,2)
    y = torch.from_numpy(img_gt).unsqueeze(0).permute(0,3,1,2) / 255.0
    result_psnr = piq.psnr( y_pred, y, data_range=1.)
    result_ssim = piq.ssim(y_pred, y,data_range=1.)
    return np.uint8(im_pred * 255).reshape(height,width, shape[2]), result_psnr, result_ssim
def main():
    parser = argparse.ArgumentParser(description='IAGCWD')
    parser.add_argument('--input', dest='input_dir', default='./input/', type=str, \
                        help='Input directory for image(s)')
    parser.add_argument('--output', dest='output_dir', default='./output/', type=str, \
                        help='Output directory for image(s)')
    parser.add_argument('--dataset')
    args = parser.parse_args()
    dataset = args.dataset
    if dataset == 'sice':
      img_paths = glob.glob(os.path.join(args.input_dir, "[0-9]*/*"))
    else:
      img_paths = glob.glob(os.path.join(args.input_dir, "*"))
    print('total img:', len(img_paths))
    psnr_over = 0
    psnr_under = 0
    psnr_all = 0
    ssim_over = 0
    ssim_under = 0
    ssim_all = 0
    count_under = 0
    count_over = 0
    # print(len(img_paths))
    model_under = load_model('model_under.hd5f',custom_objects={'loss_mix_v3': loss_definition.loss_mix_v3}, compile=False)
    model_under.load_weights('weigths_under.hd5f')
    model_over = load_model('model_over.hd5f', custom_objects={'loss_mix_v3': loss_definition.loss_mix_v3}, compile=False)
    model_over.load_weights('weigths_over.hd5f')
    for path in img_paths:
        img = cv2.imread(path, 1)
        if dataset =='sice':
          gt_path = glob.glob(os.path.join(args.input_dir, "Label", path.split("/")[-2] + ".*"))[0]
        else:
          gt_path = glob.glob(path.replace('INPUT_IMAGES','expert_e_testing_set').split('-')[0] + '*')[0]
        # print(path, gt_path)
        img_gt = cv2.imread(gt_path, 1)
        name = path.replace(args.input_dir, '')
        out_name = args.output_dir+name 
        base_dir = os.path.split(out_name)[0]
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
        if dataset == 'sice':
          is_neg = np.mean(img) < (0.5 * 255)
        else:
          is_neg = '_N' in name
        if is_neg:
            # print(name)
            pred_im, psnr, ssim = single_validate(img, img_gt, model_under)
        else:
            pred_im, psnr, ssim = single_validate(img, img_gt, model_over)
        cv2.imwrite(args.output_dir+name, pred_im)
        if is_neg:
          ssim_under += ssim
          psnr_under += psnr
          count_under += 1
        else:
          ssim_over += ssim
          psnr_over += psnr
          count_over += 1
        ssim_all += ssim
        psnr_all += psnr
        # break
    if count_under > 0:
      print('count_under', count_under)
      print('ssim_under', ssim_under / count_under)
      print('psnr_under', psnr_under / count_under)
    if count_over > 0:
      print('count_over', count_over)
      print('ssim_over', ssim_over / count_over)
      print('psnr_over', psnr_over / count_over)
    print('ssim_all', ssim_all/ (count_over + count_under))
    print('psnr_all', psnr_all/ (count_over + count_under))
if __name__ == '__main__':
    main()

