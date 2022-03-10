import os
import argparse
import logging
import tqdm
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import torchvision.transforms
from torch.utils.data import DataLoader
import kornia.utils.image
from kornia.geometry.transform import resize
from colour.difference import delta_E

from mixedillWB.src import ops
from mixedillWB.src import weight_refinement as weight_refinement
from mixedillWB.src import wb_net as mixed_wb_net
from src import wb_net
from src import dataset

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


def angular_error(a, b):
    radians_to_degrees = 180.0 / np.pi

    ab = np.sum(np.multiply(a, b), axis=1)
    a_norm = np.linalg.norm(a, axis=1)
    b_norm = np.linalg.norm(b, axis=1)

    # Avoid zero-values (to avoid NaNs)
    a_norm = np.clip(a_norm, a_min=1e-7, a_max=None)
    b_norm = np.clip(b_norm, a_min=1e-7, a_max=None)

    similarity = np.divide(ab, np.multiply(a_norm, b_norm))

    return np.arccos(similarity) * radians_to_degrees


def mean_angular_error(a, b):
    """Calculate mean angular error (via cosine similarity)."""
    return np.nanmean(angular_error(a, b))


def test_net(net, device, data_dir, model_name, out_dir, save_weights,
             multi_scale=False, keep_aspect_ratio=False, t_size=128,
             post_process=False, batch_size=32, wb_settings=None):
    """ Tests a trained network and saves the trained model in harddisk.
  """
    if wb_settings is None:
        wb_settings = ['D', 'S', 'T', 'F', 'C']

    # print(data_dir)
    # input_files = dataset.Data.load_files(data_dir)
    # if input_files == []:
    #      input_files = dataset.Data.load_files(data_dir, mode='testing')
    #
    # if multi_scale:
    #     test_set = dataset.Data(input_files, mode='testing', t_size=t_size,
    #                             wb_settings=wb_settings,
    #                             keep_aspect_ratio=keep_aspect_ratio)
    # else:
    #     test_set = dataset.Data(input_files, mode='testing', t_size=t_size,
    #                             wb_settings=wb_settings,
    #                             keep_aspect_ratio=keep_aspect_ratio)

    if args.dataset == "cubeplus":
        test_set = dataset.CubeWBDataset("/media/birdortyedi/e5042b8f-ca5e-4a22-ac68-7e69ff648bc4/RenderedWB/cube-wb", "../mixedillWB/data/cubeplus/", t_size=t_size, wb_settings=wb_settings, keep_aspect_ratio=keep_aspect_ratio)
    elif args.dataset == "mit":  ## buggy
        base_dir = "./data/mit-adobe"
        test_set = dataset.Data([os.path.join(base_dir, p) for p in os.listdir(base_dir)])
    elif args.dataset == "synthetic":
        test_set = dataset.SyntheticDataset(args.tedir.format("images/"), args.tedir.format("gt/"), t_size=t_size, wb_settings=wb_settings, keep_aspect_ratio=keep_aspect_ratio)
    else:
        raise NotImplementedError("Dataset name wrong!")
    test_set = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)

    logging.info(f'''Starting testing:
        Model Name:            {model_name}
        Batch size:            {batch_size}
        Output dir:            {out_dir}
        WB settings:           {wb_settings}
        Save weights:          {save_weights}
        Device:                {device.type}
  ''')

    os.makedirs(out_dir, exist_ok=True)

    mse_lst, mae_lst, deltaE2000_lst = list(), list(), list()
    with torch.no_grad():
        for step, batch in enumerate(tqdm.tqdm(test_set, total=len(test_set))):
            img = batch['image']
            gt = batch['gt']

            img = img.to(device=device, dtype=torch.float32)
            gt = gt.to(device=device, dtype=torch.float32)

            _, weights = net(img)

            if multi_scale:
                img_1 = resize(img, size=(int(0.5 * img.shape[2]), int(0.5 * img.shape[3])), interpolation='bilinear', align_corners=True)
                _, weights_1 = net(img_1)
                weights_1 = resize(weights_1, size=(img.shape[2], img.shape[3]), interpolation='bilinear', align_corners=True)

                img_2 = resize(img, size=(int(0.25 * img.shape[2]), int(0.25 * img.shape[3])), interpolation='bilinear', align_corners=True)
                _, weights_2 = net(img_2)
                weights_2 = resize(weights_2, size=(img.shape[2], img.shape[3]), interpolation='bilinear', align_corners=True)

                weights = (weights + weights_1 + weights_2) / 3

            d_img = batch['fs_d_img'].to(device=device, dtype=torch.float32)
            # torchvision.transforms.ToPILImage()(d_img.squeeze()).save("./data/cube-wb/{}_D.png".format(batch['filename'][0].split("/")[-1]))

            s_img = batch['fs_s_img'].to(device=device, dtype=torch.float32)
            # torchvision.transforms.ToPILImage()(s_img.squeeze()).save("./data/cube-wb/{}_S.png".format(batch['filename'][0].split("/")[-1]))

            t_img = batch['fs_t_img'].to(device=device, dtype=torch.float32)
            # torchvision.transforms.ToPILImage()(t_img.squeeze()).save("./data/cube-wb/{}_T.png".format(batch['filename'][0].split("/")[-1]))

            imgs = [d_img, s_img, t_img]
            if 'F' in wb_settings:
                f_img = batch['fs_f_img'].to(device=device, dtype=torch.float32)
                # torchvision.transforms.ToPILImage()(f_img.squeeze()).save("./data/cube-wb/{}_F.png".format(batch['filename'][0].split("/")[-1]))
                imgs.append(f_img)
            if 'C' in wb_settings:
                c_img = batch['fs_c_img'].to(device=device, dtype=torch.float32)
                # torchvision.transforms.ToPILImage()(c_img.squeeze()).save("./data/cube-wb/{}_C.png".format(batch['filename'][0].split("/")[-1]))
                imgs.append(c_img)

            filename = batch['filename']
            weights = resize(weights, size=(d_img.shape[2], d_img.shape[3]), interpolation='bilinear', align_corners=True)

            if post_process:
                for i in range(weights.shape[1]):
                    for j in range(weights.shape[0]):
                        ref = imgs[0][j, :, :, :]
                        curr_weight = weights[j, i, :, :]
                        refined_weight = weight_refinement.process_image(ref, curr_weight, tensor=True)
                        weights[j, i, :, :] = refined_weight
                        weights = weights / torch.sum(weights, dim=1)

            for i in range(weights.shape[1]):
                if i == 0:
                    out_img = torch.unsqueeze(weights[:, i, :, :], dim=1) * imgs[i]
                else:
                    out_img += torch.unsqueeze(weights[:, i, :, :], dim=1) * imgs[i]

            for gt_, out_ in zip(gt.float(), out_img.float().permute(0, 2, 3, 1)):
                mae_and_delta_Es = [[mean_angular_error(gt_.cpu().squeeze().numpy().reshape(-1, 3), out_.cpu().squeeze().numpy().reshape(-1, 3)),
                                     np.mean(delta_E(cv2.cvtColor(gt_.cpu().squeeze().numpy(), cv2.COLOR_RGB2Lab), cv2.cvtColor(out_.cpu().squeeze().numpy(), cv2.COLOR_RGB2Lab)))]]
                mae, deltaE = np.mean(mae_and_delta_Es, axis=0)
                mse = (((gt_ - out_) * 255.) ** 2).mean().cpu().item()
                print("Sample {}/{}: MSE: {}, MAE: {}, DELTA_E: {}".format(step+1, len(test_set), mse, mae, deltaE), end="\n\n")
                mse_lst.append(mse)
                mae_lst.append(mae)
                deltaE2000_lst.append(deltaE)
                print("Average:\n"
                "\nMSE: {}, Q1: {}, Q2: {}, Q3: {}"
                "\nMAE: {}, Q1: {}, Q2: {}, Q3: {}"
                "\nDELTA_E: {}, Q1: {}, Q2: {}, Q3: {}".format(np.mean(mse_lst), np.quantile(mse_lst, 0.25), np.quantile(mse_lst, 0.5), np.quantile(mse_lst, 0.75),
                                                               np.mean(mae_lst), np.quantile(mae_lst, 0.25), np.quantile(mae_lst, 0.5), np.quantile(mae_lst, 0.75),
                                                               np.mean(deltaE2000_lst), np.quantile(deltaE2000_lst, 0.25), np.quantile(deltaE2000_lst, 0.5), np.quantile(deltaE2000_lst, 0.75)))

            for i, fname in enumerate(filename):
                result = ops.to_image(out_img[i, :, :, :])
                name = os.path.join(out_dir, os.path.basename(fname).split("_")[0] + '_WB.png') if args.dataset == "cubeplus" else \
                    os.path.join(out_dir, "_".join(os.path.basename(fname).split("_")[:2]) + '_WB.jpg')
                result.save(name)

                result = ops.to_image(t_img[i, :, :, :])
                name = os.path.join(out_dir, os.path.basename(fname).split("_")[0] + '_T.png') if args.dataset == "cubeplus" else \
                    os.path.join(out_dir, "_".join(os.path.basename(fname).split("_")[:2]) + '_T.jpg')
                result.save(name)

                result = ops.to_image(d_img[i, :, :, :])
                name = os.path.join(out_dir, os.path.basename(fname).split("_")[0] + '_D.png') if args.dataset == "cubeplus" else \
                    os.path.join(out_dir, "_".join(os.path.basename(fname).split("_")[:2]) + '_D.jpg')
                result.save(name)

                result = ops.to_image(s_img[i, :, :, :])
                name = os.path.join(out_dir, os.path.basename(fname).split("_")[0] + '_S.png') if args.dataset == "cubeplus" else \
                    os.path.join(out_dir, "_".join(os.path.basename(fname).split("_")[:2]) + '_S.jpg')
                result.save(name)

                if save_weights:
                    # save weights
                    postfix = ['D', 'S', 'T']
                    if 'F' in wb_settings:
                        postfix.append('F')
                    if 'C' in wb_settings:
                        postfix.append('C')
                    for j in range(weights.shape[1]):
                        weight = torch.tile(weights[:, j, :, :], dims=(3, 1, 1))
                        weight = ops.to_image(weight)
                        name = os.path.join(out_dir, os.path.basename(fname).split("_")[0] + f'_weight_{postfix[j]}.png') if args.dataset == "cubeplus" else \
                            os.path.join(out_dir, "_".join(os.path.basename(fname).split("_")[:2]) + f'_weight_{postfix[j]}.png')
                        weight.save(name)
    final_info = "\nFinal Info--->  \nMSE: {}, Q1: {}, Q2: {}, Q3: {} \nMAE: {}, Q1: {}, Q2: {}, Q3: {} \nDELTA_E: {}, Q1: {}, Q2: {}, Q3: {}".format(
        np.mean(mse_lst), np.quantile(mse_lst, 0.25), np.quantile(mse_lst, 0.5), np.quantile(mse_lst, 0.75),
        np.mean(mae_lst), np.quantile(mae_lst, 0.25), np.quantile(mae_lst, 0.5), np.quantile(mae_lst, 0.75),
        np.mean(deltaE2000_lst), np.quantile(deltaE2000_lst, 0.25), np.quantile(deltaE2000_lst, 0.5), np.quantile(deltaE2000_lst, 0.75)
    )
    print(final_info)
    logging.info('End of testing')
    with open(os.path.join(args.outdir, f"{str(args.model_name)}_{args.dataset}.txt"), "w+") as f:
        f.write(final_info)


def get_args():
    """ Gets command-line arguments.

  Returns:
    Return command-line arguments as a set of attributes.
  """

    parser = argparse.ArgumentParser(description='Test WB Correction.')

    parser.add_argument('-d', '--dataset', type=str, default="synthetic", help="cubeplus, mit or synthetic")
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=1, help='Batch size', dest='batch_size')
    parser.add_argument('-nrm', '--normalization', dest='norm', type=bool, default=False, help='Apply BN in network')
    parser.add_argument('-ml', '--model-location', dest='model_location', default=None)
    parser.add_argument('-wbs', '--wb-settings', dest='wb_settings', nargs='+', default=['D', 'S', 'T'])  # default=['D', 'S', 'T', 'F', 'C'])
    parser.add_argument('-sw', '--save-weights', dest='save_weights', default=False, type=bool)
    parser.add_argument('-ka', '--keep-aspect-ratio', dest='keep_aspect_ratio', default=False, type=bool, help='To keep aspect ratio before processing. Only works when multi-scale is off.')
    parser.add_argument('-ms', '--multi-scale', dest='multi_scale', default=True, type=bool)
    parser.add_argument('-pp', '--post-process', dest='post_process', default=True, type=bool)
    parser.add_argument('-ted', '--testing-dir', dest='tedir', default='./data/synthetic-gt/', help='Testing directory')
    parser.add_argument('-od', '--outdir', dest='outdir', default='./results/ifrnet/images/synthetic', help='Results directory')
    parser.add_argument('-g', '--gpu', dest='gpu', default=0, type=int)
    parser.add_argument('-ts', '--target-size', dest='t_size', default=384, type=int, help='Size before feeding images to the network. '
                                                                                           'Typically, 128 or 256 give good results. If '
                                                                                           'multi-scale is used, then 384 is recommended.')
    parser.add_argument('-mn', '--model-name', dest='model_name', type=str,
                        default='ifrnet_p_64_D_S_T_200',
                        # default='WB_model_p_64_D_S_T',
                        help='Model name')
    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logging.info('Testing Lighting-as-Style WB correction')
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cpu':
        torch.cuda.set_device(args.gpu)

    logging.info(f'Using device {device}')
    ps = eval(args.model_name.split("_")[2])  # infer the patch size from the model name
    if "ifr" in args.model_name:
        net = wb_net.WBnetIFRNet(ps=ps, device=device, norm=args.norm, inchnls=3 * len(args.wb_settings))
    else:
        net = mixed_wb_net.WBnet(device=device, norm=args.norm, inchnls=3 * len(args.wb_settings))

    model_path = os.path.join('models', args.model_name + '.pth')
    net.load_state_dict(torch.load(model_path, map_location=device))
    logging.info(f'Model loaded from {model_path}')

    net.to(device=device)
    net.eval()

    test_net(net=net, device=device, data_dir=args.tedir,
             batch_size=args.batch_size, out_dir=args.outdir,
             post_process=args.post_process,
             keep_aspect_ratio=args.keep_aspect_ratio,
             t_size=args.t_size,
             multi_scale=args.multi_scale, model_name=args.model_name,
             save_weights=args.save_weights,
             wb_settings=args.wb_settings)
