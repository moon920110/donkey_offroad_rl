import os
import cv2
import numpy as np
from shutil import copyfile
from PIL import Image



class LN():
    def __init__(self, model_name, tub_name=None, train=True):
        if train:
            ref_img_path = os.path.join(tub_name, '1_cam-image_array_.jpg')
            copyfile(ref_img_path, './data/{}_reference.jpg'.format(model_name.split('/')[-1][:-3]))
        else:
            ref_img_path = './data/{}_reference.jpg'.format(model_name.split('/')[-1][:-3])
            print("Load image from", ref_img_path)

        ref_img = cv2.imread(ref_img_path)
        ref_HSV = cv2.cvtColor(ref_img, cv2.COLOR_BGR2HSV)
        ref_v = ref_HSV[:, :, 2]
        self.ref_cdf = self._get_cdf(ref_v)

    def normalize_lightness(self, src_img):
        src_img = src_img * 255.0
        src_img = src_img.astype(np.uint8)
        src_HSV = cv2.cvtColor(src_img, cv2.COLOR_RGB2HSV)
        src_v = src_HSV[:, :, 2]

        transformed_src_v = self._match_histogram(src_v)
        # transformed_src_v = _homomorphic(transformed_src_v)

        src_HSV[:, :, 2] = transformed_src_v

        output = cv2.cvtColor(src_HSV, cv2.COLOR_HSV2RGB)
        output = cv2.convertScaleAbs(output)
        output = output.astype(np.float32) / 255.0

        return output
        
    def _homomorphic(self, v):
        rows = v.shape[0]
        cols = v.shape[1]

        v_log = np.log1p(np.array(v, dtype=np.float) / 255.)

        M = 2 * rows + 1
        N = 2 * cols + 1

        sigma = 10
        (X, Y) = np.meshgrid(np.linspace(0, N-1, N), np.linspace(0, M-1, M))
        x_c = np.ceil(N/2)
        y_c = np.ceil(M/2)
        gaussian_numerator = (X - x_c) ** 2 + (Y - y_c) ** 2

        lpf = np.exp(-gaussian_numerator / (2 * sigma * sigma))
        hpf = 1 - lpf

        lpf_shift = np.fft.ifftshift(lpf)
        hpf_shift = np.fft.ifftshift(hpf)
        
        v_fft = np.fft.fft2(v_log, (M, N))
        v_lf = np.real(np.fft.ifft2(v_fft * lpf_shift, (M, N)))
        v_hf = np.real(np.fft.ifft2(v_fft * hpf_shift, (M, N)))

        g1 = 0.3
        g2 = 1.5
        v_adjusting = g1 * v_lf[0:rows, 0:cols] + g2 * v_hf[0:rows, 0:cols]

        v_exp = np.expm1(v_adjusting)
        v_exp = (v_exp - np.min(v_exp)) / (np.max(v_exp) - np.min(v_exp))
        v_out = np.array(255 * v_exp, dtype=np.uint8)

        return v_out

    def _match_histogram(self, src_v):
        src_cdf = self._get_cdf(src_v)
        lut = self._calc_lut(src_cdf)

        transformed_v = cv2.LUT(src_v, lut)

        return transformed_v

    def _calc_cdf(self, hist):
        cdf = hist.cumsum()
        norm_cdf = cdf / float(cdf.max())

        return norm_cdf

    def _calc_lut(self, src_cdf):
        lut = np.zeros(256)
        lut_val = 0

        for src_pix_val in range(len(src_cdf)):
            lut_val = src_pix_val
            for ref_pixel_val in range(len(self.ref_cdf)):
                if self.ref_cdf[ref_pixel_val] >= src_cdf[src_pix_val]:
                    lut_val = ref_pixel_val
                    break
            lut[src_pix_val] = lut_val

        return lut

    def _get_cdf(self, y):
        hist, _ = np.histogram(y.flatten(), 256, [0, 256])
        cdf = self._calc_cdf(hist)

        return cdf

