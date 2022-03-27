import cv2
import math
import logging
from timeit import default_timer as timer
from pathlib import Path
import numpy as np
import numpy.typing as npt

from matplotlib import pyplot as plt
from matplotlib import ticker, interactive
from matplotlib.patches import Circle

from skimage.morphology import binary_opening, binary_closing, remove_small_objects
from skimage import measure
from scipy.ndimage import label, generate_binary_structure

from typing import Dict

from .maps import construct_dictionary, SSIM, get_quality_map_dict
from .functions import STFT, local_constrast_enhancement_gaussian, fast_cartoon_texture, LogGaborFilter, gabor_filtering_pixel2
from .import_graph import get_patch_index, Descriptor, MinutiaeExtraction, AutoEncoder


class LatentExtractionModel:
    def __init__(self, des_model_dirs=None, minu_model_dirs=None, enhancement_model_dir=None):

        self.des_models = None
        self.minu_model = None

        self.minu_model_dirs: Path = minu_model_dirs
        self.des_model_dirs: Path = des_model_dirs
        self.enhancement_model_dir: Path = enhancement_model_dir

        # Obtaining maps
        maps = construct_dictionary(ori_num=60)
        self.dict, self.spacing, self.dict_all = maps[:3]
        self.dict_ori, self.dict_spacing = maps[3:]

        logging.info("Loading models, this may take some time...")

        if self.minu_model_dirs is not None:
            self.minu_model: list[MinutiaeExtraction] = []
            for i, minu_model_dir in enumerate(minu_model_dirs):
                echo_info = (i + 1, len(minu_model_dirs), minu_model_dir)
                logging.info(
                    "Loading minutiae model (%d of %d ): %s" % echo_info)
                model = MinutiaeExtraction(minu_model_dir)
                self.minu_model.append(model)

        patchSize = 160
        oriNum = 64
        self.patchIndexV = get_patch_index(
            patchSize, patchSize, oriNum, isMinu=1
        )

        if self.des_model_dirs is not None:
            self.des_models = []
            for i, model_dir in enumerate(des_model_dirs):

                echo_info = (i + 1, len(des_model_dirs), model_dir)
                logging.info(
                    "Loading descriptor model (%d of %d ): %s" % echo_info)

                dmodel = Descriptor(
                    model_dir, input_name="inputs:0",
                    result_name='embedding:0'
                )

                self.des_models.append(dmodel)

        if self.enhancement_model_dir is not None:
            logging.info("Loading enhancement model: " +
                         self.enhancement_model_dir.as_posix())
            emodel = AutoEncoder(enhancement_model_dir)
            self.enhancement_model = emodel

        logging.info("Finished loading models.")

    def latent_extraction(self, img_file: str, ext: str, output_dir=None, ppi=500, show_minutiae=True, minu_file=None, block_size=16):

        # Edited and moved to main package
        def image2blocks(image: npt.NDArray, block_size: int) -> npt.NDArray:
            row, col = image.shape
            blockR = row // block_size
            blockC = col // block_size

            return image[:blockR * block_size, :blockC * block_size]

        img: npt.NDArray[np.uint8] = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
        img_name = Path(img_file).name
        root_name = Path(output_dir) / img_name

        if ppi != 500:
            img = cv2.resize(img, (0, 0), fx=500.0 / ppi, fy=500.0 / ppi)

        img = image2blocks(img, block_size)
        h, w = img.shape

        # Starting timer for feature extraction
        start = timer()

        # cropping using two dictionary based approach
        if minu_file is not None:
            input_minu = np.array(np.loadtxt(minu_file))
            # remove low quality minutiae points
            input_minu[:, 2] = input_minu[:, 2] / 180.0 * np.pi
        else:
            input_minu = []

        # Preprocessing
        texture_img = fast_cartoon_texture(img, sigma=2.5, show=False)
        stft_texture_img = STFT(texture_img)
        gaus_img = local_constrast_enhancement_gaussian(img)
        stft_img = STFT(img)
        gaus_stft_img = STFT(gaus_img)

        # Saving images
        images = {
            'texture': texture_img,
            'stft_texture': stft_texture_img,
            'gaus': gaus_img.astype(np.uint8),
            'stft': stft_img,
            'gaus_stft': gaus_stft_img
        }
        minutiae = {}
        figures = {}

        # --------------------------------------------------------------------
        # Run AEC tensorflow graph

        aec_img = self.enhancement_model.enhance_image(stft_texture_img)
        images['aec'] = aec_img

        quality_map_aec, dir_map_aec, fre_map_aec = get_quality_map_dict(
            aec_img, self.dict_all, self.dict_ori, self.dict_spacing, R=500.0
        )

        mask = self.get_mask(
            quality_map_aec, stft_texture_img, aec_img, block_size
        )

        images['mask'] = mask
        images['aec_mask'] = aec_img * mask

        # --------------------------------------------------------------------

        minutiae_sets = []

        stft_mnt = self.minu_model[0].extract_minutiae(
            stft_img, minu_thr=0.05
        )
        gaus_stft_mnt = self.minu_model[0].extract_minutiae(
            gaus_stft_img, minu_thr=0.1
        )
        aec_mnt = self.minu_model[1].extract_minutiae(
            aec_img, minu_thr=0.25
        )
        aec_mnt = self.remove_spurious_minutiae(aec_mnt, mask)

        minutiae_sets.append(stft_mnt)
        minutiae_sets.append(gaus_stft_mnt)
        minutiae_sets.append(aec_mnt)

        minutiae['stft'] = stft_mnt
        minutiae['gaus_stft'] = gaus_stft_mnt
        minutiae['aec'] = aec_mnt

        if show_minutiae:
            figures['stft'] = show_minutiae_sets(
                stft_img, [input_minu, stft_mnt], mask=None)
            figures['gaus_stft'] = show_minutiae_sets(
                gaus_stft_img, [input_minu, gaus_stft_mnt], mask=None)
            figures['aec'] = show_minutiae_sets(
                aec_img, [input_minu, aec_mnt], mask=mask)

        # --------------------------------------------------------------------
        # Enhance gaussian contrast image

        gabor_gaus_img = gabor_filtering_pixel2(
            gaus_img, dir_map_aec + math.pi / 2,
            fre_map_aec, mask=np.ones((h, w)), block_size=16, angle_inc=3
        )
        images['gabor_gaus'] = gabor_gaus_img

        gabor_gaus_mnt = self.minu_model[1].extract_minutiae(
            gabor_gaus_img, minu_thr=0.25
        )
        gabor_gaus_mnt = self.remove_spurious_minutiae(
            gabor_gaus_mnt, mask)

        minutiae_sets.append(gabor_gaus_mnt)

        minutiae['gabor_gaus'] = gabor_gaus_mnt

        if show_minutiae:
            figures['gabor_gaus'] = show_minutiae_sets(
                gabor_gaus_img, [input_minu, gabor_gaus_mnt], mask=mask
            )

        # --------------------------------------------------------------------
        # Enhance gaussian contrast image CENATAV

        gfilter = LogGaborFilter(
            dir_map_aec + math.pi / 2, fre_map_aec, mask=np.ones((h, w)))
        log_gabor_gaus_img, thr = gfilter.apply(gaus_img)

        images['log_gabor_gaus'] = log_gabor_gaus_img

        log_gabor_gaus_bin = (log_gabor_gaus_img >= thr).astype(np.uint8) * 255
        log_gabor_gaus_bin[mask == 0] = 0
        images['log_gabor_gaus_bin'] = log_gabor_gaus_bin

        # --------------------------------------------------------------------
        # Extracting minutiae from the enhanced contrast gaussian image CENATAV

        log_gabor_gaus_mnt = self.minu_model[1].extract_minutiae(
            log_gabor_gaus_img, minu_thr=0.25)
        log_gabor_gaus_mnt = self.remove_spurious_minutiae(
            log_gabor_gaus_mnt, mask)

        minutiae_sets.append(log_gabor_gaus_mnt)

        if show_minutiae:
            figures['log_gabor_gaus'] = show_minutiae_sets(
                log_gabor_gaus_img, [input_minu, log_gabor_gaus_mnt], mask=mask)

        minutiae['log_gabor_gaus'] = log_gabor_gaus_mnt

        # --------------------------------------------------------------------
        # Enhanced texture image

        gabor_texture_img = gabor_filtering_pixel2(
            texture_img, dir_map_aec + math.pi / 2, fre_map_aec,
            mask=np.ones((h, w)), block_size=block_size, angle_inc=3
        )

        images['gabor_texture'] = gabor_texture_img

        gabor_texture_mnt = self.minu_model[1].extract_minutiae(
            gabor_texture_img, minu_thr=0.25)
        gabor_texture_mnt = self.remove_spurious_minutiae(
            gabor_texture_mnt, mask)

        minutiae_sets.append(gabor_texture_mnt)

        if show_minutiae:
            figures['gabor_texture'] = show_minutiae_sets(
                gabor_texture_img, [input_minu, gabor_texture_mnt], mask=mask)

        minutiae['gabor_texture'] = gabor_texture_mnt

        # --------------------------------------------------------------------
        # Enhance texture image CENATAV

        log_gabor_texture_img, thr = gfilter.apply(texture_img)

        images['log_gabor_texture'] = log_gabor_texture_img

        gabor_texture_bin = (log_gabor_texture_img >=
                             thr).astype(np.uint8) * 255
        gabor_texture_bin[mask == 0] = 0
        images['gabor_texture_bin'] = gabor_texture_bin

        log_gabor_texture_mnt = self.minu_model[1].extract_minutiae(
            log_gabor_texture_img, minu_thr=0.25)
        log_gabor_texture_mnt = self.remove_spurious_minutiae(
            log_gabor_texture_mnt, mask)
        minutiae_sets.append(log_gabor_texture_mnt)

        if show_minutiae:
            figures['log_gabor_texture'] = show_minutiae_sets(
                log_gabor_texture_img, [input_minu, log_gabor_texture_mnt], mask=mask)

        minutiae['log_gabor_texture'] = log_gabor_texture_mnt

        # --------------------------------------------------------------------

        mnt1 = self.get_common_minutiae(minutiae_sets, thr=2)
        mnt2 = self.get_common_minutiae(minutiae_sets, thr=3)
        mnt3 = self.get_common_minutiae(minutiae_sets, thr=4)
        mnt4 = self.get_common_minutiae(minutiae_sets, thr=5)

        minutiae_sets.append(mnt1)
        minutiae_sets.append(mnt2)
        minutiae_sets.append(mnt3)
        minutiae_sets.append(mnt4)

        if show_minutiae:
            figures['common1'] = show_minutiae_sets(
                img, [input_minu, mnt1], mask=mask)
            figures['common2'] = show_minutiae_sets(
                img, [input_minu, mnt2], mask=mask)
            figures['common3'] = show_minutiae_sets(
                img, [input_minu, mnt3], mask=mask)
            figures['common4'] = show_minutiae_sets(
                img, [input_minu, mnt4], mask=mask)

        minutiae['common1'] = mnt1
        minutiae['common2'] = mnt2
        minutiae['common3'] = mnt3
        minutiae['common4'] = mnt4

        self.generate_images(root_name, ext, images, minutiae, figures)

        end = timer()

        logging.info('Time for minutiae extraction: %f' % (end - start))

        return [mnt1, mnt2, mnt3, mnt4], images['mask'] * 255, images['aec'].astype(np.uint8), images['gabor_texture_bin']

    def generate_images(self, root_name: Path, ext: str, images: Dict[str, npt.NDArray], minutiae: Dict[str, npt.NDArray], figures: Dict[str, plt.Figure]) -> None:
        for item in images:
            fname = str(root_name.with_name(f'{root_name.name}_{item}{ext}'))
            cv2.imwrite(fname, images[item])

        for item in minutiae:
            fname = str(root_name.with_name(
                f'{root_name.name}_{item}_mnt.txt'))
            self.minutiae2txt(fname, minutiae[item])

        for item in figures:
            fname = str(root_name.with_name(
                f'{root_name.name}_{item}_mnt{ext}'))
            figures[item].savefig(
                fname, dpi=600, bbox_inches='tight', pad_inches=0.0
            )

    def get_mask(self, quality_map_aec, stft_texture_img, aec_img, block_size) -> npt.NDArray:
        bmask_aec = quality_map_aec > 0.45
        bmask_aec = binary_closing(
            bmask_aec, np.ones((3, 3))).astype(np.int64)
        bmask_aec = binary_opening(
            bmask_aec, np.ones((3, 3))).astype(np.int64)
        blkmask_ssim = SSIM(stft_texture_img, aec_img, thr=0.2)
        blkmask = blkmask_ssim * bmask_aec
        blk_h, blk_w = blkmask.shape
        mask = cv2.resize(
            blkmask.astype(float),
            (block_size * blk_w, block_size * blk_h),
            interpolation=cv2.INTER_LINEAR
        )
        mask[mask > 0] = 1

        # Keeping only the biggest area in mask
        structures = generate_binary_structure(mask.ndim, 1)
        labels = np.zeros_like(mask, dtype=np.int32)
        label(mask, structures, output=labels)
        component_sizes = np.bincount(labels.ravel())
        component_sizes = sorted(component_sizes, reverse=True)

        if len(component_sizes) > 1:
            min_size = component_sizes[1]
            mask = remove_small_objects(mask.astype(bool), min_size=min_size)
            mask = mask.astype(np.uint8)
        return mask

    def get_common_minutiae(self, minutiae_sets, thr=3):
        """Return common minutiae among different minutiae sets"""
        nrof_minutiae_sets = len(minutiae_sets)

        init_ind = 3
        if len(minutiae_sets[init_ind]) == 0:
            return []
        mnt = list(minutiae_sets[init_ind][:, :4])
        count = list(np.ones(len(mnt),))
        for i in range(0, nrof_minutiae_sets):
            if i == init_ind:
                continue
            for j in range(len(minutiae_sets[i])):
                x2 = minutiae_sets[i][j, 0]
                y2 = minutiae_sets[i][j, 1]
                ori2 = minutiae_sets[i][j, 2]
                found = False
                for k in range(len(mnt)):
                    x1 = mnt[k][0]
                    y1 = mnt[k][1]
                    ori1 = mnt[k][2]
                    dist = math.sqrt(
                        (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1)
                    )

                    ori_dist = math.fabs(ori1 - ori2)
                    if ori_dist > math.pi / 2:
                        ori_dist = math.pi - ori_dist
                    if dist <= 10 and ori_dist < math.pi / 6:
                        count[k] += 1
                        found = True
                        break
                if not found:
                    mnt.append([x2, y2, ori2, 1])
                    count.append(1)
        count = np.asarray(count)
        ind = np.where(count >= thr)[0]
        mnt = np.asarray(mnt)
        mnt = mnt[ind, :]
        mnt[:, 3] = 1
        return mnt

    def remove_spurious_minutiae(self, mnt, mask):
        """Remove spurious minutiae in the borders of a mask"""
        minu_num = len(mnt)
        if minu_num <= 0:
            return mnt
        flag = np.ones((minu_num,), np.uint8)
        h, w = mask.shape[:2]
        R = 10
        for i in range(minu_num):
            x = mnt[i, 0]
            y = mnt[i, 1]
            x = np.int64(x)
            y = np.int64(y)
            if x < R or y < R or x > w - R - 1 or y > h - R - 1:
                flag[i] = 0
            elif(mask[y - R, x - R] == 0 or mask[y - R, x + R] == 0 or
                 mask[y + R, x - R] == 0 or mask[y + R, x + R] == 0):
                flag[i] = 0
        mnt = mnt[flag > 0, :]
        return mnt

    def minutiae2txt(self, fname, mnt):
        with open(fname, 'w') as file:
            file.write(f'{len(mnt)}\n')
            for m in mnt:
                file.write(
                    f'{int(m[0])} {int(m[1])} {m[2].astype(np.float64)} {m[3].astype(np.float64)}\n'
                )


def load_graphs(path_config: Path):
    """Returns an instance of the latent feature extractor class"""
    DescriptorModelPatch2 = path_config / \
        "embeddingsize_64_patchtype_2/20180410-213622/"
    DescriptorModelPatch8 = path_config / \
        "embeddingsize_64_patchtype_8/20180410-213932/"
    DescriptorModelPatch11 = path_config / \
        "embeddingsize_64_patchtype_11/20180411-123356/"
    MinutiaeExtractionModel = path_config / \
        "minutiae_AEC_64_fcn_2/"
    MinutiaeExtractionModelLatentSTFT = path_config / \
        "minutiae_AEC_64_fcn_2_Latent_STFT/"
    EnhancementModel = path_config / \
        "Enhancement_AEC_128_depth_4_STFT/"

    return LatentExtractionModel(
        des_model_dirs=[
            DescriptorModelPatch2,
            DescriptorModelPatch8,
            DescriptorModelPatch11
        ],
        enhancement_model_dir=EnhancementModel,
        minu_model_dirs=[
            MinutiaeExtractionModelLatentSTFT,
            MinutiaeExtractionModel
        ]
    )


def show_minutiae_sets(img, minutiae_sets, mask=None):
    interactive(False)
    # for the latent or the low quality rolled print
    fig, ax = plt.subplots(1)
    ax.set_aspect('equal')

    plt.gca().set_axis_off()
    plt.gca().xaxis.set_major_locator(ticker.NullLocator())
    plt.gca().yaxis.set_major_locator(ticker.NullLocator())
    plt.margins(0, 0)
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                        hspace=0, wspace=0)

    arrow_len = 15
    if mask is not None:
        contours = measure.find_contours(mask, 0.8)
        for n, contour in enumerate(contours):
            ax.plot(contour[:, 1], contour[:, 0], linewidth=1)

    ax.imshow(img, cmap='gray')
    color = ['r', 'b', 'g']
    R = [8, 10, 12]
    for k in range(len(minutiae_sets)):
        minutiae = minutiae_sets[k]
        minu_num = len(minutiae)
        for i in range(0, minu_num):
            xx = minutiae[i, 0]
            yy = minutiae[i, 1]
            circ = Circle((xx, yy), R[k], color=color[k],
                          fill=False, linewidth=1.5)
            ax.add_patch(circ)

            ori = -minutiae[i, 2]
            dx = math.cos(ori) * arrow_len
            dy = math.sin(ori) * arrow_len
            ax.arrow(xx, yy, dx, dy, linewidth=1.5, head_width=0.05,
                     head_length=0.1, fc=color[k], ec=color[k])
    plt.close()
    return fig
