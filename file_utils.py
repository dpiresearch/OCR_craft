# -*- coding: utf-8 -*-
import os
import numpy as np
import cv2
import pytesseract
import imgproc

# borrowed from https://github.com/lengstrom/fast-style-transfer/blob/master/src/utils.py
def get_files(img_dir):
    imgs, masks, xmls = list_files(img_dir)
    return imgs, masks, xmls

def list_files(in_path):
    img_files = []
    mask_files = []
    gt_files = []
    for (dirpath, dirnames, filenames) in os.walk(in_path):
        for file in filenames:
            filename, ext = os.path.splitext(file)
            ext = str.lower(ext)
            if ext == '.jpg' or ext == '.jpeg' or ext == '.gif' or ext == '.png' or ext == '.pgm':
                img_files.append(os.path.join(dirpath, file))
            elif ext == '.bmp':
                mask_files.append(os.path.join(dirpath, file))
            elif ext == '.xml' or ext == '.gt' or ext == '.txt':
                gt_files.append(os.path.join(dirpath, file))
            elif ext == '.zip':
                continue
    # img_files.sort()
    # mask_files.sort()
    # gt_files.sort()
    return img_files, mask_files, gt_files

def saveResult(img_file, img, boxes, dirname='./result/', verticals=None, texts=None):
        """ save text detection result one by one
        Args:
            img_file (str): image file name
            img (array): raw image context
            boxes (array): array of result file
                Shape: [num_detections, 4] for BB output / [num_detections, 4] for QUAD output
        Return:
            None
        """
        # set padding size
        padding_size = 50

        # padd with black pixels
        padded_image = np.zeros((img.shape[0] + 2 * padding_size, img.shape[1] + 2 * padding_size, 3),
                                dtype=np.uint8)
        padded_image[padding_size:padding_size + img.shape[0], padding_size:padding_size + img.shape[1]] = img

        image_orig = padded_image.copy()
        img = np.array(img)

        # make result file list
        filename, file_ext = os.path.splitext(os.path.basename(img_file))

        # result directory
        res_file = dirname + "res_" + filename + '.txt'
        res_img_file = dirname + "res_" + filename + '.jpg'
        res_img_corr_file = dirname + "res_corr_" + filename + '.jpg'

        if not os.path.isdir(dirname):
            os.mkdir(dirname)

        angled_rois = []
        with open(res_file, 'w') as f:
            for i, box in enumerate(boxes):
                poly = np.array(box).astype(np.int32).reshape((-1))
                strResult = ','.join([str(p) for p in poly]) + '\r\n'
                f.write(strResult)

                poly = poly.reshape(-1, 2)
                cv2.polylines(img, [poly.reshape((-1, 1, 2))], True, color=(0, 0, 255), thickness=2)
                ptColor = (0, 255, 255)
                if verticals is not None:
                    if verticals[i]:
                        ptColor = (255, 0, 0)

                if texts is not None:
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.5
                    cv2.putText(img, "{}".format(texts[i]), (poly[0][0]+1, poly[0][1]+1), font, font_scale, (0, 0, 0), thickness=1)
                    cv2.putText(img, "{}".format(texts[i]), tuple(poly[0]), font, font_scale, (0, 255, 255), thickness=1)

                rotated_rect = cv2.minAreaRect(poly)

                # get vertices of box
                box_vertices = cv2.boxPoints(rotated_rect)
                box_vertices = np.int0(box_vertices)

                # append box_vertices on angled_rois list
                angled_rois.append(box_vertices)

        if angled_rois:
            print("We have angled rois")

            # get largest roi
            largest_roi = max(angled_rois, key=cv2.contourArea)

            # draw rectangle box around texts
            # cv2.drawContours(img, [largest_roi], 0, (255, 0, 0), 2)

            # get fitted line
            [vx, vy, x, y] = cv2.fitLine(largest_roi, cv2.DIST_L2, 0, 0.01, 0.01)

            # calculate start and end points for the line to be drawn
            lefty = int((-x * vy / vx) + y)
            righty = int(((img.shape[1] - x) * vy / vx) + y)

            # fit  line
            # cv2.line(img, (0, lefty), (img.shape[1] - 1, righty), (0, 255, 0), 2)

            # get angle
            angle_rad = np.arctan2(vy, vx)
            angle = np.degrees(angle_rad)[0]

            print("Angle: ", angle)

            height, width = image_orig.shape[:2]
            center = (width // 2, height // 2)

            # if angle made by roi is 90 degree then no rotation needed
            if angle != 90:
                rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale=1.0)
                img_new = cv2.warpAffine(image_orig, rotation_matrix, (width, height))
                print("Angle correction needed")
                cv2.imwrite(res_img_corr_file, img_new)

                text_line = f'Line : ' + pytesseract.image_to_string(img_new, lang='eng')
                print(text_line)

                # Save result image
        cv2.imwrite(res_img_file, img)

