import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import numpy as np
import os
import json
import re


def pre_caption(caption, max_words, icfg_rstp=False):
    if icfg_rstp:
        try:
            caption_new = re.sub(
                r'[^0-9a-z]+',
                ' ',
                caption.lower(),
            )
        except:
            print('Wrong: when deal with', caption)
        caption_words = caption_new.split()
        caption_new = ' '.join(caption_words)
    else:
        caption_new = caption

    # truncate caption
    caption_words = caption_new.split()
    if len(caption_words) > max_words:
        caption_new = ' '.join(caption_words[:max_words])

    if not len(caption_new):
        raise ValueError("pre_caption yields invalid text")

    return caption_new


def save_numpy_array(array, file_path):
    np.save(file_path, array)
    print("NumPy数组已保存到:", file_path)


def load_numpy_array(file_path):
    array = np.load(file_path)
    print("从文件加载NumPy数组:", file_path)
    return array


def load_image(image_path):
    try:
        image = Image.open(image_path)
        print("图像已加载:", image_path)
        return image
    except Exception as e:
        print("加载图像时出现错误:", e)
        return None


def combine_images_with_border(image_paths, output_path, border_colors):
    """
    合成多张图片为一张图片，并为每张图片设置不同颜色的边框

    Parameters:
        image_paths (list): 包含所有图片路径的列表
        output_path (str): 输出图片路径
        border_colors (list): 包含每张图片对应边框颜色的列表
    """
    # 打开所有图片并获取宽度和高度
    images = [Image.open(image_path) for image_path in image_paths]
    # 调整图片大小为 256x384
    resized_images = [image.resize((256, 384)) for image in images]

    # 计算合成后的图片宽度和高度
    total_width = sum(image.width for image in resized_images) + 150
    max_height = max(image.height for image in resized_images) + 20

    # 创建一个空白的图片对象
    combined_image = Image.new('RGB', (total_width, max_height), (255, 255, 255))

    # 将每张图片粘贴到合成图片中并添加边框
    x_offset = 0
    for resized_image, border_color in zip(resized_images, border_colors):
        # 计算边框大小
        border_size = 5

        # 添加边框
        bordered_image = ImageOps.expand(resized_image, border=border_size, fill=border_color)

        # 将带有边框的图片粘贴到合成图片中
        combined_image.paste(bordered_image, (x_offset, 0))
        x_offset += bordered_image.size[0] + 5

    # 保存合成图片
    combined_image.save(output_path)
    print("图片合成完成，已保存到:", output_path)


def load_data():
    ann_file = '/home/wjh/project/CAMeL/data/finetune/cuhk_test.json'
    anns = json.load(open(ann_file, 'r'))

    text = []
    image = []
    g_pids = []
    q_pids = []
    for img_id, ann in enumerate(anns):
        g_pids.append(ann['image_id'])
        image.append(ann['image'])
        for i, caption in enumerate(ann['caption']):
            q_pids.append(ann['image_id'])
            text.append(pre_caption(caption, 56))

    return image, text, g_pids, q_pids


def visualize_retrieval_results(scores_ft, scores_base, output_dir, top_k=5):
    imgs, texts, g_pids, q_pids = load_data()
    image_root = '/home/wjh/project/CAMeL/images/CUHK-PEDES'

    save_path = os.path.join(output_dir, "visualize_retrieval_results/")
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    colors = {'T': (0, 255, 0), 'F': (255, 0, 0)}
    results = []

    top_indices_ft = np.argsort(scores_ft, axis=1)[:, ::-1][:, :top_k]
    top_indices_base = np.argsort(scores_base, axis=1)[:, ::-1][:, :top_k]


    # for i in range(len(texts)):
    for i in range(1000):

        query_text = texts[i]

        if g_pids[top_indices_ft[i][0]] == q_pids[i] and g_pids[top_indices_base[i][0]] != q_pids[i]:

            results.append("index: " + str(i))
            results.append("Query: " + query_text)

            re_imgs = []
            re_imgs_id = []
            results.append("ft results:")
            for j, idx in enumerate(top_indices_ft[i]):
                image = os.path.join(image_root, imgs[idx])
                re_imgs.append(image)

                if g_pids[idx] == q_pids[i]:
                    re_imgs_id.append('T')
                    results.append('R' + str(j) + ': T --> ' + imgs[idx])
                else:
                    re_imgs_id.append('F')
                    results.append('R' + str(j) + ': F --> ' + imgs[idx])
            results.append('')
            border_colors = [colors[temp] for temp in re_imgs_id]  # 每张图片对应的边框颜色
            combine_images_with_border(re_imgs, save_path + str(i) + "_ft.jpg", border_colors)

            re_imgs = []
            re_imgs_id = []
            results.append("base results:")
            for j, idx in enumerate(top_indices_base[i]):
                image = os.path.join(image_root, imgs[idx])
                re_imgs.append(image)

                if g_pids[idx] == q_pids[i]:
                    re_imgs_id.append('T')
                    results.append('R' + str(j) + ': T --> ' + imgs[idx])
                else:
                    re_imgs_id.append('F')
                    results.append('R' + str(j) + ': F --> ' + imgs[idx])
            results.append('')
            results.append('')
            border_colors = [colors[temp] for temp in re_imgs_id]  # 每张图片对应的边框颜色
            combine_images_with_border(re_imgs, save_path + str(i) + "_base.jpg", border_colors)



    file_path = output_dir + '/re.txt'
    try:
        with open(file_path, 'a') as file:  # 'a' 模式表示追加模式，如果文件不存在则创建
            for string in results:
                file.write(string + '\n')  # 写入字符串并添加换行符
        print("字符串已成功写入文件:", file_path)
    except Exception as e:
        print("写入文件时出现错误:", e)


if __name__ == '__main__':
    out_ft = '/home/wjh/project/CAMeL/output/ft_cuhk/test' 
    

    out_base = '/home/wjh/project/CAMeL/output/ft_cuhk/test_1' 

    out = '/home/wjh/project/CAMeL/out/test2'

    loaded_array_ft = load_numpy_array(os.path.join(out_ft, "my_array.npy"))
    loaded_array_base = load_numpy_array(os.path.join(out_base, "my_array.npy"))

    visualize_retrieval_results(loaded_array_ft, loaded_array_base, out)

    # 保存NumPy数组

    # save_numpy_array(my_array, "my_array.npy")
