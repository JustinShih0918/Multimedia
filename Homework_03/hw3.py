import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import time
import os
import cv2

# Load a color image as RGB
def load_image_color(path):
    img = Image.open(path)
    return np.array(img, dtype=np.float32)

def save_image(array, path):
    img = Image.fromarray(np.clip(array, 0, 255).astype(np.uint8))
    img.save(path)

# Compute SAD between two blocks
def compute_SAD(block1, block2):
    return np.sum(np.abs(block1 - block2))

# Compute PSNR between two images
def compute_PSNR(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    PIXEL_MAX = 255.0
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))

# ======== Full Search algorithm ========
def full_search(ref, target, block_size, p):
    h, w, c = ref.shape
    mv_field = np.zeros((h // block_size, w // block_size, 2), dtype=np.int32)
    pred = np.zeros_like(ref)

    # Slide macroblocks and search exhaustively
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            min_sad = float('inf')
            best_dx, best_dy = 0, 0
            block_t = target[i:i+block_size, j:j+block_size, :]

            for dy in range(-p, p+1):
                for dx in range(-p, p+1):
                    ref_i = i + dy
                    ref_j = j + dx
                    if (0 <= ref_i <= h - block_size) and (0 <= ref_j <= w - block_size):
                        block_r = ref[ref_i:ref_i+block_size, ref_j:ref_j+block_size, :]
                        sad = compute_SAD(block_t, block_r)
                        if sad < min_sad:
                            min_sad = sad
                            best_dx, best_dy = dx, dy

            mv_field[i//block_size, j//block_size] = [best_dy, best_dx]
            ref_i = i + best_dy
            ref_j = j + best_dx
            pred[i:i+block_size, j:j+block_size, :] = ref[ref_i:ref_i+block_size, ref_j:ref_j+block_size, :]
    return mv_field, pred

# ======== 2D Logarithmic Search algorithm ========
def log_search(ref, target, block_size, p):
    h, w, c = ref.shape
    mv_field = np.zeros((h // block_size, w // block_size, 2), dtype=np.int32)
    pred = np.zeros_like(ref)

    # Define search positions for log search (center + 4 directions)
    def search_positions(center, step):
        dy, dx = center
        return [(dy, dx), (dy - step, dx), (dy + step, dx), (dy, dx - step), (dy, dx + step)]

    # Slide macroblocks and perform logarithmic search
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block_t = target[i:i+block_size, j:j+block_size, :]
            center = (0, 0)
            step = 2 ** int(np.floor(np.log2(p)))

            best_center = center
            min_sad = float('inf')

            while step >= 1:
                for dy, dx in search_positions(best_center, step):
                    ref_i = i + dy
                    ref_j = j + dx
                    if (-p <= dy <= p and -p <= dx <= p and
                        0 <= ref_i <= h - block_size and 0 <= ref_j <= w - block_size):
                        block_r = ref[ref_i:ref_i+block_size, ref_j:ref_j+block_size, :]
                        sad = compute_SAD(block_t, block_r)
                        if sad < min_sad:
                            min_sad = sad
                            best_center = (dy, dx)
                step //= 2

            best_dy, best_dx = best_center
            mv_field[i//block_size, j//block_size] = [best_dy, best_dx]
            ref_i = i + best_dy
            ref_j = j + best_dx
            pred[i:i+block_size, j:j+block_size, :] = ref[ref_i:ref_i+block_size, ref_j:ref_j+block_size, :]
    return mv_field, pred

# ======== Three-Step Search algorithm ========
def three_step_search(ref, target, block_size, p):
    h, w, c = ref.shape
    mv_field = np.zeros((h // block_size, w // block_size, 2), dtype=np.int32)
    pred = np.zeros_like(ref)

    # Slide macroblocks and perform three-step search
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block_t = target[i:i+block_size, j:j+block_size, :]
            center = (0, 0)
            step = p // 2
            best_center = center
            min_sad = float('inf')

            while step >= 1:
                for dy in [-step, 0, step]:
                    for dx in [-step, 0, step]:
                        test_dy = best_center[0] + dy
                        test_dx = best_center[1] + dx
                        ref_i = i + test_dy
                        ref_j = j + test_dx
                        if (-p <= test_dy <= p and -p <= test_dx <= p and
                            0 <= ref_i <= h - block_size and 0 <= ref_j <= w - block_size):
                            block_r = ref[ref_i:ref_i+block_size, ref_j:ref_j+block_size, :]
                            sad = compute_SAD(block_t, block_r)
                            if sad < min_sad:
                                min_sad = sad
                                best_center = (test_dy, test_dx)
                step //= 2

            best_dy, best_dx = best_center
            mv_field[i//block_size, j//block_size] = [best_dy, best_dx]
            ref_i = i + best_dy
            ref_j = j + best_dx
            pred[i:i+block_size, j:j+block_size, :] = ref[ref_i:ref_i+block_size, ref_j:ref_j+block_size, :]
    return mv_field, pred

# Draw motion vectors on the target image and save as figure using OpenCV
def draw_motion_vectors(target, mv_field, block_size, save_path):
    target_img = np.clip(target, 0, 255).astype(np.uint8)
    
    output_img = target_img.copy()
    
    h_blocks, w_blocks, _ = mv_field.shape
    for i in range(h_blocks):
        for j in range(w_blocks):
            dy, dx = mv_field[i, j]
            y = i * block_size + block_size // 2
            x = j * block_size + block_size // 2
            end_y = int(y + dy)
            end_x = int(x + dx)
            
            cv2.arrowedLine(output_img, (x, y), (end_x, end_y), color=(255, 0, 0), thickness=1, tipLength=0.3)
    
    cv2.imwrite(save_path, cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR))

# Compute residual image
def compute_residual(target, predicted):
    return np.clip(np.abs(target - predicted), 0, 255)

def run_and_save_results(ref_img, target_img, method_name, method_fn, block_size, p, output_prefix):
    start = time.time()
    mv, pred = method_fn(ref_img, target_img, block_size, p)
    elapsed = time.time() - start

    residual = compute_residual(target_img, pred)

    save_image(pred, f'out/{method_name}_predicted_r{p}_b{block_size}.jpg')
    draw_motion_vectors(target_img, mv, block_size, f'out/{method_name}_motion_vector_r{p}_b{block_size}.jpg')
    save_image(residual, f'out/{method_name}_residual_r{p}_b{block_size}.jpg')

    total_SAD = compute_SAD(target_img, pred)
    psnr = compute_PSNR(target_img, pred)
    return total_SAD, psnr, elapsed

# ========== Task 1 ==========
def task1():
    ref = load_image_color('img/008.jpg')
    target = load_image_color('img/009.jpg')

    methods = {'full': full_search, 'log': log_search}
    block_sizes = [8, 16]
    ps = [8, 16]

    all_results = []

    for method_name, method_fn in methods.items():
        for block_size in block_sizes:
            for p in ps:
                prefix = f'{method_name}_r{p}_b{block_size}'
                sad, psnr, time_cost = run_and_save_results(ref, target, method_name, method_fn, block_size, p, prefix)
                all_results.append((method_name, block_size, p, sad, psnr, time_cost))
                print(f'{prefix}: SAD={sad}, PSNR={psnr:.2f} dB, Time={time_cost:.3f}s')

    return all_results

# ========== Task 2 ==========
def task2():
    methods = {'full': full_search, 'log': log_search, 'three': three_step_search}
    block_size = 16
    p = 8

    ref = load_image_color('img/000.jpg') 

    SAD_curves = {k: [] for k in methods}
    PSNR_curves = {k: [] for k in methods}

    for idx in range(1, 18):
        target = load_image_color(f'img/{idx:03d}.jpg')
        for method_name, method_fn in methods.items():
            sad, psnr, _ = run_and_save_results(ref, target, method_name, method_fn, block_size, p, f'{method_name}_seq{idx:03d}')
            SAD_curves[method_name].append(sad)
            PSNR_curves[method_name].append(psnr)

    os.makedirs('out/plots', exist_ok=True)

    plt.figure()
    for method in methods:
        plt.plot(SAD_curves[method], label=method)
    plt.xlabel('Frame index')
    plt.ylabel('Total SAD')
    plt.title('Total SAD vs Frame Index (Task 2)')
    plt.legend()
    plt.savefig('out/plots/SAD_curve.png')
    plt.close()

    # Plot PSNR curve
    plt.figure()
    for method in methods:
        plt.plot(PSNR_curves[method], label=method)
    plt.xlabel('Frame index')
    plt.ylabel('PSNR (dB)')
    plt.title('PSNR vs Frame Index (Task 2)')
    plt.legend()
    plt.savefig('out/plots/PSNR_curve.png')
    plt.close()

    print('Task 2 completed: SAD and PSNR curves saved.')

# ========== Task 3 ==========
def task3():
    ref = load_image_color('img/008.jpg')
    target = load_image_color('img/012.jpg')

    block_size = 16
    p = 8

    sad, psnr, time_cost = run_and_save_results(ref, target, 'log', log_search, block_size, p, 'log_008_012')

    print(f'Task 3 result (Log Search, p=8, block=16): SAD={sad}, PSNR={psnr:.2f} dB, Time={time_cost:.3f}s')
    return sad, psnr

# ========== Task 4 ==========
def task4():
    methods = {'full': full_search, 'log': log_search, 'three': three_step_search}
    block_size = 16
    ps = [8, 16]

    ref = load_image_color('img/008.jpg')
    target = load_image_color('img/009.jpg')

    for method_name, method_fn in methods.items():
        for p in ps:
            start = time.time()
            method_fn(ref, target, block_size, p)
            elapsed = time.time() - start
            print(f'Task 4 - {method_name} search (p={p}): Time = {elapsed:.3f} seconds')

# ========== Main execution ==========
if __name__ == '__main__':
    os.makedirs('out', exist_ok=True)
    print("Task 1:")
    task1()
    print("Task 2:")
    task2()
    print("Task 3:")
    task3()
    print("Task 4:")
    task4()
