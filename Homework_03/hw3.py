# Motion Estimation Homework 3
# Implement Full Search, 2D Logarithmic Search, and Three-Step Search algorithms
# Complete Tasks 1 ~ 4 as specified in the assignment (motion estimation and analysis)

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import time
import os

# Load a grayscale image from file
# Input: file path
# Output: numpy array (float32)
def load_image_gray(path):
    img = Image.open(path).convert('L')
    return np.array(img, dtype=np.float32)

# Save a numpy array as an image file
def save_image(array, path):
    img = Image.fromarray(np.clip(array, 0, 255).astype(np.uint8))
    img.save(path)

# Compute Sum of Absolute Differences (SAD) between two blocks
def compute_SAD(block1, block2):
    return np.sum(np.abs(block1 - block2))

# Compute Peak Signal-to-Noise Ratio (PSNR) between two images
def compute_PSNR(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    PIXEL_MAX = 255.0
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))

# ======== Full Search algorithm (Exhaustive Search) ========
def full_search(ref, target, block_size, p):
    h, w = ref.shape
    mv_field = np.zeros((h//block_size, w//block_size, 2), dtype=np.int32)
    pred = np.zeros_like(ref)

    # Slide macroblocks and search exhaustively
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            min_sad = float('inf')
            best_dx, best_dy = 0, 0
            block_t = target[i:i+block_size, j:j+block_size]

            for dy in range(-p, p+1):
                for dx in range(-p, p+1):
                    ref_i = i + dy
                    ref_j = j + dx
                    if (0 <= ref_i < h - block_size +1) and (0 <= ref_j < w - block_size +1):
                        block_r = ref[ref_i:ref_i+block_size, ref_j:ref_j+block_size]
                        sad = compute_SAD(block_t, block_r)
                        if sad < min_sad:
                            min_sad = sad
                            best_dx, best_dy = dx, dy

            mv_field[i//block_size, j//block_size] = [best_dy, best_dx]
            ref_i = i + best_dy
            ref_j = j + best_dx
            pred[i:i+block_size, j:j+block_size] = ref[ref_i:ref_i+block_size, ref_j:ref_j+block_size]
    return mv_field, pred

# ======== 2D Logarithmic Search algorithm ========
def log_search(ref, target, block_size, p):
    h, w = ref.shape
    mv_field = np.zeros((h//block_size, w//block_size, 2), dtype=np.int32)
    pred = np.zeros_like(ref)

    # Define search positions for log search (center + 4 directions)
    def search_positions(center, step):
        dy, dx = center
        return [(dy, dx), (dy - step, dx), (dy + step, dx), (dy, dx - step), (dy, dx + step)]

    # Slide macroblocks and perform logarithmic search
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block_t = target[i:i+block_size, j:j+block_size]
            center = (0, 0)
            step = 2 ** int(np.floor(np.log2(p)))

            best_center = center
            min_sad = float('inf')

            while step >= 1:
                for dy, dx in search_positions(best_center, step):
                    ref_i = i + dy
                    ref_j = j + dx
                    if (-p <= dy <= p and -p <= dx <= p and
                        0 <= ref_i < h - block_size +1 and 0 <= ref_j < w - block_size +1):
                        block_r = ref[ref_i:ref_i+block_size, ref_j:ref_j+block_size]
                        sad = compute_SAD(block_t, block_r)
                        if sad < min_sad:
                            min_sad = sad
                            best_center = (dy, dx)
                step //= 2

            best_dy, best_dx = best_center
            mv_field[i//block_size, j//block_size] = [best_dy, best_dx]
            ref_i = i + best_dy
            ref_j = j + best_dx
            pred[i:i+block_size, j:j+block_size] = ref[ref_i:ref_i+block_size, ref_j:ref_j+block_size]
    return mv_field, pred

# ======== Three-Step Search algorithm ========
def three_step_search(ref, target, block_size, p):
    h, w = ref.shape
    mv_field = np.zeros((h//block_size, w//block_size, 2), dtype=np.int32)
    pred = np.zeros_like(ref)

    # Slide macroblocks and perform three-step search
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block_t = target[i:i+block_size, j:j+block_size]
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
                            0 <= ref_i < h - block_size +1 and 0 <= ref_j < w - block_size +1):
                            block_r = ref[ref_i:ref_i+block_size, ref_j:ref_j+block_size]
                            sad = compute_SAD(block_t, block_r)
                            if sad < min_sad:
                                min_sad = sad
                                best_center = (test_dy, test_dx)
                step //= 2

            best_dy, best_dx = best_center
            mv_field[i//block_size, j//block_size] = [best_dy, best_dx]
            ref_i = i + best_dy
            ref_j = j + best_dx
            pred[i:i+block_size, j:j+block_size] = ref[ref_i:ref_i+block_size, ref_j:ref_j+block_size]
    return mv_field, pred

# Draw motion vectors on the target image and save as figure
def draw_motion_vectors(target, mv_field, block_size, save_path):
    plt.figure(figsize=(8,6))
    plt.imshow(target, cmap='gray')
    h_blocks, w_blocks, _ = mv_field.shape
    for i in range(h_blocks):
        for j in range(w_blocks):
            dy, dx = mv_field[i,j]
            y = i * block_size + block_size // 2
            x = j * block_size + block_size // 2
            plt.arrow(x, y, dx, dy, color='red', head_width=2, head_length=3)
    plt.axis('off')
    plt.savefig(save_path)
    plt.close()

# Compute residual image for visualization
def compute_residual(target, predicted):
    residual = np.clip(target - predicted + 128, 0, 255)
    return residual

# Wrapper to execute search, compute outputs, save images, and return metrics
def run_and_save_results(ref_img, target_img, method_name, method_fn, block_size, p, output_prefix):
    start = time.time()
    mv, pred = method_fn(ref_img, target_img, block_size, p)
    elapsed = time.time() - start

    residual = compute_residual(target_img, pred)
    total_SAD = compute_SAD(target_img, pred)
    psnr = compute_PSNR(target_img, pred)

    save_image(pred, f'results/predicted/{output_prefix}.png')
    draw_motion_vectors(target_img, mv, block_size, f'results/motion_vectors/{output_prefix}.png')
    save_image(residual, f'results/residual/{output_prefix}.png')

    return total_SAD, psnr, elapsed

# Ensure all output directories exist
def ensure_dirs():
    os.makedirs('results/predicted', exist_ok=True)
    os.makedirs('results/motion_vectors', exist_ok=True)
    os.makedirs('results/residual', exist_ok=True)

# ========== Task 1 ==========
# Compare Full Search and Log Search (p=8/16, block=8/16)
def task1():
    ref = load_image_gray('img/008.jpg')
    target = load_image_gray('img/009.jpg')

    methods = {'full': full_search, 'log': log_search}
    block_sizes = [8, 16]
    ps = [8, 16]

    all_results = []

    for method_name, method_fn in methods.items():
        for block_size in block_sizes:
            for p in ps:
                prefix = f'{method_name}_p{p}_b{block_size}'
                sad, psnr, time_cost = run_and_save_results(ref, target, method_name, method_fn, block_size, p, prefix)
                all_results.append((method_name, block_size, p, sad, psnr, time_cost))
                print(f'{prefix}: SAD={sad}, PSNR={psnr:.2f} dB, Time={time_cost:.3f}s')

    return all_results

# ========== Task 2 ==========
def task2():
    methods = {'full': full_search, 'log': log_search, 'three': three_step_search}
    block_size = 16
    p = 8

    ref = load_image_gray('img/000.jpg')  # Reference image is fixed

    SAD_curves = {k: [] for k in methods}
    PSNR_curves = {k: [] for k in methods}

    for idx in range(1, 18):
        target = load_image_gray(f'img/{idx:03d}.jpg')
        for method_name, method_fn in methods.items():
            sad, psnr, _ = run_and_save_results(ref, target, method_name, method_fn, block_size, p, f'{method_name}_seq{idx:03d}')
            SAD_curves[method_name].append(sad)
            PSNR_curves[method_name].append(psnr)

    os.makedirs('results/plots', exist_ok=True)

    # Plot Total SAD curve
    plt.figure()
    for method in methods:
        plt.plot(SAD_curves[method], label=method)
    plt.xlabel('Frame index')
    plt.ylabel('Total SAD')
    plt.title('Total SAD vs Frame Index (Task 2)')
    plt.legend()
    plt.savefig('results/plots/SAD_curve.png')
    plt.close()

    # Plot PSNR curve
    plt.figure()
    for method in methods:
        plt.plot(PSNR_curves[method], label=method)
    plt.xlabel('Frame index')
    plt.ylabel('PSNR (dB)')
    plt.title('PSNR vs Frame Index (Task 2)')
    plt.legend()
    plt.savefig('results/plots/PSNR_curve.png')
    plt.close()

    print('Task 2 completed: SAD and PSNR curves saved.')

# ========== Task 3 ==========
# Compare 2D Log Search between 008.jpg and 012.jpg (p=8, block=16)
def task3():
    ref = load_image_gray('img/008.jpg')
    target = load_image_gray('img/012.jpg')

    block_size = 16
    p = 8

    sad, psnr, time_cost = run_and_save_results(ref, target, 'log', log_search, block_size, p, 'log_008_012')

    print(f'Task 3 result (Log Search, p=8, block=16): SAD={sad}, PSNR={psnr:.2f} dB, Time={time_cost:.3f}s')
    return sad, psnr

# ========== Task 4 ==========
# Time complexity analysis: measure runtime for all 3 methods at p=8 and p=16
def task4():
    methods = {'full': full_search, 'log': log_search, 'three': three_step_search}
    block_size = 16
    ps = [8, 16]

    ref = load_image_gray('img/008.jpg')
    target = load_image_gray('img/009.jpg')

    for method_name, method_fn in methods.items():
        for p in ps:
            start = time.time()
            method_fn(ref, target, block_size, p)
            elapsed = time.time() - start
            print(f'Task 4 - {method_name} search (p={p}): Time = {elapsed:.3f} seconds')

# ========== Main execution ==========
if __name__ == '__main__':
    ensure_dirs()
    print('Running Task 1...')
    task1()
    print('Running Task 2...')
    task2()
    print('Running Task 3...')
    task3()
    print('Running Task 4...')
    task4()
