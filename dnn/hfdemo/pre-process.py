import numpy as np
import scipy.io
import scipy.stats

from counter import Progress
    
def pairs_to_image(A, B, image_size, PIT=True):
    # Take PIT of inputs
    if PIT:
        A = scipy.stats.norm.cdf(scipy.stats.zscore(A))
        B = scipy.stats.norm.cdf(scipy.stats.zscore(B))
    # Converts pairs into scatter plot bitmap image
    image = np.zeros([image_size, image_size])
    min_A = min(A)
    min_B = min(B)
    max_A = max(A)
    max_B = max(B)
    for (a, b) in zip(A, B):
        # Record the data point in the square it falls in
        image[np.floor(min(image_size - 1, (a - min_A) * image_size / (max_A - min_A))), min(image_size - 1, np.floor((b - min_B) * image_size / (max_B - min_B)))] += 1
    # Normalise
    image = 1.0 * image / len(A)
    # Remove mean
    image = image - np.mean(image)
    # Truncate to 3 std and scale to -1 to 1
    std = 3 * np.std(image)
    image = np.clip(image, -std, std) / std;
    # Rescale from [-1,1] to [0.1,0.9]
    image = (image + 1) * 0.4 + 0.1;
    return image.ravel()

def main(image_size=10):
    print('Loading pairs data and converting to images')
    with open('../../data/training-flipped/CEdata_train_pairs.csv', 'r') as pairs_data_file:
        pairs_header = pairs_data_file.readline()
        pairs_body = pairs_data_file.readlines()
    Inps = np.zeros([len(pairs_body), image_size ** 2])
    prog = Progress(len(pairs_body))
    for (i, line) in enumerate(pairs_body):
        A = np.array([float(a) for a in line.strip().split(',')[1].strip().split(' ')])
        B = np.array([float(b) for b in line.strip().split(',')[2].strip().split(' ')])
        Inps[i,:] = pairs_to_image(A, B, image_size)
        prog.tick()
    prog.done()
    
    print('Loading validation data and converting to images')
    with open('../../data/validation/CEfinal_valid_pairs.csv', 'r') as valid_data_file:
        valid_header = valid_data_file.readline()
        valid_body = valid_data_file.readlines()
    validInps = np.zeros([len(valid_body), image_size ** 2])
    prog = Progress(len(valid_body))
    for (i, line) in enumerate(valid_body):
        A = np.array([float(a) for a in line.strip().split(',')[1].strip().split(' ')])
        B = np.array([float(b) for b in line.strip().split(',')[2].strip().split(' ')])
        validInps[i,:] = pairs_to_image(A, B, image_size)
        prog.tick()
    prog.done()
    
    print('Saving data to MATLAB format')
    
    scipy.io.savemat('images_10_pit.mat', {'train_images' : Inps, 'valid_images' : validInps})

if __name__ == "__main__":
    main()
