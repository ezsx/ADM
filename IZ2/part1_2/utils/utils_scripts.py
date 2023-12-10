import matplotlib.pyplot as plt

def clear_output_folder():
    import os
    import glob

    files = glob.glob('output/*')
    for f in files:
        os.remove(f)

def show_plot(image, title):
    plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.show()
