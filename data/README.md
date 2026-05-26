# Dataset Directory

This project uses the **Flickr30k** dataset.

## How to set up
1. Ensure you have the [Kaggle CLI](https://github.com/Kaggle/kaggle-api)
    installed.
2. Download the dataset into this folder:
    ```bash
    kaggle datasets download -d srinivasac/flickr30k-dataset
    unzip flickr30k-dataset.zip -d flickr30k-dataset
    ```
3. Your folder structure should look like this:
    ```
    data/
    └── flickr30k-dataset/
        ├── captions.txt
        └── Images/
    ```
