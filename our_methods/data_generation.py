import os
import numpy as np
import math
import matplotlib.pyplot as plt
from PIL import Image
import requests
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose, ToTensor, Lambda, ToPILImage, CenterCrop, Resize
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torch.optim import Adam
from pathlib import Path
import generate_data_component as gdc


def data_generation(folder_path, folder_path_gen, timesteps=300, batch_size=64, channels=1, epochs=50, loss_type="huber")
    image = Image.open(f"{folder_path}")
    width, height = image.size
    
    print(f"Image Size: {width}x{height}")
    
    # Folder path to store generated data
    # 生成データを格納するフォルダのパス   
    if not os.path.exists(f'{folder_path_gen}'):
        os.makedirs(f'{folder_path_gen}')
    else:
        pass
    
    # image_size = 28 # 一番train_dataの量が多い人に合わせる
    print(f'width：{width}')
    image_size = width
    
    # define beta schedule
    betas = gdc.linear_beta_schedule(timesteps=timesteps)
    
    # define alphas
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
    
    # calculations for diffusion q(x_t | x_{t-1}) and others
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
    
    # calculations for posterior q(x_{t-1} | x_t, x_0)
    posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
    
    ################################################################################################
    
    transform = Compose([
      Resize(image_size),
      CenterCrop(image_size),
      ToTensor(), # turn into torch Tensor of shape CHW, divide by 255
      Lambda(lambda t: (t * 2) - 1),
    
    ])
    
    x_start = transform(image).unsqueeze(0)
    
    ################################################################################################
    
    reverse_transform = Compose([
      Lambda(lambda t: (t + 1) / 2),
      # If the tensor is 2D, use unsqueeze(0) to convert it to 3D
      # テンソルが2次元の場合、unsqueeze(0)を使用して3次元に変換
      Lambda(lambda t: t.unsqueeze(0) if t.dim() == 2 else t), # Added to use grayscale images (グレースケールの画像を使用するため追加)
      Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
      Lambda(lambda t: t * 255.),
      Lambda(lambda t: t.numpy().astype(np.uint8)),
      ToPILImage(),
    ])
    
    ################################################################################################
    
    # take time step
    t = torch.tensor([40])
    
    gdc.get_noisy_image(x_start, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, reverse_transform)
    
    ################################################################################################
    
    # 画像変換の定義
    transform = transforms.Compose([
      transforms.RandomHorizontalFlip(),
      transforms.ToTensor(),
      transforms.Lambda(lambda t: (t * 2) - 1)
    ])
    
    img_paths = f'{folder_path}'
    
    # カスタムデータセットのインスタンスを作成
    custom_dataset = gdc.CustomImageDataset(img_paths=img_paths, transform=transform)
    
    # DataLoaderの作成
    dataloader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=True)
    
    ################################################################################################
    
    results_folder = Path("./results")
    results_folder.mkdir(exist_ok = True)
    save_and_sample_every = 100
    
    ################################################################################################
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = gdc.Unet(
      dim=image_size,
      channels=channels,
      dim_mults=(1, 2, 4,)
    )
    model.to(device)
    
    optimizer = Adam(model.parameters(), lr=1e-3)
    
    ################################################################################################
    
    samples = gdc.sample(model, betas, image_size, sqrt_recip_alphas, sqrt_one_minus_alphas_cumprod, posterior_variance, timesteps, batch_size, channels)
    
    ################################################################################################

    # Generated image list
    # 生成された画像一覧

    # Stores image and index correspondence
    # 画像とインデックスの対応を格納
    
    generated_images = []
    
    for n in range(len(samples[-1])):
      image = samples[-1][n].reshape(image_size, image_size, channels)
      generated_images.append((n, image))
    
    # the number of image (イメージの数)
    num_rows = math.sqrt(batch_size)
    num_cols = num_rows
    
    plt.figure(figsize=(12, 6))
    for n in range(num_rows):
      for j in range(num_cols):
          index = n * num_cols + j
          plt.subplot(num_rows, num_cols, index + 1)
          plt.imshow(generated_images[index][1], cmap="gray")
          plt.title(f"Image {generated_images[index][0]}")
          plt.axis('off')
          # plt.savefig(f'{path}/generated_images_{i}_{j}.png')
          # plt.close()
    
    plt.tight_layout()
    plt.show()
    
    ################################################################################################
    
    for j in range(len(samples[-1])):
      image = samples[-1][j].reshape(image_size, image_size, channels)
      image = np.squeeze(image)

      # diagonal component extraction
      # 対角成分抽出
      image = np.diag(np.diag(image))

      # Scales data from 0 to 255
      # データを0から255の範囲にスケーリング

      # normalize using the minimum and maximum values of the data
      # データの最小値と最大値を使用して正規化
      min_val = image.min()
      max_val = image.max()
      scaled_image = (image - min_val) / (max_val - min_val)  # Normalized to a range of 0 to 1 (0から1の範囲に正規化)

      # Multiply by 255 and convert to integer
      # 次に255を掛けて整数に変換
      # scaled_image = (scaled_image * 255).astype(np.uint8)
      scaled_image = (scaled_image * 255)
    
      # diagonal component extraction
      scaled_image = np.diag(np.diag(scaled_image))
      scaled_image = scaled_image.astype(np.uint8)
    
    
      print('Image data pixel value')
      print(scaled_image)
    
      scaled_image = Image.fromarray(scaled_image, 'L')
      plt.figure(figsize=(8, 6))
      plt.imshow(scaled_image, cmap='gray')  # 'gray' is used for grayscale images
      plt.colorbar()
      plt.title('Visual Representation of the Image')
      plt.show()
    
      # 画像を保存
      scaled_image.save(f'{folder_path_gen}/generated_image_{j}.png')
    
    ################################################################################################
