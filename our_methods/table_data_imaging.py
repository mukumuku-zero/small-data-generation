import os
import pandas as pd
import umap
from scipy.sparse.csgraph import connected_components
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image

def make_image(train_df, excluded_col, folder_path, diagonal=True):

  scaler = StandardScaler()
  X = train_df.drop(excluded_col, axis=1)
  X_scaled = scaler.fit_transform(X)
  X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

  ump = umap.UMAP(random_state=42)
  embedding = ump.fit_transform(X_scaled_df)
  im = np.dot(embedding[:,0].reshape(-1,1), embedding[:,1].reshape(1,-1))

  print(f'X-axis：{embedding[:,0]}')
  print(f'Y-axis：{embedding[:,1]}')
  print(len(embedding[:,0]), len(embedding[:,1]))

  if diagonal:
    # 対角成分以外を全て0に置き換え
    # Replace all but the diagonal component with 0
    im = np.diag(np.diag(im))
    im_normalized = (im - np.min(im)) / (np.max(im) - np.min(im)) * 255
    im_normalized = np.diag(np.diag(im_normalized))
  else:
    im_normalized = (im - np.min(im)) / (np.max(im) - np.min(im)) * 255

  im_normalized = im_normalized.astype(np.uint8)

  # embeddingの長さが奇数の場合に行と列を追加
  # Add rows and columns when embedding length is odd
  length = len(embedding[:,0])
  if length % 4 != 0:
      add_rows = 4 - (length % 4)  # Number of rows to add (追加する行の数)
      zero_col = np.zeros((im_normalized.shape[0], add_rows))  # 0 of the column to be added (追加する列の0)
      zero_row = np.zeros((add_rows, im_normalized.shape[1] + add_rows))  # 0 for rows to be added (taking into account new columns) (追加する行の0 (新しい列を加味))
      im_normalized = np.hstack((im_normalized, zero_col))  # Add columns (列の追加)
      im_normalized = np.vstack((im_normalized, zero_row))  # Add rows (行の追加)
      im_normalized = im_normalized.astype(np.uint8)

      # print(f'resize ID：{n}')
      print(f'add_row：{add_rows}')

  print('Image data pixel value')
  print(im_normalized)

  # Generate images using PIL (PILを使用して画像を生成)
  image = Image.fromarray(im_normalized, 'L')
    
  #  Show image (optional) (画像を表示（オプション）)
  plt.figure(figsize=(8, 6))
  plt.imshow(image, cmap='gray')  # 'gray' is used for grayscale images
  plt.colorbar()
  plt.title('Visual Representation of the Image')
  plt.show()
  # image.save(f'{folder_path}/umap_image.png')
  image.save(f'{folder_path}')

  if not add_row:
    add_row=0

  return scaler, ump, np.max(im), np.min(im), add_row
