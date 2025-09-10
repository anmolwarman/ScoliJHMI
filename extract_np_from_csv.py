import pandas as pd
import numpy as np
import os

label_csv_path='/mnt/c/Users/swapnil/Downloads/XRAY_Dataset/Final_JH_UCSF_Stan_WU.csv'

jhu_data_dir='/mnt/c/Users/swapnil/Downloads/XRAY_Dataset/XRAY_JHU/JPG'
ucsf_data_dir='/mnt/c/Users/swapnil/Downloads/XRAY_Dataset/XRAY_UCSF/UCSF'
washu_data_dir='/mnt/c/Users/swapnil/Downloads/XRAY_Dataset/XRAY_WashU/final_images'

df=pd.read_csv(label_csv_path)


# jh_mask=df['Study ID'].str.startswith('JH')
# us_mask=df['Study ID'].str.startswith('US')
# wu_mask=df['Study ID'].str.startswith('WU')

# jh_original_count = jh_mask.sum()
# us_original_count = us_mask.sum()
# wu_original_count = wu_mask.sum()

# print(jh_original_count)
# print(us_original_count)
# print(wu_original_count)
# # First filter by institution, then remove NaN ODI values
# jh_df = df.loc[jh_mask].dropna(subset=['ODI'])
# us_df = df.loc[us_mask].dropna(subset=['ODI'])
# wu_df = df.loc[wu_mask].dropna(subset=['ODI'])

# print(jh_df.shape)
# print(us_df.shape)
# print(wu_df.shape)

# jh_odi = jh_df['ODI'].to_numpy()
# us_odi = us_df['ODI'].to_numpy()
# wu_odi = wu_df['ODI'].to_numpy()

# jh_ids=jh_df['Study ID'].tolist()
# us_ids=us_df['Study ID'].tolist()
# wu_ids=wu_df['Study ID'].tolist()

# jh_ids=[item for item in jh_ids for _ in range(2)]
# us_ids=[item for item in us_ids for _ in range(2)]
# wu_ids=[item for item in wu_ids for _ in range(2)]
# print(len(jh_ids))
# print(len(us_ids))
# print(len(wu_ids))

# jhu_image_names=[img.split('_')[0] for img_folder in sorted(os.listdir(jhu_data_dir)) for img in os.listdir(os.path.join(jhu_data_dir,img_folder)) ]
# ucsf_image_names=[img.split(' ')[0] for img_folder in sorted(os.listdir(ucsf_data_dir)) for img in os.listdir(os.path.join(ucsf_data_dir,img_folder)) ]
# wu_image_names=[img.split(' ')[0] for img_folder in sorted(os.listdir(washu_data_dir)) for img in os.listdir(os.path.join(washu_data_dir,img_folder))]
# print(len(jhu_image_names))
# print(len(ucsf_image_names))
# print(len(wu_image_names))
# # for i in range(len(wu_image_names)):
# #     print(i, wu_ids[i],wu_image_names[i], wu_ids[i]==wu_image_names[i])

jhu_image_names=[img_folder.split('_')[0] for img_folder in sorted(os.listdir(jhu_data_dir))  ]
ucsf_image_names=[img_folder for img_folder in sorted(os.listdir(ucsf_data_dir))  ]
wu_image_names=[img_folder for img_folder in sorted(os.listdir(washu_data_dir)) ]

def find_missing_and_odi(image_folder_name):
    jh_odi=[]
    jh_missing_ids=[]
    print(len(image_folder_name))
    for folder_name in image_folder_name:
        matching_rows =df[df['Study ID']==folder_name]
        print(folder_name,matching_rows.index)
        if len(matching_rows)==0:
            jh_missing_ids.append(folder_name)
            continue
        odi_value=matching_rows['ODI'].values[0]
        if pd.isna(odi_value):
            jh_missing_ids.append(folder_name)
            continue
        jh_odi.append(odi_value)
    np.save(f'washu.npy',np.array(jh_odi))
    print(jh_missing_ids)

find_missing_and_odi(wu_image_names)