
# coding: utf-8

# In[1]:


from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import os


# In[2]:


datagen = ImageDataGenerator(
        shear_range=0.1,
        zoom_range=0.1,
        rotation_range=10.,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest')


# In[6]:


imglist=[]
for folder in os.listdir('./CUB_200_2011/299/train/'):
    for imgpath in os.listdir(os.path.join('./CUB_200_2011/299/train',folder)):
        imglist.append(os.path.join('./CUB_200_2011/299/train/',folder,imgpath))


# In[7]:


len(imglist)


# In[8]:


for imgpath in imglist:
    img = load_img(imgpath)
    x = img_to_array(img)
    x = x.reshape((1,) + x.shape)
    savepath='/'.join(imgpath.split('/')[:-1])
    i = 0
#     print(savepath)
    for batch in datagen.flow(x, 
                              batch_size=1,
                              save_to_dir=savepath,  
                              save_prefix='enhance%s'%i, 
                              save_format='jpg'):
        i += 1
        if i > 5:
            break  # otherwise the generator would loop indefinitely

