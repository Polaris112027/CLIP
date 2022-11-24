from sentence_transformers import SentenceTransformer, util
from PIL import Image
import glob
import torch
import pickle
import zipfile
from IPython.display import display
from IPython.display import Image as IPImage
import os
from tqdm.autonotebook import tqdm
import torch
import cv2

en_model = SentenceTransformer('clip-ViT-B-32')

simplified_words = []

english_dict='1w.txt'
    
with open(file = english_dict, mode =  'r', encoding='utf-8') as f:
    i = 0
    for line in f:
        i += 1
        if i <= 31:
            continue
        #traditional_words.append(line.split()[0])
        if len(line.split())>=1:
            simplified_words.append(line.split()[0])
            #print(simplified_words) 
    labels=simplified_words
# And compute the text embeddings for these labels
en_emb = en_model.encode(labels, convert_to_tensor=True)
#cos_scores=[[]for i in range(len(labels))]

def function():
    # We use the original CLIP model for computing image embeddings and English text embeddings
    en_model = SentenceTransformer('clip-ViT-B-32')

    # We download some images from our repository which we want to classify
    ##img_names = ['eiffel-tower-day.jpg', 'eiffel-tower-night.jpg', 'two_dogs_in_snow.jpg', 'cat.jpg']
    '''
    img_names = str(img)
    url = 'https://github.com/UKPLab/sentence-transformers/raw/master/examples/applications/image-search/'
    for filename in img_names:
        if not os.path.exists(filename):
            util.http_get(url+filename, filename)
    '''
    # And compute the embeddings for these images
    img_emb = en_model.encode(Image.open('test.png'), convert_to_tensor=True)
    
    #cos_scores=[[]for i in range(len(labels))]
    # Now, we compute the cosine similarity between the images and the labels
    cos_scores = util.cos_sim(img_emb, en_emb)
  
    # Then we look which label has the highest cosine similarity with the given images
    #pred_labels = torch.argmax(cos_scores, dim=1)
    p=[]
    a=[]
    for i in range(5):
        q=torch.argmax(cos_scores, dim=1)
        p.append(q)
        cos_scores[0][q]=0
        
    #multi_model = SentenceTransformer('clip-ViT-B-32-multilingual-v1')

    #return labels[pred_labels]
    for i in range(5):
        a.append(labels[p[i]])
    out=" ".join(a)    
    return out
    # Then, we define our labels as text. Here, we use 4 labels
    ##labels = ['dog', 'dogs', 'cat', 'Paris at night', 'Paris']
    
    #chinese_dict = 'Own.u8'
    
    '''
    file=['adventure.txt','belles_lettres.txt','editorial.txt','fiction.txt','government.txt','hobbies.txt','humor.txt','learned.txt','lore.txt','mystery.txt','news.txt','religion.txt','reviews.txt','romance.txt','science_fiction.txt']

    k=0
    for k in range(15):
            chinese_dict= open(chinese_dict, 'a+', encoding='utf-8')
            with open(file[k], 'r', encoding='utf-8') as f2:
                chinese_dict.write('\n')
                for i in f2:
                    chinese_dict.write(i)
            #merge(file1, file2)
    
    
    
    def merge(file1, file2):
        f1 = open(file1, 'a+', encoding='utf-8')
        with open(file2, 'r', encoding='utf-8') as f2:
            f1.write('\n')
            for i in f2:
                f1.write(i)

    for k in range(15):
        merge(chinese_dict, file[k])




    #traditional_words = []
    simplified_words = []

    english_dict='1w.txt'
    
    with open(file = english_dict, mode =  'r', encoding='utf-8') as f:
        i = 0
        for line in f:
            i += 1
            if i <= 31:
                continue
            #traditional_words.append(line.split()[0])
            if len(line.split())>=1:
                simplified_words.append(line.split()[0])
                #print(simplified_words) 
        labels=simplified_words
        
        #read_data = f.readline()

    
    # And compute the text embeddings for these labels
    en_emb = en_model.encode(labels, convert_to_tensor=True)
   '''
    

    '''
    # Then, we define our labels as text. Here, we use 4 labels
    labels = ['Hund',     # German: dog
              'gato',     # Spanish: cat 
              '多条狗',   # Chinese:dogs
              '巴黎晚上',  # Chinese: Paris at night
              'Париж'     # Russian: Paris
             ]


    # And compute the text embeddings for these labels
    txt_emb = multi_model.encode(labels, convert_to_tensor=True)

    # Now, we compute the cosine similarity between the images and the labels
    cos_scores = util.cos_sim(img_emb, txt_emb)

    # Then we look which label has the highest cosine similarity with the given images
    pred_labels = torch.argmax(cos_scores, dim=1)
    '''
    
    
def ff(name):
    en_model = SentenceTransformer('clip-ViT-B-32')

    img_emb = en_model.encode(Image.open(name), convert_to_tensor=True)

    chinese_dict = 'Own.u8'
    
    simplified_words = []

    # Now, we compute the cosine similarity between the images and the labels
    cos_scores = util.cos_sim(img_emb, en_emb)
  
    p=[]
    a=[]
    for i in range(5):
        q=torch.argmax(cos_scores, dim=1)
        '''
        if cos_scores[0][q]>=0.7:
            p.append(q)
        else:
            p.append(414)
        cos_scores[0][q]=0
        '''
        p.append(q)
        cos_scores[0][q]=0
    #multi_model = SentenceTransformer('clip-ViT-B-32-multilingual-v1')

    #return labels[pred_labels]
    for i in range(5):
        a.append(labels[p[i]])
    out=" ".join(a)    
    return out

def accuracy(a,pic_number):
    #a是给定的答案列表
    #photos是所有图片
    #方案一：需要调用一组测试图片：答案a编码为aa(pic_number X m)。clip-bot的p2文件夹，第一张图片运行ff得到5个out，将out按照空格重新化为列表labels(1x5)，用五次循环：先将该答案编码，再将答案与每个out跟aa的每个答案做进行cos相似度比较（嵌套使用循环，循环次数为该图片的答案个数），超过0.9视为正确，有三个元素正确视为该图片分类成功（最终使用的方案）。运行number次，处理number张图片
    #方案二：需要调用一组测试图片：clip-bot的photos文件夹，n张图片挨个运行fuction得到n个out，每个out有5个单词，组成一个nx5的list名为p。计算一行中每列单词与a中元素的cos相似度，超过0.8视为正确，有三个元素正确视为此行代表的图片分类成功。
    aa=[[] for i in range(len(a))]
    en_model = SentenceTransformer('clip-ViT-B-32')
    for i in range(len(a)):
        for j in range(len(a[i])):
            aa[i].append(en_model.encode(a[i][j], convert_to_tensor=True))
        #aa.append(en_model.encode(a[i], convert_to_tensor=True))
        
    #aa = en_model.encode(a, convert_to_tensor=True)
    cos_scores=[[] for i in range(5)]
    right=success=number=0
    for i in range(pic_number):
        name="./p2/"+str(i)+".jpg"
        labels=ff(name).split( )
        print(number,labels)
        for n in range(5):
            l=en_model.encode(labels[n], convert_to_tensor=True)
            for j in range(len(aa[i])):
                cos_scores[n].append(util.cos_sim(aa[i][j], l))
        #cos_scores = util.cos_sim(aa[number], labels)#cos_socres(mx5) 
        number=number+1
        right=0
        for k in cos_scores:
            for t in k:
                if t>=0.8:
                    right=right+1
        if right>=3:
            success=success+1
        
    nnn=number+ 1   
    acc=success/(nnn)
    return(acc)
                   
 

            
    
        
   
    


    
    
    
  


            
    
        
   
    


    
    
    
  

