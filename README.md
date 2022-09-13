
# Contrastive Language-Image Pretraining

CLIP is pre-trained using large-scale text-image pairs and can be directly transmitted to imagenet to achieve zero classification without fine-tuning the image labels. The CLIP model could lead the development of CVs to an era of large-scale pre-learning and text-to-image interconnection.

## Zero-shot image classification method

### Image Classification
![517063291594231564](https://user-images.githubusercontent.com/113433202/189910758-3afa5aba-1a72-4192-80f4-92280838ba58.jpg)

### Image Search
![149300099319477992](https://user-images.githubusercontent.com/113433202/189911404-2098c560-b5e4-468f-b3f5-819a51876af3.jpg)

### CLIP training process
<img width="516" alt="image" src="https://user-images.githubusercontent.com/113433202/189938321-a5a4fb7d-235e-4995-b2ee-f43c7b166df8.png">

## Word2vec
Word2Vec is Word to Vector, a method for converting a word to a vector.

Word2Vec uses a neural network layer to map word vectors in one-hot encoded form to word vectors in distributed form.

The earliest word vector is generated using dummy coding (once). The dimension of the vector of words it generates is equal to the dimension of the entire dictionary. For each particular word in the dictionary, the corresponding position is set to 1. Representing vectors of words in this way is very simple, but there are many problems. There are millions of words in our vocabulary, and each word is represented by a millionth vector, which is a memory disaster and a huge amount of computation. And such a vector is actually 0, except for one position, and the rest of the positions are 0, which is very inefficient. The distributed view is trained to map each word into a shorter vector of words. All these word vectors make up a vector space, and the usual statistical methods can then be used to study the relationships between words. What is the dimension of this shorter vector of words? As a rule, this must be indicated to us during training.

For example, we map the word "king" from a space where a possibly very sparse vector resides to a space where this four-dimensional vector currently resides. This process is called word embedding, which means embedding high-dimensional word vectors into a low-dimensional space.

![152378920732813369](https://user-images.githubusercontent.com/113433202/189917719-ac0ec820-afa9-4fff-b6e1-4e3b40fab75b.jpg)

When we use word embedding, there can be some relationship between words, for example:
The vector of the word king minus the vector of the word man plus the vector of the word woman will equal the vector of the word queen!

![750438759072812142](https://user-images.githubusercontent.com/113433202/189918035-747c7409-8dc4-4ded-a687-d0083245fecd.jpg)

Find the top N most similar words. Positive words have a positive effect on similarity, negative words have a negative effect.

This method calculates the cosine similarity between the simple average projection weight vector of the given words and the vectors for each word in the model. The method matches the word and distance analogy scenarios in the original implementation of word2vec.

## Image Transformers（VIT）

![190114577871466194](https://user-images.githubusercontent.com/113433202/189919725-af340fac-b4c0-4bd5-b3d0-944185dd3c62.jpg)

![167499627567032482](https://user-images.githubusercontent.com/113433202/189919963-19dbd978-1e91-4595-9d85-8c30626e0e17.jpg)

![841184451968499208](https://user-images.githubusercontent.com/113433202/189920815-ff27b92f-1f9e-420a-99ce-4b77ce7c7d6e.jpg)

![126724004019842884](https://user-images.githubusercontent.com/113433202/189920944-70e55a43-ed87-4cf0-b3b2-ffd1bc39f41e.jpg)


## Train

The image is represented as a 3D tensor

Text as a vector

model = SentenceTransformer(...)

train_loss_sts = losses.CosineSimilarityLoss(model=model)

evaluator = EmbeddingSimilarityEvaluator.from_input_examples(samples, name='sts-dev')

![733293105174849872](https://user-images.githubusercontent.com/113433202/189920567-21a2634b-4e75-4303-81a8-25c1554415d5.jpg)

# Telegram bot

![69990157007312459](https://user-images.githubusercontent.com/113433202/189921372-51f8376b-b699-4570-a4b5-2e9288b6fd4d.jpg)

A cross-platform instant messaging system with VoIP functions that allows you to exchange text, voice and video messages, stickers and photos, files of many formats.
You can also make video and audio calls and broadcasts in channels and groups, organize conferences, multi-user groups and channels. With the help of bots, the functionality of the application is practically unlimited.

- Before starting development, the bot must be registered and its unique token must be obtained. For this, Telegram has a special bot - @BotFather.
- If successful, BotFather returns the bot token. Something like '5302769688:ABRr6kLe8Zg2l9ClaW5hayYMVin_5epr1ZZ'
- That's enough to get started.


