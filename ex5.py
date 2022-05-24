from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize.casual import casual_tokenize
import pandas as pd
from sklearn.decomposition import PCA
import numpy as np

#read corpus
f1 = open("YorkAustralia.txt", "r")
corpus= f1.readlines()
f2 = open("NewYork.txt", "r")
corpus.append(f2.readlines()[0])
corpus.append("")
print("step 1 = ", corpus)
# corpus.append("York is a cathedral city with Roman origins at the confluence of the rivers Ouse and Foss in North Yorkshire, England. It is the historic county town of Yorkshire. The city has long-standing buildings and structures, such as a minster, castle and city walls. The city was founded as Eboracum in 71 AD. It became the capital of the Roman province of Britannia Inferior, and later of the kingdoms of Deira, Northumbria and Jórvík. In the Middle Ages, the northern England ecclesiastical province's centre and grew as a wool-trading centre. In the 19th century, it became a major railway network hub and confectionery manufacturing centre. During the Second World War, part of the Baedeker Blitz bombed the city; it was less affected by the war than other northern cities, with several historic buildings being gutted and restored upto the 1960s.")


#2. make TF-IDF vector
tfidf = TfidfVectorizer(tokenizer=casual_tokenize)
tfidf_docs = pd.DataFrame(tfidf.fit_transform(raw_documents=corpus).toarray())
id_words = [(i, w) for (w, i) in tfidf.vocabulary_.items()]
tfidf_docs.columns = list(zip(*sorted(id_words)))[1]
print("step 2 = ",tfidf_docs)

#3. do 1 topic
pca = PCA(n_components=2)
pca = pca.fit(tfidf_docs)
pca_topic_vectors = pca.transform(tfidf_docs)
columns = ['topic{}'.format(i) for i in range(pca.n_components)]
index = ['York', 'New York']
pca_topic_vectors = pd.DataFrame(pca_topic_vectors, columns=columns, index=index)
print("step 3 = \n ",pca_topic_vectors.round(3).head(6))

# now do the querry:
print("step 4 = ")
Q = ["What is biggest city?"]
tfidf_Q = pd.DataFrame(tfidf.transform(raw_documents=Q).toarray())
#print("tfidf_Q = ", tfidf_Q)
pca_Q = pca.transform(tfidf_Q)
print("pca_Q = ", pca_Q)
#pca_topic_vectors.iloc[:,0]
print(pca_topic_vectors)
# print("distance to each topic = ", list(map(lambda x: abs(x[1]-pca_Q[0]), pca_topic_vectors.items())))
print("distance to each topic = ", list(map(lambda x: abs(x-pca_Q[0]), pca_topic_vectors.iloc[:,0])))
print(np.linalg.norm(list(map(lambda x: abs(x-pca_Q[0]), pca_topic_vectors.iloc[:,0]))))
# [
#    0.0120506  ,-0.0076749  , 0.02423401, -0.02110332, -0.02744004,  0.00393172,
#   -0.00848137 , 0.00034783 ,-0.01591566,  0.01505379,  0.01323104,  0.01849282,
#   -0.02040643 , 0.03412115 , 0.03840143,  0.05729189, -0.03869846,  0.01120091,
#    0.01260541 , 0.01348758 ,-0.02829763, -0.0109276 ,  0.00835944,  0.00938827,
#   -0.0045032  , 0.00952881 ,-0.01441151, -0.00197061, -0.0064818 , -0.02355901,
#   -0.00646157 ,-0.00214335 ,-0.03403759,  0.01846899,  0.02523388, -0.00783765,
#   -0.02487168 , 0.0073294  ,-0.02798893, -0.02192862, -0.01479432, -0.01559148,
#   -0.03780337 , 0.0359661  ,-0.03300197, -0.00464275, -0.01945396,  0.04503846,
#    0.03984885 ,-0.01292448 , 0.01884679,  0.02876027, -0.02198624,  0.02251153,
#    0.00110005 ,-0.00455072 , 0.02516813,  0.00793256,  0.00315264,  0.00837508,
#   -0.02282014 , 0.00420448 , 0.01226936, -0.00542777,  0.00620779,  0.00128378,
#    0.01122374 , 0.04036514 , 0.00974766, -0.03893959, -0.00588493, -0.00734235,
#    0.00616083 ,-0.01815706 ,-0.00398844,  0.01975237,  0.0021465 , -0.02232426,
#   -0.02029232 , 0.03379174 ,-0.02866245, -0.01454126, -0.00766867, -0.0113131,
#    0.00048093 , 0.00697294 ,-0.03522912,  0.01196897,  0.04672287,  0.01886405,
#    0.01213017 ,-0.04845661 , 0.00101   ,  0.03017406,  0.0152389 ,  0.00644608,
#    0.01499764 ,-0.01401259 ,-0.00760349,  0.00868344, -0.0235829 , -0.01549412,
#    0.01554992 , 0.016942   , 0.01523524,  0.00370882, -0.0007553 ,  0.02182341,
#   -0.01336438 ,-0.0130863  ,-0.00180709,  0.01759415, -0.01572797, -0.00451619,
#    0.01318925 ,-0.03441588 ,-0.03285336,  0.01983181,  0.00147151, -0.00641324,
#   -0.0358176  ,-0.00952742 ,-0.02428837,  0.01185358, -0.00454745, -0.0324017,
#   -0.04249225 ,-0.03363195 ,-0.00349145,  0.00338066,  0.01366382, -0.010159,
#   -0.00532971 , 0.02034615 ,-0.0106675 ,  0.0248752 , -0.00701822, -0.0278922,
#    0.02370826 ,-0.02502838 , 0.01323128,  0.00526964,  0.00439748, -0.00851288,
#   -0.0016264  , 0.0104956  , 0.0086681 , -0.0135031 ,  0.03093598, -0.00902461,
#    0.00846741 ,-0.01272146 , 0.00389073,  0.01442243,  0.01945405,  0.01035334,
#    0.00059027 , 0.01454794 , 0.01268528,  0.02047189,  0.00010013,  0.00690517,
#   -0.01695504 , 0.0094431  , 0.00469897, -0.00545841, -0.00414109, -0.00848112,
#    0.00745611 ,-0.00550143 ,-0.00374815, -0.00253129, -0.00131303, -0.00473575,
#    0.00421192 , 0.004505   , 0.00198576, -0.00411352,  0.00132741]