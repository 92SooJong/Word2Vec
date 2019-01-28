import gensim # Word2Vec
from konlpy.tag import Okt # 품사태깅



# 파일을 읽어와 한줄씩 data에 저장한다.
def read_data(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        data = [line.split('\t') for line in f.read().splitlines()]
        data = data[1:]  # header 제외
    return data


# 구분된 단어에 '/'를 붙이고, 그뒤에 품사를 태깅한다.
def tokenize(sentence):
    tagged_words = []
    pos_tagger = Okt()
    for t in pos_tagger.pos(sentence, norm=True, stem=True):
        tagged_words.append('/'.join(t))
    return tagged_words


vector_size = 300
learning_rate = 0.5
epoch = 300
cur_learning_rate = learning_rate/float(epoch)
print("current_learning_rate {}:".format(cur_learning_rate))


train_data = read_data("./ratings_train.txt")
# 데이터 수 조절
train_data = train_data[:-149900]

# 품사태깅
tokens = [tokenize(row[1]) for row in train_data]


# word2vec 학습 옵션 설정
# "size"옵션을 통해 단어의 vector 길이 결정
# 'sg' Skip-Gram 사용 여부
word2vec_params = { 'sg': 1, "size": vector_size, "alpha": learning_rate, "min_alpha": 0.001, 'seed': 1234}
model = gensim.models.Word2Vec(**word2vec_params)
model.build_vocab(tokens)

# 학습시작
for i in range(epoch):
    model.train(tokens, total_examples=model.corpus_count, epochs=model.epochs, compute_loss=True)
    print("#epoch {}: #loss = {}".format(i, model.get_latest_training_loss()))
    print("#rate  {}: ".format(model.alpha))
    model.alpha -= cur_learning_rate
    if i%100 == 0 and i != 0 :
        print("{} step model save!".format(i))

#결과값 저장
model.wv.save_word2vec_format('./word2vec.txt', binary=False)

    


