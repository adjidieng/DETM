import scipy.io 
import matplotlib.pyplot as plt 
import data 
import pickle 
import numpy as np 

beta = scipy.io.loadmat('./beta_100.mat')['values'] ## K x T x V
print('beta: ', beta.shape)

with open('un/min_df_100/timestamps.pkl', 'rb') as f:
    timelist = pickle.load(f)
print('timelist: ', timelist)
T = len(timelist)
ticks = [str(x) for x in timelist]
print('ticks: ', ticks)

## get vocab
data_file = 'un/min_df_100'
vocab, train, valid, test = data.get_data(data_file, temporal=True)
vocab_size = len(vocab)

## plot topics 
num_words = 10
times = [0, 10, 40]
num_topics = 50
for k in range(num_topics):
    for t in times:
        gamma = beta[k, t, :]
        top_words = list(gamma.argsort()[-num_words+1:][::-1])
        topic_words = [vocab[a] for a in top_words]
        print('Topic {} .. Time: {} ===> {}'.format(k, t, topic_words)) 

print('Topic Climate Change...')
num_words = 10
for t in range(46):
    gamma = beta[46, t, :]
    top_words = list(gamma.argsort()[-num_words+1:][::-1])
    topic_words = [vocab[a] for a in top_words]
    print('Time: {} ===> {}'.format(t, topic_words)) 

fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(18, 9), dpi=80, facecolor='w', edgecolor='k')
ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8 = axes.flatten()
ticks = [str(x) for x in timelist]
#plt.xticks(np.arange(T)[0::10], timelist[0::10])

words_1 = ['vietnam', 'war', 'pakistan', 'indonesia']
tokens_1 = [vocab.index(w) for w in words_1]
betas_1 = [beta[1, :, x] for x in tokens_1]
for i, comp in enumerate(betas_1):
    ax1.plot(range(T), comp, label=words_1[i], lw=2, linestyle='--', marker='o', markersize=4)
ax1.legend(frameon=False)
print('np.arange(T)[0::10]: ', np.arange(T)[0::10])
ax1.set_xticks(np.arange(T)[0::10])
ax1.set_xticklabels(timelist[0::10])
ax1.set_title('Topic "Southeast Asia"', fontsize=12)


words_5 = ['health', 'education', 'hunger', 'terrorism', 'water']
tokens_5 = [vocab.index(w) for w in words_5]
betas_5 = [beta[5, :, x] for x in tokens_5]
for i, comp in enumerate(betas_5):
    ax2.plot(comp, label=words_5[i], lw=2, linestyle='--', marker='o', markersize=4)
ax2.legend(frameon=False)
ax2.set_xticks(np.arange(T)[0::10])
ax2.set_xticklabels(timelist[0::10])
ax2.set_title('Topic "Poverty & Development"', fontsize=12)


words_11 = ['iran', 'iraq', 'imperialism']
tokens_11 = [vocab.index(w) for w in words_11]
betas_11 = [beta[11, :, x] for x in tokens_11]
for i, comp in enumerate(betas_11):
    ax3.plot(comp, label=words_11[i], lw=2, linestyle='--', marker='o', markersize=4)
ax3.legend(frameon=False)
ax3.set_xticks(np.arange(T)[0::10])
ax3.set_xticklabels(timelist[0::10])
ax3.set_title('Topic "War"', fontsize=12)


words_13 = ['iran', 'treaty', 'trade', 'race', 'nonproliferation']
tokens_13 = [vocab.index(w) for w in words_13]
betas_13 = [beta[13, :, x] for x in tokens_13]
for i, comp in enumerate(betas_13):
    ax4.plot(comp, label=words_13[i], lw=2, linestyle='--', marker='o', markersize=4)
ax4.legend(frameon=False)
ax4.set_xticks(np.arange(T)[0::10])
ax4.set_xticklabels(timelist[0::10])
ax4.set_title('Topic "Nuclear Weapons"', fontsize=12)


#words_28 = ['men', 'equality', 'gender', 'female', 'education']
words_28 = ['education', 'gender', 'equality']
tokens_28 = [vocab.index(w) for w in words_28]
betas_28 = [beta[28, :, x] for x in tokens_28]
for i, comp in enumerate(betas_28):
    ax5.plot(comp, label=words_28[i], lw=2, linestyle='--', marker='o', markersize=4)
ax5.legend(frameon=False)
ax5.set_xticks(np.arange(T)[0::10])
ax5.set_xticklabels(timelist[0::10])
ax5.set_title('Topic "Human Rights"', fontsize=12)


words_30 = ['exploitation', 'legal', 'rules', 'negotiations']
tokens_30 = [vocab.index(w) for w in words_30]
betas_30 = [beta[30, :, x] for x in tokens_30]
for i, comp in enumerate(betas_30):
    ax6.plot(comp, label=words_30[i], lw=2, linestyle='--', marker='o', markersize=4)
ax6.legend(frameon=False)
ax6.set_xticks(np.arange(T)[0::10])
ax6.set_xticklabels(timelist[0::10])
ax6.set_title('Topic "Ocean Exploitation"', fontsize=12)


words_46 = ['ozone', 'warming', 'emissions', 'waste']
tokens_46 = [vocab.index(w) for w in words_46]
betas_46 = [beta[46, :, x] for x in tokens_46]
for i, comp in enumerate(betas_46):
    ax7.plot(comp, label=words_46[i], lw=2, linestyle='--', marker='o', markersize=4)
ax7.legend(frameon=False)
ax7.set_xticks(np.arange(T)[0::10])
ax7.set_xticklabels(timelist[0::10])
ax7.set_title('Topic "Climate Change"', fontsize=12)


words_49 = ['apartheid', 'independence', 'colonial', 'democratic']
tokens_49 = [vocab.index(w) for w in words_49]
betas_49 = [beta[49, :, x] for x in tokens_49]
for i, comp in enumerate(betas_49):
    ax8.plot(comp, label=words_49[i], lw=2, linestyle='--', marker='o', markersize=4)
ax8.legend(frameon=False)
ax8.set_title('Topic "Africa"', fontsize=12)
ax8.set_xticks(np.arange(T)[0::10])
ax8.set_xticklabels(timelist[0::10])
plt.savefig('word_evolution.png')
plt.show()
