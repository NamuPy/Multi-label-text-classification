import tensorflow as tf
import pandas as pd
import tensorflow_hub as hub
import os
import re
from keras import backend as K
import keras.layers as layers
from keras.models import Model, load_model
from keras.engine import Layer
import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import auc
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
# from keras.callbacks import Callback
from sklearn.model_selection import KFold
from collections import Counter

# from skmultilearn.problem_transform import LabelPowerset # pip3 install scikit-multilearn
# from imblearn.over_sampling import RandomOverSampler # pip3 install imblearn

import random
sess = tf.Session()
K.set_session(sess)

class ElmoEmbeddingLayer(Layer):
	def __init__(self, **kwargs):
		self.dimensions = 1024
		self.trainable=True
		super(ElmoEmbeddingLayer, self).__init__(**kwargs)	
	def build(self, input_shape):
		self.elmo = hub.Module('https://tfhub.dev/google/elmo/3', trainable=self.trainable,name="{}_module".format(self.name))
		self._trainable_weights += tf.compat.v1.trainable_variables(scope="^{}_module/.*".format(self.name))
		super(ElmoEmbeddingLayer, self).build(input_shape)	
	def call(self, x, mask=None):
		result = self.elmo(K.squeeze(K.cast(x, tf.string), axis=1),
					  as_dict=True,
					  signature='default',
					  )['default']
		return result	
	def compute_mask(self, inputs, mask=None):
		return K.not_equal(inputs, '--PAD--')	
	def compute_output_shape(self, input_shape):
		return (input_shape[0], self.dimensions)



def build_model(): 
	input_text = layers.Input(shape=(1,), dtype="string")
	embedding_lyr = ElmoEmbeddingLayer()(input_text)
	dense = layers.Dense(512, activation='relu')(embedding_lyr)
	## Output layer has size 12 because there are 12 classes in the OPP-115 dataset
	pred = layers.Dense(12, activation='softmax')(dense)
	model = Model(inputs=[input_text], outputs=pred)
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	model.summary()
	return model


if __name__ == "__main__":

	# Epochs
	E=10
	batch_size = 512
	threads = 10
	multi=True

	# Read dataset and we would like to remove the category 'Other'
	Dataset_ori= pd.read_csv('/storage/work/n/nmc5751/ELMo/OPP15/Cleaned_multilabel_dataset_upsampled.txt',sep='\t')
	del Dataset_ori['Other']
	Dataset_ori["Id"] = Dataset_ori['website'] + Dataset_ori['segment ID'].astype('str')

	labels_order = Dataset_ori.columns[3:-1]

	## Initializations to perform uniform distribution of websites into train-test (75-40) across 10 folds
	all_websites = Dataset_ori['website'].unique()
	web_dict = Counter(all_websites)
	selection_per_loop = dict()
	# lp = LabelPowerset()
	# ros = RandomOverSampler(random_state=42)

	for count in range(10):

		
		## Uniform split of websites into train-test (75-40)
		itemminValue = min(web_dict.items(), key=lambda x: x[1])

		min_val = itemminValue[1]
		minimum_webs = list()
		print(min_val)
		for key, value in web_dict.items():
			if value == min_val:
				minimum_webs.append(key)
		random.shuffle(minimum_webs)

		selected_test_webs = minimum_webs[0:40]

		while len(selected_test_webs)<40:
			min_val+=1
			minimum_webs = list()
			for key, value in web_dict.items():
				if value == min_val:
					minimum_webs.append(key)
			random.shuffle(minimum_webs)
			selected_test_webs = list(set(selected_test_webs)|set(minimum_webs[0:40-len(selected_test_webs)]))
			
		for tw in selected_test_webs:
			web_dict[tw] += 1
		selection_per_loop[count] = selected_test_webs

		train_df = Dataset_ori[~Dataset_ori['website'].isin(selected_test_webs)]
		test_df = Dataset_ori[Dataset_ori['website'].isin(selected_test_webs)]

		train_text_ori = train_df['alltext'].tolist()
		train_text_ori = [' '.join(t.split()) for t in train_text_ori]
		train_text_ori = np.array(train_text_ori, dtype=object)[:, np.newaxis]
		train_label_ori = train_df.values[:,3:-1]

		test_text = test_df['alltext'].tolist()
		test_text = [' '.join(t.split()) for t in test_text]
		test_text = np.array(test_text, dtype=object)[:, np.newaxis]
		test_label = test_df.values[:,3:-1]

		train_text=train_text_ori
		train_label = train_label_ori

		model = build_model()

		print('train X and Y',train_text.shape,train_label.shape)
		print('test X and Y',test_text.shape,test_label.shape)

		# Training and testing
		history = model.fit(train_text, train_label,validation_data=(test_text, test_label),epochs=E,batch_size=batch_size,use_multiprocessing=multi,workers=threads)

		training_res = pd.DataFrame(history.history['val_loss'],columns=['val_loss'])
		training_res['val_accuracy'] = history.history['val_accuracy']
		training_res['loss'] = history.history['loss']
		training_res['accuracy'] = history.history['accuracy']
		training_res['Epoch'] = training_res.index
		training_res['Fold'] = count

		if count==0:
			training_res.to_csv('.../results/training_metrics_per_epoch.csv',header=True,mode='a',sep='\t')
		else:
			training_res.to_csv('.../results/training_metrics_per_epoch.csv',header=False,mode='a',sep='\t')

		model.save('.../ElmoModel_Fold_{}.h5'.format(count))
		pre_save_preds = model.predict(test_text) 

		pred_binary = np.where(pre_save_preds>=0.5,1,0)

		result_scores = pd.DataFrame()

		## Validation for all 12 labels and storing results (recall, precision, f-1 scores) in a table
		for category in range(train_label.shape[1]):
			label_name = labels_order[category]

			lr_precision, lr_recall, _ = precision_recall_curve(test_label[:,category].astype('float'), pre_save_preds[:,category])
			lr_f1, lr_auc = f1_score(test_label[:,category].astype(int), pred_binary[:,category]), auc(lr_recall, lr_precision)

			if sum(test_label[:,category])==0 and sum(pred_binary[:,category])==0:
				tp=0
				fp=0
				tn=len(pred_binary[:,category])
				fn=0
			else:
				tn, fp, fn, tp = confusion_matrix(test_label[:,category].astype(int), pred_binary[:,category]).ravel()
		
			if tp+fp==0:
				precision = 0
			else:
				precision = float(tp)/float(tp+fp)		

			if tp+fn==0:
				recall = 0
			else:
				recall = float(tp)/float(tp+fn)
			
			df = pd.DataFrame([{'category':label_name,'tp':tp,'fp':fp,'tn':tn,'fn':fn,'recall':recall,'precision':precision,'f1':lr_f1,'auc':lr_auc}])

			result_scores = result_scores.append(df)

		if count==0:
			result_scores.to_csv('.../results/result_scores_ELMo.csv',header=True,mode='a',sep='\t')
		else:
			result_scores.to_csv('.../results/result_scores_ELMo.csv',header=False,mode='a',sep='\t')

		count=count+1

