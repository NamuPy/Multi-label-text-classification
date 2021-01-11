import tensorflow as tf
import pandas as pd
import tensorflow_hub as hub
import os
import re
import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import auc
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import KFold
from collections import Counter

from bert.tokenization import FullTokenizer
from tqdm import tqdm_notebook
from tensorflow.keras import backend as K
import random

from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.optimizers import Adam

from skmultilearn.problem_transform import LabelPowerset # pip3 install scikit-multilearn
from imblearn.over_sampling import RandomOverSampler # pip3 install imblearn

# Initialize session
sess = tf.Session()

# Params for bert model and tokenization
bert_path = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"
out_size = 768
max_seq_length = 512

class PaddingInputExample(object):
	"""Fake example so the num input examples is a multiple of the batch size.
	When running eval/predict on the TPU, we need to pad the number of examples
	to be a multiple of the batch size, because the TPU requires a fixed batch
	size. The alternative is to drop the last batch, which is bad because it means
	the entire output data won't be generated.
	We use this class instead of `None` because treating `None` as padding
	battches could cause silent errors.
	"""

class InputExample(object):
	"""A single training/test example for simple sequence classification."""

	def __init__(self, guid, text_a, text_b=None, label=None):
		"""Constructs a InputExample.
	Args:
	  guid: Unique id for the example.
	  text_a: string. The untokenized text of the first sequence. For single
		sequence tasks, only this sequence must be specified.
	  text_b: (Optional) string. The untokenized text of the second sequence.
		Only must be specified for sequence pair tasks.
	  label: (Optional) string. The label of the example. This should be
		specified for train and dev examples, but not for test examples.
	"""
		self.guid = guid
		self.text_a = text_a
		self.text_b = text_b
		self.label = label

def create_tokenizer_from_hub_module():
	"""Get the vocab file and casing info from the Hub module."""
	bert_module =  hub.Module(bert_path)
	tokenization_info = bert_module(signature="tokenization_info", as_dict=True)
	vocab_file, do_lower_case = sess.run(
		[
			tokenization_info["vocab_file"],
			tokenization_info["do_lower_case"],
		]
	)

	return FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)

def convert_single_example(tokenizer, example, max_seq_length=256):
	"""Converts a single `InputExample` into a single `InputFeatures`."""

	if isinstance(example, PaddingInputExample):
		input_ids = [0] * max_seq_length
		input_mask = [0] * max_seq_length
		segment_ids = [0] * max_seq_length
		label = 0
		return input_ids, input_mask, segment_ids, label

	tokens_a = tokenizer.tokenize(example.text_a)
	if len(tokens_a) > max_seq_length - 2:
		tokens_a = tokens_a[0 : (max_seq_length - 2)]

	tokens = []
	segment_ids = []
	tokens.append("[CLS]")
	segment_ids.append(0)
	for token in tokens_a:
		tokens.append(token)
		segment_ids.append(0)
	tokens.append("[SEP]")
	segment_ids.append(0)

	input_ids = tokenizer.convert_tokens_to_ids(tokens)

	# The mask has 1 for real tokens and 0 for padding tokens. Only real
	# tokens are attended to.
	input_mask = [1] * len(input_ids)

	# Zero-pad up to the sequence length.
	while len(input_ids) < max_seq_length:
		input_ids.append(0)
		input_mask.append(0)
		segment_ids.append(0)

	assert len(input_ids) == max_seq_length
	assert len(input_mask) == max_seq_length
	assert len(segment_ids) == max_seq_length

	return input_ids, input_mask, segment_ids, example.label

def convert_examples_to_features(tokenizer, examples, max_seq_length=256):
	"""Convert a set of `InputExample`s to a list of `InputFeatures`."""

	input_ids, input_masks, segment_ids, labels = [], [], [], []
	for example in tqdm_notebook(examples, desc="Converting examples to features"):
		input_id, input_mask, segment_id, label = convert_single_example(
			tokenizer, example, max_seq_length
		)
		input_ids.append(input_id)
		input_masks.append(input_mask)
		segment_ids.append(segment_id)
		labels.append(label)
	return (
		np.array(input_ids),
		np.array(input_masks),
		np.array(segment_ids),
		np.array(labels)#.reshape(-1, 1)
		,
	)

def convert_text_to_examples(texts, labels):
	"""Create InputExamples"""
	InputExamples = []
	for text, label in zip(texts, labels):
		InputExamples.append(
			InputExample(guid=None, text_a=" ".join(text), text_b=None, label=label)
		)
	return InputExamples


class BertLayer(tf.keras.layers.Layer):
	def __init__(
		self,
		n_fine_tune_layers=10,
		pooling="first",
		bert_path=bert_path,#"https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1",
		**kwargs,
	):
		self.n_fine_tune_layers = n_fine_tune_layers
		self.trainable = True
		self.output_size = out_size
		self.pooling = pooling
		self.bert_path = bert_path
		if self.pooling not in ["first", "mean"]:
			raise NameError(
				f"Undefined pooling type (must be either first or mean, but is {self.pooling}"
			)

		super(BertLayer, self).__init__(**kwargs)

	def build(self, input_shape):
		self.bert = hub.Module(
			self.bert_path, trainable=self.trainable, name=f"{self.name}_module"
		)

		# Remove unused layers
		trainable_vars = self.bert.variables
		if self.pooling == "first":
			trainable_vars = [var for var in trainable_vars if not "/cls/" in var.name]
			trainable_layers = ["pooler/dense"]

		elif self.pooling == "mean":
			trainable_vars = [
				var
				for var in trainable_vars
				if not "/cls/" in var.name and not "/pooler/" in var.name
			]
			trainable_layers = []
		else:
			raise NameError(
				f"Undefined pooling type (must be either first or mean, but is {self.pooling}"
			)

		# Select how many layers to fine tune
		for i in range(self.n_fine_tune_layers):
			trainable_layers.append(f"encoder/layer_{str(11 - i)}")

		# Update trainable vars to contain only the specified layers
		trainable_vars = [
			var
			for var in trainable_vars
			if any([l in var.name for l in trainable_layers])
		]

		# Add to trainable weights
		for var in trainable_vars:
			self._trainable_weights.append(var)

		for var in self.bert.variables:
			if var not in self._trainable_weights:
				self._non_trainable_weights.append(var)

		super(BertLayer, self).build(input_shape)

	def call(self, inputs):
		inputs = [K.cast(x, dtype="int32") for x in inputs]
		input_ids, input_mask, segment_ids = inputs
		bert_inputs = dict(
			input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids
		)
		if self.pooling == "first":
			pooled = self.bert(inputs=bert_inputs, signature="tokens", as_dict=True)[
				"pooled_output"
			]
		elif self.pooling == "mean":
			result = self.bert(inputs=bert_inputs, signature="tokens", as_dict=True)[
				"sequence_output"
			]

			mul_mask = lambda x, m: x * tf.expand_dims(m, axis=-1)
			masked_reduce_mean = lambda x, m: tf.reduce_sum(mul_mask(x, m), axis=1) / (
					tf.reduce_sum(m, axis=1, keepdims=True) + 1e-10)
			input_mask = tf.cast(input_mask, tf.float32)
			pooled = masked_reduce_mean(result, input_mask)
		else:
			raise NameError(f"Undefined pooling type (must be either first or mean, but is {self.pooling}")

		return pooled

	def compute_output_shape(self, input_shape):
		return (input_shape[0], self.output_size)


# Build model
def build_model(max_seq_length): 
	in_id = tf.keras.layers.Input(shape=(max_seq_length,), name="input_ids")
	in_mask = tf.keras.layers.Input(shape=(max_seq_length,), name="input_masks")
	in_segment = tf.keras.layers.Input(shape=(max_seq_length,), name="segment_ids")
	bert_inputs = [in_id, in_mask, in_segment]

	bert_output = BertLayer(n_fine_tune_layers=10, pooling="first")(bert_inputs)
	dense = tf.keras.layers.Dense(512, activation='relu')(bert_output)
	pred = tf.keras.layers.Dense(12, activation='softmax')(dense)

	model = tf.keras.models.Model(inputs=bert_inputs, outputs=pred)
	
	optimizer = Adam(learning_rate=5e-05,epsilon=1e-08,decay=0.01,clipnorm=1.0)

	loss = CategoricalCrossentropy(from_logits = True)
	metric = CategoricalAccuracy('accuracy')

	model.compile(loss=loss, optimizer=optimizer, metrics=[metric])#['accuracy'])
	model.summary()

	return model

def initialize_vars(sess):
	sess.run(tf.local_variables_initializer())
	sess.run(tf.global_variables_initializer())
	sess.run(tf.tables_initializer())
	K.set_session(sess)




if __name__ == "__main__":

	#Epochs
	E=10
	batch_size = 256
	threads = 10
	multi=True

	direc = '/path_to_result_directory/results/'

	Dataset_ori= pd.read_csv('/storage/work/n/nmc5751/ELMo/OPP15/Cleaned_multilabel_dataset.txt',sep='\t')
	del Dataset_ori['Other']
	Dataset_ori["Id"] = Dataset_ori['website'] + Dataset_ori['segment ID'].astype('str')

	labels_order = Dataset_ori.columns[3:-1]

	all_websites = Dataset_ori['website'].unique()

	web_dict = Counter(all_websites)
	selection_per_loop = dict()
	lp = LabelPowerset()
	ros = RandomOverSampler(random_state=42)

	for count in range(10):

		print('Fold',count)

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
			# print('final len of test sites',len(selected_test_webs))
			
		for tw in selected_test_webs:
			web_dict[tw] += 1
		selection_per_loop[count] = selected_test_webs

		train_df = Dataset_ori[~Dataset_ori['website'].isin(selected_test_webs)]
		test_df = Dataset_ori[Dataset_ori['website'].isin(selected_test_webs)]

		train_df=train_df.sample(frac=fraction)
		test_df=test_df.sample(frac=fraction)

		train_text_ori = train_df['alltext'].tolist()
		train_text_ori = [' '.join(t.split()) for t in train_text_ori]
		train_text_ori = np.array(train_text_ori, dtype=object)[:, np.newaxis]
		train_label_ori = train_df.values[:,3:-1]

		test_text = test_df['alltext'].tolist()
		test_text = [' '.join(t.split()) for t in test_text]
		test_text = np.array(test_text, dtype=object)[:, np.newaxis]
		test_label = test_df.values[:,3:-1]

		
		## Upsampling of data for each label
		print('Before upsampling: ',train_text_ori.shape,train_label_ori.shape,test_text.shape,test_label.shape)

		yt = lp.transform(train_label_ori.astype('int'))
		train_text, y_resampled = ros.fit_sample(train_text_ori.astype('str'), yt)

		train_label = lp.inverse_transform(y_resampled).toarray()

		# train_text=train_text_ori
		# train_label = train_label_ori

		print('After Up-sampling',train_text.shape,train_label.shape,test_text.shape,test_label.shape)


		# Instantiate tokenizer
		tokenizer = create_tokenizer_from_hub_module()

		# Convert data to InputExample format
		train_examples = convert_text_to_examples(train_text, train_label)
		test_examples = convert_text_to_examples(test_text, test_label)

		# Convert to features
		(train_input_ids, train_input_masks, train_segment_ids, train_labels 
		) = convert_examples_to_features(tokenizer, train_examples, max_seq_length=max_seq_length)

		(test_input_ids, test_input_masks, test_segment_ids, test_labels
		) = convert_examples_to_features(tokenizer, test_examples, max_seq_length=max_seq_length)

		model = build_model(max_seq_length)

		# Instantiate variables
		initialize_vars(sess)


		history = model.fit([train_input_ids, train_input_masks, train_segment_ids], train_labels,
			validation_data=([test_input_ids, test_input_masks, test_segment_ids], test_labels)
			,epochs=E,batch_size=batch_size,use_multiprocessing=multi,workers=threads)

		# print(history.history)
		training_res = pd.DataFrame(history.history['val_loss'],columns=['val_loss'])
		training_res['val_accuracy'] = history.history['val_accuracy']
		training_res['loss'] = history.history['loss']
		training_res['accuracy'] = history.history['accuracy']
		training_res['Epoch'] = training_res.index
		training_res['Fold'] = count

		if count==0:
			training_res.to_csv(direc + 'training_metrics_per_epoch.csv',header=True,mode='a',sep='\t')
		else:
			training_res.to_csv(direc + 'training_metrics_per_epoch.csv',header=False,mode='a',sep='\t')

		
		pre_save_preds = model.predict([test_input_ids, test_input_masks,test_segment_ids]) 

		pred_binary = np.where(pre_save_preds>=0.5,1,0)


		result_scores = pd.DataFrame()

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
			# print(df)
			result_scores = result_scores.append(df)

		if count==0:
			result_scores.to_csv(direc + 'result_scores_OPP115_BERT.csv',header=True,mode='a',sep='\t')
		else:
			result_scores.to_csv(direc + 'result_scores_OPP115_BERT.csv',header=False,mode='a',sep='\t')

		count=count+1

