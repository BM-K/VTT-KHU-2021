import h5py
import copy
import torch
from transformers import RobertaTokenizer


class DataPreprocessing():
    def __init__(self):
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        """
        init token, idx = <s>, 0
        pad token, idx = <pad>, 1
        sep token, idx = </s>, 2
        unk token, idx = <unk>, 3
        """
        self.pad_token = self.tokenizer.pad_token
        self.unk_token = self.tokenizer.unk_token
        self.init_token = self.tokenizer.cls_token
        self.sep_token = self.tokenizer.sep_token

        self.pad_token_idx = self.tokenizer.convert_tokens_to_ids(self.pad_token)
        self.unk_token_idx = self.tokenizer.convert_tokens_to_ids(self.unk_token)
        self.init_token_idx = self.tokenizer.convert_tokens_to_ids(self.init_token)
        self.sep_token_idx = self.tokenizer.convert_tokens_to_ids(self.sep_token)

        self.vid2sub_h5 = './data/MissO_vid2sub_v3.h5'
        self.vid2image_h5 = './data/MissO_vid2image_v3.h5'

        self.max_len = 512
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    def preprocessing(self, question, vid):
        image_inputs, subtitle = self.get_image(vid), self.get_subtitle(vid)

        text_inputs = self.get_text(question, subtitle)

        return text_inputs, image_inputs

    def get_image(self, vid):

        hq_file = h5py.File(self.vid2image_h5, 'r')

        stack_images = hq_file[str(vid)][:]

        stack_images = [img for step, img in enumerate(stack_images) if step % 3 == 0]

        image_inputs = [torch.tensor(stack_images).to(self.device)]

        hq_file.close()

        return image_inputs

    def get_subtitle(self, vid):

        hq_file = h5py.File(self.vid2sub_h5, 'r')

        subtitle = hq_file[str(vid)][()]

        hq_file.close()

        return str(subtitle)

    def get_text(self, question, subtitle):
        question_tokens = [self.init_token_idx] + \
                          self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(question)) + \
                          [self.sep_token_idx]

        if len(subtitle) == 0:
            que_utterance_tokens = question_tokens[:-1]
        else:
            que_utterance_tokens = question_tokens + [
                self.init_token_idx] + self.tokenizer.convert_tokens_to_ids(
                self.tokenizer.tokenize(subtitle)) + [self.sep_token_idx]

        if len(que_utterance_tokens) > self.max_len:
            que_utterance_tokens = que_utterance_tokens[:self.max_len - 1]
            que_utterance_tokens += [self.sep_token_idx]

        que_utterance = copy.deepcopy(que_utterance_tokens)

        for i in range(self.max_len - len(que_utterance_tokens)): que_utterance.append(self.pad_token_idx)

        attention_mask = self.get_bert_parameter(que_utterance,
                                                  self.pad_token_idx)

        text_inputs = {'input_ids': torch.tensor(que_utterance).unsqueeze(0).to(self.device),
                       'attention_mask': attention_mask.unsqueeze(0).to(self.device)}

        return text_inputs

    def get_bert_parameter(self, inputs, pad_token_idx):

        pad_index = [idx for idx, value in enumerate(inputs) if value == pad_token_idx]

        if len(pad_index) == 0:
            attention_mask = [1] * self.max_len
        else:
            attention_mask = ([1] * pad_index[0]) + ([0] * (self.max_len - pad_index[0]))

        attention_mask = torch.tensor(attention_mask)

        return attention_mask.float()
