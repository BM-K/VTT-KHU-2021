import torch
from model.qle_memory import QuestionLevelDifficulty_M
from model.qle_logic import QuestionLevelDifficulty_L


class ModelProcessor():
    def __init__(self):
        super(ModelProcessor, self).__init__()
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        memory_ckpt = './model/memory_v3.pt'
        logic_ckpt = './model/logic_v3.pt'

        self.memory_model = QuestionLevelDifficulty_M()
        self.logic_model = QuestionLevelDifficulty_L()

        self.memory_model.load_state_dict(torch.load(memory_ckpt), strict=False)
        self.logic_model.load_state_dict(torch.load(logic_ckpt), strict=False)

        self.memory_model.to(self.device).eval()
        self.logic_model.to(self.device).eval()

    def run(self, img_inputs, text_inputs):
        memory_level = self.memory_model(text_inputs, img_inputs)
        logic_level = self.logic_model(text_inputs, img_inputs)

        return memory_level, logic_level
