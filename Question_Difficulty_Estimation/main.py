from data.preprocessing import DataPreprocessing
from model.processor import ModelProcessor


def main():
    """
        DataPreprocessing()
            - 질의, 자막, 그리고 비디오 이미지 데이터를 얻고, 모델의 입력으로 들어갈 수 있는 형태의 Tensor로 변환합니다.
        ModelProcessor()
            - 입력을 처리하여 결과를 도출합니다.
    """

    DataClass = DataPreprocessing()
    ModelClass = ModelProcessor()

    questions = ["How did Haeyoung1 feel such surreal emotions?", "What does Haeyoung1 feel?"]
    vids = ["AnotherMissOh18_001_0000", "AnotherMissOh18_020_0429"]

    for question, vid in zip(questions, vids):
        text_inputs, img_inputs = DataClass.preprocessing(question, vid)
        memory_level, logic_level = ModelClass.run(img_inputs, text_inputs)

        print(f"{question}:\n  Memory Level {memory_level} | Logic Level {logic_level} \n")


if __name__ == '__main__':
    main()
