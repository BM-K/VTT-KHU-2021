# Video Turing Test 2021

## Question Difficulty Estimation
입력 질문에 대한 난이도를 측정하는 모델입니다.

### How to use
  1. 다음과 같이 clone을 해주세요.
  ```
  git clone https://github.com/khu-nlplab/VTT-KHU-2021.git
  cd Question_Difficulty_Estimation
  pip install -r requirements.txt
  ```
  2. [Google Drive](https://drive.google.com/drive/u/1/folders/15SUdNCiw_Q1Bmh_CksodztG8rW2XMvpv)에서 h5 파일과, pt파일을 다운 받은 후 아래와 같이 디렉토리를 구성해주세요.
  ```
  Question_Difficulty_Estimation/
      main.py
      requirements.txt
      
      model/
        processor.py
        qle_logic.py
        qle_memory.py
        memory.pt ✔
        logic.pt  ✔

      data/
        preprocessing.py
        MissO_vid2image.h5  ✔
        MissO_vid2sub.h5  ✔
  ```
  3. output은 다음과 같습니다.
  ```
  python main.py
  
  How did Haeyoung1 feel such surreal emotions?:
    Memory Level 3 | Logic Level 3 
  ```
