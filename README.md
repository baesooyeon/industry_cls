
# Industry_cls
통계데이터 인공지능 활용대회

## Description
>2022.03.21 ~ 2022.04.13 (약 3주)
- 통계데이터 인공지능 활용 대회
- 산업 설명 텍스트를 이용하여 "산업 분류 자동화 모델"을 개발하는 대회


## Files
- `dataset2.py`    
  데이터를 전처리, train/test split, upsampling 해주며, 모델의 input 형식에 알맞게 변환해줍니다.

- `inference2.py`   
  Output값을 통해 정답 파일을 도출합니다.
  
- `load.py`   
  Pretrained language model과 Tokenizer를 불러옵니다.
  
- `loss.py`      
  CrossEntropy, FocalCrossEntropy를 정의한 모듈입니다.
  
- `network2.py`   
  개별 언어 모델을 Text Classification 과제에 Fine Tuning하기 위해 nn.Module 형태로 정의한 모듈입니다.
  
- `spell_check.py`   
  데이터 전처리 중 한스펠을 이용해 맞춤법 및 띄어쓰기를 교정하며, 특수문자를 제거합니다.   
  
- `train2.py`   
  학습을 진행합니다.
  
- `utils2.py`   
  디렉토리 생성, 모델 저장 path 생성, 모델 저장, 그래프 저장 등을 해줍니다.


## Usage
**Example**
<pre><code>'python train2.py --root=${ROOT} 
                  --model=${MODEL} 
                  --loss=${LOSS} 
                  --epoch=${EPOCH} 
                  --batch-size=${BATCH} 
                  --optimizer=${OPTIMIZER} 
                  --beta1=${BETA1} 
                  --upsample=${UPSAMPLE} 
                  --minimum=${MINIMUM} 
                  --n-layers=${N_LAYERS} 
                  --num-test=${NUM_TEST} 
                  --learning-rate=${LR} 
                  --dr-rate=${DR_RATE}  
                  --device=${DEVICE} 
                  --lr-scheduler=${SCHEDULER} 
                  --patience=${PATIENCE}'</code></pre>
